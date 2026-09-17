"""real-time OpenCV preview using the verified D2E vLLM path.

This prototype intentionally displays raw D2E event text. It does not require
the OWA packages and does not write MCAP. Press ``q`` in the preview to stop.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import cv2
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.internvl.processing_internvl import (
    InternVLProcessor as HFInternVLProcessor,
)
from vllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "open-world-agents/checkpoints/cyberpunk_trial_best"
DEFAULT_VIDEO = ROOT.parent / "open-world-agents/data/cyberpunk_test_trial2/trial2.mkv"

IMAGE_BLOCK = "<img>" + "<IMG_CONTEXT>" * 256 + "</img>"
TIMESTAMP_UNIT_NS = 10_000_000  # 10 ms
# This checkpoint uses the factorized event vocabulary (<VK_x>, <MB_x>, and
# separate sign tokens). Its three timestamp digits wrap every 10 seconds.
TIMESTAMP_BASES = (10, 10, 10)
TIMESTAMP_RANGE_NS = 10_000_000_000
_REAL_PROMPT_MAX_LENGTH = 8192


class D2EInternVLProcessor(HFInternVLProcessor):
    """Raise the text limit only for accumulated real-image requests.

    vLLM's startup profiler sends one synthetic 10000x10000 image and is
    sensitive to tokenizer kwargs supplied through mm_processor_kwargs. Real
    D2E requests need a larger ceiling once their image count grows beyond the
    processor's implicit 4096-token truncation limit.
    """

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        image_count = len(images) if isinstance(images, (list, tuple)) else int(images is not None)
        if image_count > 1:
            # InternVLProcessor validates modality-specific kwargs strictly;
            # tokenizer options must be nested under text_kwargs rather than
            # supplied as a top-level processor kwarg.
            text_kwargs = dict(kwargs.get("text_kwargs") or {})
            text_kwargs.setdefault("max_length", _REAL_PROMPT_MAX_LENGTH)
            kwargs["text_kwargs"] = text_kwargs
        return super().__call__(images=images, text=text, videos=videos, **kwargs)


def patch_vllm_transformers_internvl_processor(max_length: int) -> None:
    """Use the native HF InternVL processor expected by this checkpoint"""
    global _REAL_PROMPT_MAX_LENGTH
    from vllm.transformers_utils import processors as vllm_processors

    _REAL_PROMPT_MAX_LENGTH = max_length
    vllm_processors.InternVLProcessor = D2EInternVLProcessor


def timestamp_tokens(timestamp_ns: int) -> str:
    """Encode a timestamp exactly like this checkpoint's factorized encoder"""
    value = timestamp_ns // TIMESTAMP_UNIT_NS
    digits: list[int] = []
    for base in reversed(TIMESTAMP_BASES):
        digits.insert(0, value % base)
        value //= base
    return "".join(f"<{digit}>" for digit in digits)


def screen_event_text(timestamp_ns: int) -> str:
    return f"<EVENT_START><SCREEN>{timestamp_tokens(timestamp_ns)}{IMAGE_BLOCK}<EVENT_END>"


def compact_multimodal_prompt(expanded_text: str) -> str:
    """Replace each expanded image block with one vLLM image placeholder"""
    return re.sub(r"<img>.*?</img>", "<IMG_CONTEXT>", expanded_text)


def vk_name(vk: int) -> str:
    special = {
        9: "TAB",
        13: "ENTER",
        16: "SHIFT",
        17: "CTRL",
        18: "ALT",
        27: "ESC",
        32: "SPACE",
        37: "LEFT",
        38: "UP",
        39: "RIGHT",
        40: "DOWN",
    }
    if 48 <= vk <= 57 or 65 <= vk <= 90:
        return chr(vk)
    return special.get(vk, f"VK_{vk}")


def summarize_event(text: str) -> str:
    """Turn hierarchical D2E keyboard tokens into a readable label"""
    tokens = re.findall(r"<[^>]+>", text)
    if len(tokens) >= 7 and tokens[:2] == ["<EVENT_START>", "<KEYBOARD>"]:
        try:
            vk_token = tokens[5][1:-1]
            if vk_token.startswith("VK_"):
                vk_token = vk_token[3:]
            key = vk_name(int(vk_token))
            if len(key) == 1:
                key = key.lower()
            action = tokens[6][1:-1]
            return f"{key} {action}"
        except (ValueError, IndexError):
            pass
    if len(tokens) >= 2 and tokens[:2] == ["<EVENT_START>", "<MOUSE>"]:
        return text
    if tokens[:2] == ["<EVENT_START>", "<SCREEN>"]:
        return "SCREEN"
    return text[:100] if text else "no output"


def decode_event_timestamp_ns(text: str) -> int | None:
    """Decode the three hierarchical timestamp tokens from a generated event"""
    tokens = re.findall(r"<[^>]+>", text)
    if len(tokens) < 5 or tokens[0] != "<EVENT_START>":
        return None
    try:
        digits = [int(token[1:-1]) for token in tokens[2:5]]
    except ValueError:
        return None
    if not all(0 <= digit < 10 for digit in digits):
        return None
    units = digits[0] * 100 + digits[1] * 10 + digits[2]
    return units * TIMESTAMP_UNIT_NS


def is_valid_action_event(text: str) -> bool:
    """Reject incomplete generated actions before they reach the MCAP writer."""
    tokens = re.findall(r"<[^>]+>", text)
    if not tokens or tokens[0] != "<EVENT_START>" or tokens[-1] != "<EVENT_END>":
        return False
    if len(tokens) == 8 and tokens[1] == "<KEYBOARD>":
        return True
    return len(tokens) == 19 and tokens[1] == "<MOUSE>"


@dataclass
class ContextEntry:
    text: str
    image: Image.Image | None = None


class RollingContext:
    def __init__(self, tokenizer, max_tokens: int, max_images: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_images = max_images
        self.entries: deque[ContextEntry] = deque()

    def append(self, text: str, image: Image.Image | None = None) -> None:
        self.entries.append(ContextEntry(text=text, image=image))
        while len(self) > self.max_tokens or self.image_count > self.max_images:
            if len(self.entries) <= 1:
                break
            self.entries.popleft()

    def request(self) -> dict:
        expanded = "".join(entry.text for entry in self.entries)
        images = [entry.image for entry in self.entries if entry.image is not None]
        request: dict = {"prompt": compact_multimodal_prompt(expanded)}
        if images:
            request["multi_modal_data"] = {"image": images}
        return request

    @property
    def image_count(self) -> int:
        return sum(entry.image is not None for entry in self.entries)

    def __len__(self) -> int:
        text = "".join(entry.text for entry in self.entries)
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def draw_overlay(frame, lines: list[str]) -> None:
    y = 32
    for line in lines:
        cv2.putText(frame, line[:115], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        y += 28


def frame_to_image(frame) -> Image.Image:
    """Convert an OpenCV frame to the 448x448 RGB image"""
    resized = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def unwrap_timestamp(raw_ns: int, previous_ns: int | None, wrap_bias_ns: int) -> tuple[int, int]:
    """Map the model's repeating 0-10 second timestamp onto the video timeline."""
    if previous_ns is not None and previous_ns % TIMESTAMP_RANGE_NS > raw_ns:
        wrap_bias_ns += TIMESTAMP_RANGE_NS
    return raw_ns + wrap_bias_ns, wrap_bias_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--kv-cache-memory-mib", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=0, help="Run warmup inference requests before video playback")
    parser.add_argument("--model-impl", choices=("vllm", "transformers"), default="transformers",
                        help="Model execution backend: native vLLM or Hugging Face Transformers")
    parser.add_argument("--quantization", default=None,
                        help=(
                            "Optional vLLM load-time quantization scheme, for example "
                            "'fp8_per_tensor', 'fp8_per_block', or "
                            "'int8_per_channel_weight_only'"))
    parser.add_argument("--action-display-seconds", type=float, default=1.0, help="How long an action remains visible")
    parser.add_argument("--max-events-per-frame", type=int, default=8, help="Maximum actions generated for one frame")
    parser.add_argument("--enable-cudagraph", action="store_true",
                        help=(
                            "Enable vLLM PIECEWISE CUDA graphs; startup is slower but warm "
                            "inference may improve. FULL capture is incompatible with this "
                            "Transformers InternVL path."))
    parser.add_argument("--infer-every-n", type=int, default=1, help="Run inference every Nth video frame")
    args = parser.parse_args()

    if args.infer_every_n < 1:
        parser.error("--infer-every-n must be at least 1")
    if args.max_events_per_frame < 1:
        parser.error("--max-events-per-frame must be at least 1")
    if args.action_display_seconds < 0:
        parser.error("--action-display-seconds cannot be negative")
    if args.max_new_tokens < 1 or args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be between 1 and --max-model-len - 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if not args.input_video.exists():
        parser.error(f"Video does not exist: {args.input_video}")
    return args


def load_inference_engine(args: argparse.Namespace):
    """Load the tokenizer, vLLM engine, and deterministic generation settings."""

    patch_vllm_transformers_internvl_processor(args.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    event_end_id = tokenizer.convert_tokens_to_ids("<EVENT_END>")
    screen_id = tokenizer.convert_tokens_to_ids("<SCREEN>")
    precision = args.quantization or "none (BF16)"
    print(
        f"Loading persistent vLLM engine; backend: {args.model_impl}; "
        f"quantization: {precision} ...")
    if args.enable_cudagraph:
        print("CUDA graph mode: PIECEWISE (FULL capture disabled)")
    llm = LLM(model=str(args.model),
              tokenizer=str(args.model),
              trust_remote_code=True,
              dtype="bfloat16",
              quantization=args.quantization,
              model_impl=args.model_impl,
              hf_overrides={"tie_word_embeddings": False},
              mm_processor_kwargs={"image_seq_length": 256, "crop_to_patches": False},
              max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": args.max_images},
              kv_cache_memory_bytes=args.kv_cache_memory_mib * 1024**2,
              # Offline LLM.generate disables request metrics by default.
              # Enable them so RequestOutput contains first-token latency.
              disable_log_stats=False,
              enforce_eager=not args.enable_cudagraph,
              compilation_config={"cudagraph_mode": "PIECEWISE"} if args.enable_cudagraph else None,)
    
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=[event_end_id, screen_id], skip_special_tokens=False)
    return tokenizer, llm, sampling, event_end_id, screen_id


def main() -> int:
    args = parse_args()
    tokenizer, llm, sampling, event_end_id, screen_id = load_inference_engine(args)

    if args.warmup_runs:
        warmup_context = RollingContext(
            tokenizer=tokenizer,
            max_tokens=args.max_model_len - args.max_new_tokens,
            max_images=args.max_images)
        warmup_context.append(screen_event_text(0), image=Image.new("RGB", (448, 448)))
        warmup_started = time.perf_counter()
        print(f"Running {args.warmup_runs} inference warm-up request(s) ...", flush=True)
        for _ in range(args.warmup_runs):
            llm.generate(warmup_context.request(), sampling_params=sampling, use_tqdm=False)
        print(f"Warm-up complete in {time.perf_counter() - warmup_started:.2f}s", flush=True)

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")

    # Reserve space for SamplingParams.max_tokens; vLLM validates the combined
    # prompt and requested output against max_model_len before generation.
    context = RollingContext(
        tokenizer=tokenizer,
        max_tokens=args.max_model_len - args.max_new_tokens,
        max_images=args.max_images)
    frame_number = 0
    inference_count = 0
    display_output = ""
    display_output_until = 0.0
    last_latency_ms = 0.0
    last_ttft_ms = 0.0
    total_inference_seconds = 0.0
    total_ttft_seconds = 0.0
    ttft_count = 0
    loop_started = time.perf_counter()
    last_context_timestamp_ns: int | None = None
    timestamp_bias_ns = 0
    first_w_press: tuple[int, int, int] | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1

            should_infer = frame_number % args.infer_every_n == 0
            if should_infer:
                frame_action_output = ""
                position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_ns = int(position_ms * 1_000_000)
                image = frame_to_image(frame)

                if context.entries:
                    for event_index in range(args.max_events_per_frame):
                        start = time.perf_counter()
                        request_output = llm.generate(
                            context.request(), sampling_params=sampling, use_tqdm=False
                        )[0]
                        last_latency_ms = (time.perf_counter() - start) * 1000
                        total_inference_seconds += last_latency_ms / 1000
                        metrics = request_output.metrics
                        if metrics is not None and metrics.first_token_latency > 0:
                            last_ttft_ms = metrics.first_token_latency * 1000
                            total_ttft_seconds += metrics.first_token_latency
                            ttft_count += 1
                            ttft_text = f"{last_ttft_ms:.1f} ms"
                        else:
                            last_ttft_ms = 0.0
                            ttft_text = "n/a"

                        result = request_output.outputs[0]
                        output_ids = list(result.token_ids)
                        last_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                        last_summary = summarize_event(last_output)
                        inference_count += 1
                        print(
                            f"frame {frame_number} | event {event_index + 1} | "
                            f"TTFT {ttft_text} | complete {last_latency_ms:.1f} ms | "
                            f"{last_summary}",
                            flush=True)

                        if screen_id in output_ids:
                            break
                        if not output_ids or output_ids[-1] != event_end_id:
                            break

                        raw_timestamp_ns = decode_event_timestamp_ns(last_output)
                        if raw_timestamp_ns is None:
                            break
                        adjusted_timestamp_ns, candidate_bias_ns = unwrap_timestamp(
                            raw_timestamp_ns, last_context_timestamp_ns, timestamp_bias_ns)
                        if adjusted_timestamp_ns > timestamp_ns:
                            break

                        context.append(last_output)
                        last_context_timestamp_ns = adjusted_timestamp_ns
                        timestamp_bias_ns = candidate_bias_ns
                        if first_w_press is None and last_summary == "w press":
                            first_w_press = (adjusted_timestamp_ns, timestamp_ns, frame_number)
                        if last_output.startswith(("<EVENT_START><KEYBOARD>", "<EVENT_START><MOUSE>")):
                            frame_action_output = last_summary

                context.append(screen_event_text(timestamp_ns), image=image)
                last_context_timestamp_ns = timestamp_ns
                timestamp_bias_ns = timestamp_ns - timestamp_ns % TIMESTAMP_RANGE_NS
                if frame_action_output:
                    display_output = frame_action_output
                    display_output_until = time.perf_counter() + args.action_display_seconds

            preview = cv2.resize(frame, (960, 540))
            if time.perf_counter() >= display_output_until:
                display_output = ""
            elapsed = time.perf_counter() - loop_started
            loop_fps = frame_number / elapsed if elapsed > 0 else 0.0
            inference_fps = inference_count / total_inference_seconds if total_inference_seconds > 0 else 0.0
            overlay_lines = [
                f"FPS {loop_fps:.2f} | inference FPS {inference_fps:.2f} | "
                f"latency {last_latency_ms:.1f} ms"]
            if display_output:
                overlay_lines.append(f"output: {display_output}")
            draw_overlay(preview, overlay_lines)
            cv2.imshow("D2E vLLM realtime prototype", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    loop_seconds = time.perf_counter() - loop_started
    if inference_count:
        avg_ttft = (
            f"{total_ttft_seconds * 1000 / ttft_count:.1f} ms"
            if ttft_count
            else "n/a")
        print(
            f"Inference summary: requests={inference_count} | "
            f"avg TTFT={avg_ttft} | "
            f"avg complete={total_inference_seconds * 1000 / inference_count:.1f} ms | "
            f"throughput={inference_count / total_inference_seconds:.2f} requests/s | "
            f"playback={frame_number / loop_seconds:.2f} FPS",
            flush=True)

    if first_w_press is None:
        print("First w press: not generated", flush=True)
    else:
        action_ns, detected_ns, detected_frame = first_w_press
        print(f"First w press: action_time={action_ns / 1e9:.2f}s "
              f"detected_at_video={detected_ns / 1e9:.2f}s "
              f"frame={detected_frame}",flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
