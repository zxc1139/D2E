"""Run D2E vLLM inference offline and save an OWA-compatible MCAP.

The generation schedule mirrors ``inference_capture_opencv.py``: sample screen
events at 20 Hz, then repeatedly generate actions in each gap until the model
requests the next screen or predicts an action after that screen's timestamp.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import cv2
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from inference_vllm_opencv import (
    DEFAULT_MODEL,
    DEFAULT_VIDEO,
    TIMESTAMP_RANGE_NS,
    RollingContext,
    decode_event_timestamp_ns,
    is_valid_action_event,
    patch_vllm_transformers_internvl_processor,
    screen_event_text,
    summarize_event,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OWA_PYTHON = Path("/home/labs/miniconda3/envs/owa-env-local/bin/python")


def video_duration_seconds(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError("Video does not report a valid FPS/frame count")
    return frame_count / fps


def video_start_time_ns(video_path: Path) -> int:
    """Return the video stream's absolute starting PTS using ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=start_time",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    value = result.stdout.strip()
    if not value or value == "N/A":
        return 0
    return int(round(float(value) * 1_000_000_000))


def read_frame_at(
    cap: cv2.VideoCapture, timestamp_ns: int, stream_start_ns: int
) -> Image.Image:
    # MCAP/MediaRef timestamps are absolute stream PTS. OpenCV's POS_MSEC is
    # relative to the first frame, so subtract a non-zero stream start time.
    relative_timestamp_ns = max(0, timestamp_ns - stream_start_ns)
    cap.set(cv2.CAP_PROP_POS_MSEC, relative_timestamp_ns / 1_000_000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not decode video at {timestamp_ns / 1e9:.3f}s")
    # Keep the source resolution. The InternVL image processor performs the
    # same bicubic 448x448 resize used by the original offline OWA pipeline.
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def write_record(stream, record: dict) -> None:
    stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-mcap", type=Path, default=Path("vllm_output.mcap"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--screen-rate-hz", type=float, default=20.0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--kv-cache-memory-mib", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--model-impl",
        choices=("vllm", "transformers"),
        default="transformers",
        help="Model execution backend: native vLLM or Hugging Face Transformers",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help=(
            "Optional vLLM load-time quantization scheme, for example "
            "'fp8_per_tensor', 'fp8_per_block', or "
            "'int8_per_channel_weight_only'"
        ),
    )
    parser.add_argument(
        "--enable-cudagraph",
        action="store_true",
        help=(
            "Enable vLLM PIECEWISE CUDA graphs; startup is slower but warm "
            "inference may improve"
        ),
    )
    parser.add_argument(
        "--max-events-per-gap",
        type=int,
        default=256,
        help="Safety limit only; the old loop was otherwise unbounded",
    )
    parser.add_argument("--time-shift", type=float, default=0.1)
    parser.add_argument("--owa-python", type=Path, default=DEFAULT_OWA_PYTHON)
    args = parser.parse_args()

    if not args.input_video.exists():
        parser.error(f"Video does not exist: {args.input_video}")
    if not args.model.exists():
        parser.error(f"Model does not exist: {args.model}")
    if not args.owa_python.exists():
        parser.error(f"OWA Python does not exist: {args.owa_python}")
    if args.screen_rate_hz <= 0:
        parser.error("--screen-rate-hz must be positive")
    if args.max_duration is not None and args.max_duration <= 0:
        parser.error("--max-duration must be positive")
    if args.max_events_per_gap < 1:
        parser.error("--max-events-per-gap must be at least 1")
    if args.max_new_tokens < 1 or args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be between 1 and --max-model-len - 1")

    patch_vllm_transformers_internvl_processor(args.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    event_end_id = tokenizer.convert_tokens_to_ids("<EVENT_END>")
    screen_id = tokenizer.convert_tokens_to_ids("<SCREEN>")

    precision = args.quantization or "none (BF16)"
    print(
        f"Loading persistent vLLM engine; backend: {args.model_impl}; "
        f"quantization: {precision} ...",
        flush=True,
    )
    if args.enable_cudagraph:
        print("CUDA graph mode: PIECEWISE (FULL capture disabled)", flush=True)
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        trust_remote_code=True,
        dtype="bfloat16",
        quantization=args.quantization,
        model_impl=args.model_impl,
        hf_overrides={"tie_word_embeddings": False},
        mm_processor_kwargs={
            "image_seq_length": 256,
            "crop_to_patches": False,
        },
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_images},
        kv_cache_memory_bytes=args.kv_cache_memory_mib * 1024**2,
        enforce_eager=not args.enable_cudagraph,
        compilation_config=(
            {"cudagraph_mode": "PIECEWISE"}
            if args.enable_cudagraph
            else None))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[event_end_id, screen_id],
        skip_special_tokens=False)

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")
    duration = video_duration_seconds(cap)
    stream_start_ns = video_start_time_ns(args.input_video)
    print(
        f"Video stream start PTS: {stream_start_ns / 1e9:.3f}s; "
        "aligning OpenCV frames with OWA MediaRef",
        flush=True)
    if args.max_duration is not None:
        duration = min(duration, args.max_duration)
    screen_count = int(duration * args.screen_rate_hz)
    interval_ns = int(1_000_000_000 / args.screen_rate_hz)
    if screen_count < 1:
        raise RuntimeError("Selected duration contains no screen samples")

    context = RollingContext(
        tokenizer=tokenizer,
        max_tokens=args.max_model_len - args.max_new_tokens,
        max_images=args.max_images)
    last_context_timestamp_ns: int | None = None
    timestamp_bias_ns = 0
    keyboard_count = 0
    mouse_count = 0
    invalid_count = 0
    inference_count = 0
    inference_seconds = 0.0

    try:
        with tempfile.TemporaryDirectory(prefix="d2e-vllm-") as tmpdir:
            records_path = Path(tmpdir) / "records.jsonl"
            with records_path.open("w", encoding="utf-8") as records:
                for index in range(screen_count):
                    timestamp_ns = index * interval_ns
                    image = read_frame_at(cap, timestamp_ns, stream_start_ns)

                    if context.entries:
                        for event_index in range(args.max_events_per_gap):
                            started = time.perf_counter()
                            result = llm.generate(
                                context.request(),
                                sampling_params=sampling,
                                use_tqdm=False,
                            )[0].outputs[0]
                            inference_seconds += time.perf_counter() - started
                            inference_count += 1
                            output_ids = list(result.token_ids)
                            output_text = tokenizer.decode(
                                output_ids, skip_special_tokens=False)

                            if screen_id in output_ids:
                                break
                            if not output_ids or output_ids[-1] != event_end_id:
                                invalid_count += 1
                                break

                            # Ending at EVENT_END is not sufficient: the model
                            # can still emit an incomplete mouse payload. The
                            # original OWA loop rejects that during decode.
                            if not is_valid_action_event(output_text):
                                invalid_count += 1
                                print(
                                    f"screen={index + 1}/{screen_count} "
                                    f"rejected malformed output={output_text}",
                                    flush=True)
                                break

                            raw_timestamp_ns = decode_event_timestamp_ns(output_text)
                            if raw_timestamp_ns is None:
                                invalid_count += 1
                                break
                            candidate_bias_ns = timestamp_bias_ns
                            if (
                                last_context_timestamp_ns is not None
                                and last_context_timestamp_ns % TIMESTAMP_RANGE_NS
                                > raw_timestamp_ns
                            ):
                                candidate_bias_ns += TIMESTAMP_RANGE_NS
                            adjusted_timestamp_ns = (raw_timestamp_ns + candidate_bias_ns)
                            if adjusted_timestamp_ns > timestamp_ns:
                                break

                            summary = summarize_event(output_text)
                            if output_text.startswith("<EVENT_START><KEYBOARD>"):
                                keyboard_count += 1
                            elif output_text.startswith("<EVENT_START><MOUSE>"):
                                mouse_count += 1
                            else:
                                invalid_count += 1
                                break

                            context.append(output_text)
                            last_context_timestamp_ns = adjusted_timestamp_ns
                            timestamp_bias_ns = candidate_bias_ns
                            write_record(
                                records,
                                {
                                    "kind": "action",
                                    "timestamp_ns": adjusted_timestamp_ns,
                                    "token_ids": output_ids,
                                },)
                            print(
                                f"screen={index + 1}/{screen_count} "
                                f"event={event_index + 1} "
                                f"time={adjusted_timestamp_ns / 1e9:.2f}s "
                                f"output={summary}",
                                flush=True)
                        else:
                            raise RuntimeError("Reached --max-events-per-gap; generation may be stuck")

                    context.append(screen_event_text(timestamp_ns), image=image)
                    last_context_timestamp_ns = timestamp_ns
                    timestamp_bias_ns = timestamp_ns - timestamp_ns % TIMESTAMP_RANGE_NS
                    write_record(
                        records,
                        {"kind": "screen", "timestamp_ns": timestamp_ns})
                    if (index + 1) % 20 == 0 or index + 1 == screen_count:
                        rate = (
                            inference_count / inference_seconds
                            if inference_seconds > 0
                            else 0.0)
                        print(
                            f"progress={index + 1}/{screen_count} "
                            f"actions={keyboard_count + mouse_count} "
                            f"model_calls/s={rate:.2f}",
                            flush=True)

            command = [
                str(args.owa_python),
                str(ROOT / "write_vllm_mcap.py"),
                "--records",
                str(records_path),
                "--output-mcap",
                str(args.output_mcap.resolve()),
                "--model",
                str(args.model.resolve()),
                "--video",
                str(args.input_video.resolve()),
                "--time-shift",
                str(args.time_shift)]
            subprocess.run(command, check=True)
    finally:
        cap.release()

    print(
        f"Done: {args.output_mcap} | keyboard={keyboard_count} "
        f"mouse={mouse_count} invalid={invalid_count} "
        f"model_calls={inference_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
