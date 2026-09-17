"""Run one deterministic D2E multimodal inference with vLLM.

The input and expected output come from ``golden_sample.pt``, captured from
the Transformers implementation.  This keeps the smoke test independent of
the OWA runtime and video pipeline.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# This is a deterministic greedy smoke test, so FlashInfer's top-k/top-p
# sampler and its JIT/autotuning provide no benefit and can delay startup.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "open-world-agents/checkpoints/cyberpunk_trial_best"
DEFAULT_SAMPLE = ROOT / "golden_sample.pt"

# Normalization used by the checkpoint's processor_config.json.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def recover_image(pixel_values: torch.Tensor) -> Image.Image:
    """Invert the checkpoint's image normalization back to an RGB image."""
    if pixel_values.shape != (1, 3, 448, 448):
        raise ValueError(f"Expected pixel_values shape (1, 3, 448, 448), got {tuple(pixel_values.shape)}")

    pixels = pixel_values[0].float()
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    pixels = ((pixels * std + mean) * 255).round().clamp(0, 255).byte()
    return Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode="RGB")


def recover_prompt(tokenizer, input_ids: torch.Tensor) -> str:
    """Collapse the expanded InternVL image tokens to vLLM's placeholder."""
    prompt = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    # The HF-format InternVL processor uses one <IMG_CONTEXT> as the raw-image
    # placeholder, then expands it to <img> + 256 context tokens + </img>.
    prompt, replacements = re.subn(
        r"<img>.*?</img>", "<IMG_CONTEXT>", prompt, count=1
    )
    if replacements != 1:
        raise ValueError("The golden prompt does not contain one <img>...</img> block")
    return prompt


def patch_vllm_transformers_internvl_processor() -> None:
    """Use the HF-format processor expected by this converted checkpoint."""
    from transformers.models.internvl.processing_internvl import (
        InternVLProcessor as HFInternVLProcessor,
    )
    from vllm.transformers_utils import processors as vllm_processors

    # vLLM maps the architecture to its legacy InternVL/InternS1 processor,
    # whose required constructor fields and image processor are incompatible
    # with this checkpoint's native Transformers processor_config.json.
    vllm_processors.InternVLProcessor = HFInternVLProcessor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--kv-cache-memory-mib",
        type=int,
        default=None,
        help=(
            "Explicit decoder KV-cache allocation in MiB. When set, vLLM "
            "ignores --gpu-memory-utilization for cache sizing."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional vLLM quantization method, for example 'fp8'",
    )
    parser.add_argument(
        "--model-impl",
        choices=("auto", "vllm", "transformers"),
        default="auto",
        help="Model implementation used inside vLLM",
    )
    parser.add_argument("--strict", action="store_true", help="Return a failure code if vLLM and Transformers tokens differ")
    args = parser.parse_args()

    processor_overrides = None
    if args.model_impl == "transformers":
        # vLLM 0.27.1's generic Transformers multimodal backend does not
        # recover this required InternVLProcessor field from processor_config.
        # Keep vLLM's token-budget calculation consistent with this checkpoint.
        # HF InternVL's call-time default is crop_to_patches=True even though
        # processor_config.json records false; without this explicit override,
        # vLLM budgets 10 patches (2562 tokens) but preprocessing emits one
        # patch (258 tokens), and startup profiling fails in torch.split().
        processor_overrides = {
            "image_seq_length": 256,
            "crop_to_patches": False,
        }
        patch_vllm_transformers_internvl_processor()

    print(f"Running script: {Path(__file__).resolve()}")
    print(f"Model implementation: {args.model_impl}")
    print(f"Quantization: {args.quantization or 'none (BF16)'}")
    print(f"KV-cache memory: {args.kv_cache_memory_mib or 'automatic'} MiB")
    print(f"Processor overrides: {processor_overrides}")
    print("HF config overrides: {'tie_word_embeddings': False}")
    print(f"Loading {args.model} with vLLM ...")
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        trust_remote_code=True,
        # The outer InternVL config incorrectly says true while text_config
        # says false and the checkpoint contains distinct trained
        # embed_tokens/lm_head tensors. vLLM otherwise ties them and skips the
        # checkpoint's lm_head, changing every generated token.
        hf_overrides={"tie_word_embeddings": False},
        dtype="bfloat16",
        quantization=args.quantization,
        model_impl=args.model_impl,
        mm_processor_kwargs=processor_overrides,
        max_model_len=args.max_model_len,
        # Keep engine initialization identical to test_vllm_load.py, which is
        # our known-good baseline. The request below still contains one image.
        limit_mm_per_prompt={"image": 8},
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=(
            args.kv_cache_memory_mib * 1024**2
            if args.kv_cache_memory_mib is not None
            else None
        ),
        enforce_eager=True,
    )
    print("vLLM engine ready; preparing the golden request ...")

    # Do this only after vLLM has spawned and initialized EngineCore. Starting
    # Hugging Face tokenizer worker threads before that process is created can
    # deadlock a forked child.
    sample = torch.load(args.sample, map_location="cpu", weights_only=True)
    input_ids = sample["sequences"][0]
    expected_ids = sample["outputs"].tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = recover_prompt(tokenizer, input_ids)
    image = recover_image(sample["pixel_values"])

    screen_token_id = tokenizer.convert_tokens_to_ids("<SCREEN>")
    event_end_token_id = tokenizer.convert_tokens_to_ids("<EVENT_END>")
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max(2, len(expected_ids)),
        stop_token_ids=[screen_token_id, event_end_token_id],
        skip_special_tokens=False,
    )
    request_output = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        sampling_params=sampling,
    )[0]
    result = request_output.outputs[0]

    actual_ids = list(result.token_ids)
    processed_prompt_ids = list(request_output.prompt_token_ids)
    expected_prompt_ids = input_ids.tolist()
    prompt_matches = processed_prompt_ids == expected_prompt_ids

    print(f"Prompt IDs match:      {prompt_matches}")
    print(f"vLLM prompt length:    {len(processed_prompt_ids)}")
    print(f"Expected prompt length: {len(expected_prompt_ids)}")
    if not prompt_matches:
        mismatch = next(
            (
                i
                for i, (actual, expected) in enumerate(
                    zip(processed_prompt_ids, expected_prompt_ids)
                )
                if actual != expected
            ),
            min(len(processed_prompt_ids), len(expected_prompt_ids)),
        )
        print(f"First prompt mismatch: index {mismatch}")
        print(f"vLLM prompt IDs near mismatch:    {processed_prompt_ids[max(0, mismatch - 4):mismatch + 5]}")
        print(f"Expected prompt IDs near mismatch: {expected_prompt_ids[max(0, mismatch - 4):mismatch + 5]}")
    print(f"vLLM token IDs:        {actual_ids}")
    print(f"Transformers token IDs: {expected_ids}")
    print(f"vLLM text:             {tokenizer.decode(actual_ids, skip_special_tokens=False)!r}")
    print(f"Transformers text:      {tokenizer.decode(expected_ids, skip_special_tokens=False)!r}")

    if prompt_matches and actual_ids == expected_ids:
        print("PASS: vLLM output exactly matches the Transformers golden sample.")
        return 0

    print("ENGINE TEST PASSED, CORRECTNESS FAILED: vLLM generated, but its tokens differ from Transformers.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
