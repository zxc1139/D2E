"""Load the D2E checkpoint with vLLM without running generation."""

import argparse
from pathlib import Path

from vllm import LLM


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "open-world-agents/checkpoints/cyberpunk_trial_best"
DEFAULT_TOKENIZER = DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    print(f"Loading D2E checkpoint: {args.model}")
    LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 8},
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Start with the simplest execution path. Enable compilation only after
        # the BF16 output matches the Transformers reference implementation.
        enforce_eager=True,
    )
    print("SUCCESS: D2E checkpoint loaded through vLLM in BF16")


if __name__ == "__main__":
    main()
