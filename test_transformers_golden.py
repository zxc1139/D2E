"""Replay the captured D2E sample using pure Hugging Face Transformers.

This intentionally does not import or use vLLM. It verifies that the current
checkpoint reproduces the token IDs saved in golden_sample.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "open-world-agents/checkpoints/cyberpunk_trial_best"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sample", type=Path, default=ROOT / "golden_sample.pt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this BF16 checkpoint test")

    print("Backend: pure Hugging Face Transformers (vLLM is not imported)")
    print(f"Model:   {args.model.resolve()}")
    print(f"Sample:  {args.sample.resolve()}")

    sample = torch.load(args.sample, map_location="cpu", weights_only=True)
    expected_ids = sample["outputs"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().to("cuda")

    input_ids = sample["sequences"].to("cuda")
    pixel_values = sample["pixel_values"].to("cuda", dtype=torch.bfloat16)
    attention_mask = sample["attention_mask"].to("cuda")
    event_end_id = tokenizer.convert_tokens_to_ids("<EVENT_END>")
    screen_id = tokenizer.convert_tokens_to_ids("<SCREEN>")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pad_token_id=event_end_id,
            eos_token_id=[event_end_id, screen_id],
            max_new_tokens=max(2, len(expected_ids)),
            do_sample=False,
            use_cache=True,
        )

    actual_ids = output_ids[0, input_ids.shape[1] :].tolist()
    print(f"Actual token IDs:   {actual_ids}")
    print(f"Expected token IDs: {expected_ids}")
    print(f"Actual text:        {tokenizer.decode(actual_ids, skip_special_tokens=False)!r}")
    print(f"Expected text:      {tokenizer.decode(expected_ids, skip_special_tokens=False)!r}")

    if actual_ids == expected_ids:
        print("PASS: the checkpoint exactly reproduces the golden output.")
        return 0

    print("FAIL: the current checkpoint does not reproduce the golden output.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
