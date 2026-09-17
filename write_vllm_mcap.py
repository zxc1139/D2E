"""Serialize vLLM offline predictions to an OWA-compatible MCAP.

This helper is launched with the OWA conda environment.  Keeping it separate
lets the inference process use the dedicated vLLM environment without mixing
their Torch dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from mcap_owa.highlevel import OWAMcapWriter
from owa.core import MESSAGES
from owa.data.encoders import EventEncoderError
from owa.data.episode_tokenizer import EpisodeTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-mcap", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--time-shift", type=float, default=0.1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    episode_tokenizer = EpisodeTokenizer.from_transformers(str(args.model))
    episode_tokenizer.prepare_model(tokenizer=tokenizer)
    ScreenCaptured = MESSAGES["desktop/ScreenCaptured"]
    video_path = str(args.video.resolve())
    shift_ns = int(args.time_shift * 1_000_000_000)

    screen_count = 0
    action_count = 0
    skipped_count = 0
    with OWAMcapWriter(args.output_mcap) as writer, args.records.open(
        encoding="utf-8"
    ) as records:
        for line in records:
            record = json.loads(line)
            timestamp_ns = int(record["timestamp_ns"])
            if record["kind"] == "screen":
                message = ScreenCaptured(
                    utc_ns=timestamp_ns,
                    media_ref={"uri": video_path, "pts_ns": timestamp_ns},
                )
                writer.write_message(message, topic="screen", timestamp=timestamp_ns)
                screen_count += 1
                continue

            try:
                event = episode_tokenizer.decode_event(
                    np.asarray(record["token_ids"], dtype=np.int64)
                )
            except EventEncoderError as error:
                skipped_count += 1
                print(f"Skipping malformed action record: {error}", flush=True)
                continue
            event.timestamp = max(0, timestamp_ns - shift_ns)
            writer.write_message(event)
            action_count += 1

    print(
        f"MCAP written: {args.output_mcap} "
        f"({screen_count} screens, {action_count} actions, "
        f"{skipped_count} malformed actions skipped)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
