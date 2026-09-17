#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru==0.7.2",
#     "numpy>=1.24.0",
#     "opencv-python>=4.8.0",
#     "mcap-owa-support @ git+https://github.com/lastdefiance20/open-world-agents.git#subdirectory=projects/mcap-owa-support",
#     "owa-core @ git+https://github.com/lastdefiance20/open-world-agents.git#subdirectory=projects/owa-core",
#     "owa-msgs @ git+https://github.com/lastdefiance20/open-world-agents.git#subdirectory=projects/owa-msgs",
# ]
# ///
"""
G-IDM evaluation script: compare ground truth MCAP with predicted MCAP and
optionally play a synchronized video/action viewer.

Usage:
    uv run evaluate_video.py video.mkv ground_truth.mcap predicted.mcap
    uv run evaluate_video.py video.mkv ground_truth.mcap predicted.mcap --no-viewer
    uv run evaluate_video.py video.mkv ground_truth.mcap predicted.mcap --output results.json

Metrics (from D2E paper Section F.2):
    - Mouse: Pearson correlation (X/Y), Scale ratio (X/Y)
    - Keyboard: Per-key accuracy
    - Mouse buttons: Per-button accuracy
    - All metrics computed over non-overlapping 50ms temporal bins
"""

import argparse
import copy
import json
import sys
import time
import cv2
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, TypedDict

import numpy as np
from loguru import logger

from mcap_owa.highlevel import OWAMcapReader
from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.msgs.desktop.mouse import RawMouseEvent


class EventType(StrEnum):
    """Type of event."""

    SCREEN = "screen"
    KEYBOARD = "keyboard"
    MOUSE_OP = "mouse_op"
    MOUSE_NOP = "mouse_nop"


def _determine_event_type(event: McapMessage) -> EventType:
    """Determine the event type from a decoded event object."""
    event_type = event.topic
    if event_type == "mouse/raw":
        if event.decoded.button_flags != 0:
            return EventType.MOUSE_OP
        else:
            return EventType.MOUSE_NOP
    if event_type in ["keyboard", "screen"]:
        return EventType(event_type)
    raise ValueError(f"Unknown event type: {event_type}")


@dataclass
class BinningConfig:
    bin_size_ns: int = 50_000_000  # 50ms bins (20Hz)
    empty_bins_as_correct: bool = False


class Bin(TypedDict):
    start_ts: int
    end_ts: int
    events: List[McapMessage]


@dataclass
class AlignmentInfo:
    src_offset: int
    dst_offset: int
    start_pts: int
    end_pts: int
    duration_sec: float

def create_bins(start_time: int, end_time: int, bin_size_ns: int) -> List[Bin]:
    """Create time bins for the given time range."""
    bins = []
    current_time = start_time
    while current_time < end_time:
        bin_end = min(current_time + bin_size_ns, end_time)
        bins.append({"start_ts": current_time, "end_ts": bin_end, "events": []})
        current_time = bin_end
    return bins

def bin_events(events: Iterator[McapMessage], bins: List[Bin]) -> List[Bin]:
    """Assign events to appropriate time bins."""
    last_bin_idx = 0
    for event in events:
        while last_bin_idx < len(bins) and bins[last_bin_idx]["end_ts"] <= event.timestamp:
            last_bin_idx += 1
        if last_bin_idx >= len(bins):
            break
        bins[last_bin_idx]["events"].append(event)
    return bins



class MouseMoveMetric:
    """Mouse movement metrics: Pearson correlation and Scale ratio (X/Y)."""

    name = "mouse_move"

    def __init__(self, cfg: BinningConfig):
        self.empty_bins_as_correct = cfg.empty_bins_as_correct
        self.src_x_values: List[float] = []
        self.dst_x_values: List[float] = []
        self.src_y_values: List[float] = []
        self.dst_y_values: List[float] = []
        self.src_sequences: List[np.ndarray] = []
        self.dst_sequences: List[np.ndarray] = []

    def update(self, src_bin: Bin, dst_bin: Bin):
        """Update the metric with a new bin."""
        src_moves = self._extract_mouse_moves(src_bin)
        dst_moves = self._extract_mouse_moves(dst_bin)

        src_total = np.array(src_moves).sum(axis=0) if src_moves else np.array([0, 0])
        dst_total = np.array(dst_moves).sum(axis=0) if dst_moves else np.array([0, 0])

        # Only add to Pearson calculation if at least one has movement
        # (matches original metrics.py behavior)
        if src_moves or dst_moves:
            self.src_x_values.append(float(src_total[0]))
            self.dst_x_values.append(float(dst_total[0]))
            self.src_y_values.append(float(src_total[1]))
            self.dst_y_values.append(float(dst_total[1]))

        # Scale ratio always includes all bins
        self.src_sequences.append(src_total)
        self.dst_sequences.append(dst_total)

    def _extract_mouse_moves(self, bin_data: Bin) -> List[Tuple[int, int]]:
        """Extract mouse movement deltas from a bin."""
        moves = []
        for event in bin_data["events"]:
            if _determine_event_type(event) == EventType.MOUSE_NOP:
                decoded = event.decoded
                moves.append((decoded.dx, decoded.dy))
        return moves

    @property
    def metric(self) -> Dict[str, Any]:
        """Return the current metric value."""
        result = {}

        # Pearson correlation for X
        if len(self.src_x_values) > 1:
            try:
                corr_x = np.corrcoef(self.src_x_values, self.dst_x_values)[0, 1]
                result["pearson_x"] = float(corr_x) if not np.isnan(corr_x) else None
            except (ValueError, FloatingPointError):
                result["pearson_x"] = None
        else:
            result["pearson_x"] = None

        # Pearson correlation for Y
        if len(self.src_y_values) > 1:
            try:
                corr_y = np.corrcoef(self.src_y_values, self.dst_y_values)[0, 1]
                result["pearson_y"] = float(corr_y) if not np.isnan(corr_y) else None
            except (ValueError, FloatingPointError):
                result["pearson_y"] = None
        else:
            result["pearson_y"] = None

        # Scale ratio for X and Y (from paper: mean absolute value ratio)
        if self.src_sequences and self.dst_sequences:
            src_x = [abs(seq[0]) for seq in self.src_sequences]
            src_y = [abs(seq[1]) for seq in self.src_sequences]
            dst_x = [abs(seq[0]) for seq in self.dst_sequences]
            dst_y = [abs(seq[1]) for seq in self.dst_sequences]

            src_x_mean = np.mean(src_x)
            dst_x_mean = np.mean(dst_x)
            src_y_mean = np.mean(src_y)
            dst_y_mean = np.mean(dst_y)

            # Scale ratio: ensure >= 1 (invert if < 1)
            if dst_x_mean > 0:
                scale_x = src_x_mean / dst_x_mean
                result["scale_ratio_x"] = float(scale_x if scale_x >= 1 else 1 / scale_x)
            else:
                result["scale_ratio_x"] = None

            if dst_y_mean > 0:
                scale_y = src_y_mean / dst_y_mean
                result["scale_ratio_y"] = float(scale_y if scale_y >= 1 else 1 / scale_y)
            else:
                result["scale_ratio_y"] = None
        else:
            result["scale_ratio_x"] = None
            result["scale_ratio_y"] = None

        result["sample_count"] = len(self.src_x_values)
        return result


class MouseButtonMetric:
    """Mouse button metrics: per-button count accuracy."""

    name = "mouse_button"

    def __init__(self, cfg: BinningConfig):
        self.empty_bins_as_correct = cfg.empty_bins_as_correct
        self.button_accuracies: List[float] = []

    def update(self, src_bin: Bin, dst_bin: Bin):
        """Update the metric with a new bin."""
        src_counts = self._count_button_events(src_bin)
        dst_counts = self._count_button_events(dst_bin)

        all_buttons = set(src_counts.keys()) | set(dst_counts.keys())
        if not all_buttons:
            if self.empty_bins_as_correct:
                self.button_accuracies.append(1.0)
            return

        for button in all_buttons:
            src_count = src_counts.get(button, 0)
            dst_count = dst_counts.get(button, 0)
            accuracy = 1.0 if src_count == dst_count else 0.0
            self.button_accuracies.append(accuracy)

    def _count_button_events(self, bin_data: Bin) -> Dict[str, int]:
        """Count button events by type in a bin."""
        counts = defaultdict(int)
        for event in bin_data["events"]:
            if _determine_event_type(event) == EventType.MOUSE_OP:
                decoded = event.decoded
                flags = decoded.button_flags
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN:
                    counts["left_down"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP:
                    counts["left_up"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN:
                    counts["right_down"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP:
                    counts["right_up"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_DOWN:
                    counts["middle_down"] += 1
                if flags & RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_UP:
                    counts["middle_up"] += 1
        return dict(counts)

    @property
    def metric(self) -> Dict[str, Any]:
        """Return the current metric value."""
        if self.button_accuracies:
            return {
                "button_accuracy": float(np.mean(self.button_accuracies)),
                "sample_count": len(self.button_accuracies),
            }
        return {"button_accuracy": None, "sample_count": 0}


class KeyboardMetric:
    """Keyboard metrics: per-key count accuracy."""

    name = "keyboard"

    def __init__(self, cfg: BinningConfig):
        self.empty_bins_as_correct = cfg.empty_bins_as_correct
        self.key_accuracies: List[float] = []

    def update(self, src_bin: Bin, dst_bin: Bin):
        """Update the metric with a new bin."""
        src_counts = self._count_key_events(src_bin)
        dst_counts = self._count_key_events(dst_bin)

        all_keys = set(src_counts.keys()) | set(dst_counts.keys())
        if not all_keys:
            if self.empty_bins_as_correct:
                self.key_accuracies.append(1.0)
            return

        for key in all_keys:
            src_count = src_counts.get(key, 0)
            dst_count = dst_counts.get(key, 0)
            accuracy = 1.0 if src_count == dst_count else 0.0
            self.key_accuracies.append(accuracy)

    def _count_key_events(self, bin_data: Bin) -> Dict[str, int]:
        """Count keyboard events by key and type in a bin."""
        counts = defaultdict(int)
        for event in bin_data["events"]:
            if _determine_event_type(event) == EventType.KEYBOARD:
                decoded = event.decoded
                if hasattr(decoded, "event_type") and hasattr(decoded, "vk"):
                    key = f"{decoded.event_type}_{decoded.vk}"
                    counts[key] += 1
        return dict(counts)

    @property
    def metric(self) -> Dict[str, Any]:
        """Return the current metric value."""
        if self.key_accuracies:
            return {
                "key_accuracy": float(np.mean(self.key_accuracies)),
                "sample_count": len(self.key_accuracies),
            }
        return {"key_accuracy": None, "sample_count": 0}


def _get_first_screen_info(reader: OWAMcapReader) -> Tuple[int, int]:
    """Get (mcap_timestamp, pts_ns) of the first screen event."""
    for msg in reader.iter_messages(topics=["screen"]):
        pts_ns = msg.decoded.media_ref.pts_ns
        return msg.timestamp, pts_ns
    raise ValueError("No screen events found in MCAP file")


def _normalize_events(reader: OWAMcapReader, offset: int, start_pts: int, end_pts: int) -> Iterator[McapMessage]:
    """Normalize event timestamps to pts_ns-based time and filter by range."""
    for event in reader.iter_messages(topics=["screen", "keyboard", "mouse/raw"]):
        normalized_ts = event.timestamp - offset
        if start_pts <= normalized_ts <= end_pts:
            event.timestamp = normalized_ts
            yield event


def _get_alignment_info(src_reader: OWAMcapReader, dst_reader: OWAMcapReader) -> AlignmentInfo:
    """Align two MCAP streams into video pts_ns time."""
    src_mcap_ts, src_pts = _get_first_screen_info(src_reader)
    dst_mcap_ts, dst_pts = _get_first_screen_info(dst_reader)

    src_offset = src_mcap_ts - src_pts
    dst_offset = dst_mcap_ts - dst_pts

    src_end_pts = src_pts + (src_reader.end_time - src_mcap_ts)
    dst_end_pts = dst_pts + (dst_reader.end_time - dst_mcap_ts)

    start_pts = max(src_pts, dst_pts)
    end_pts = min(src_end_pts, dst_end_pts)
    duration_sec = (end_pts - start_pts) / 1e9

    logger.info(f"Source: first_screen_ts={src_mcap_ts}, first_pts={src_pts}, offset={src_offset}")
    logger.info(f"Dest:   first_screen_ts={dst_mcap_ts}, first_pts={dst_pts}, offset={dst_offset}")
    logger.info(f"Common pts range: {start_pts} - {end_pts} (duration: {duration_sec:.2f}s)")

    if end_pts <= start_pts:
        raise ValueError(
            f"No temporal overlap between source and destination MCAP files. "
            f"Source pts range: [{src_pts}, {src_end_pts}], "
            f"Destination pts range: [{dst_pts}, {dst_end_pts}]"
        )

    return AlignmentInfo(src_offset, dst_offset, start_pts, end_pts, duration_sec)


def compute_binned_metrics(src_mcap_path: str, dst_mcap_path: str, cfg: BinningConfig) -> Dict[str, Any]:
    """
    Compute binned metrics from two MCAP files.

    Timestamps are aligned based on the first screen event's pts_ns in each file.
    This allows comparing mcap files with different absolute timestamps (e.g., 2026 vs 1970).

    Args:
        src_mcap_path: Path to source/ground truth MCAP file
        dst_mcap_path: Path to predicted/destination MCAP file
        cfg: Binning configuration

    Returns:
        Dictionary of computed metrics
    """
    with OWAMcapReader(src_mcap_path) as src_reader, OWAMcapReader(dst_mcap_path) as dst_reader:
        alignment = _get_alignment_info(src_reader, dst_reader)

        # Create bins in normalized (pts_ns-based) time
        bins = create_bins(alignment.start_pts, alignment.end_pts, cfg.bin_size_ns)
        logger.info(f"Created {len(bins)} bins of size {cfg.bin_size_ns / 1e6:.1f}ms")

        # Get normalized events from both files
        src_events = _normalize_events(src_reader, alignment.src_offset, alignment.start_pts, alignment.end_pts)
        dst_events = _normalize_events(dst_reader, alignment.dst_offset, alignment.start_pts, alignment.end_pts)

        # Bin the events
        src_bins = bin_events(src_events, copy.deepcopy(bins))
        dst_bins = bin_events(dst_events, copy.deepcopy(bins))

        # Initialize metrics
        metrics = [
            MouseMoveMetric(cfg),
            MouseButtonMetric(cfg),
            KeyboardMetric(cfg),
        ]

        # Process each bin pair
        for src_bin, dst_bin in zip(src_bins, dst_bins):
            for m in metrics:
                m.update(src_bin, dst_bin)

        # Aggregate results
        results: Dict[str, Any] = {
            "config": {
                "bin_size_ms": cfg.bin_size_ns / 1e6,
                "empty_bins_as_correct": cfg.empty_bins_as_correct,
            },
            "summary": {
                "duration_sec": alignment.duration_sec,
                "num_bins": len(bins),
            },
        }

        for m in metrics:
            metric_results = m.metric
            results[m.name] = metric_results

        return results


VK_NAMES = {
    1: "MouseLeft",
    2: "MouseRight",
    8: "Backspace",
    9: "Tab",
    13: "Enter",
    16: "Shift",
    17: "Ctrl",
    18: "Alt",
    27: "Esc",
    32: "Space",
    37: "Left",
    38: "Up",
    39: "Right",
    40: "Down",
    65: "A",
    68: "D",
    83: "S",
    87: "W",
    112: "F1",
    113: "F2",
    160: "LShift",
    161: "RShift",
    162: "LCtrl",
    163: "RCtrl",
}


def _field(obj: Any, *names: str, default: Any = 0) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _button_names(flags: int) -> List[str]:
    checks = [
        ("left down", RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN),
        ("left up", RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP),
        ("right down", RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN),
        ("right up", RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP),
        ("middle down", RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_DOWN),
        ("middle up", RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_UP),
    ]
    return [name for name, flag in checks if flags & flag]


def _event_lines(bin_data: Bin) -> List[str]:
    mouse_dx = 0
    mouse_dy = 0
    buttons: List[str] = []
    keys: List[str] = []

    for event in bin_data["events"]:
        event_type = _determine_event_type(event)
        decoded = event.decoded
        if event_type == EventType.MOUSE_NOP:
            mouse_dx += int(_field(decoded, "dx", "last_x"))
            mouse_dy += int(_field(decoded, "dy", "last_y"))
        elif event_type == EventType.MOUSE_OP:
            flags = int(_field(decoded, "button_flags"))
            buttons.extend(_button_names(flags) or [f"button_flags={flags}"])
        elif event_type == EventType.KEYBOARD:
            vk = int(_field(decoded, "vk", default=-1))
            event_name = str(_field(decoded, "event_type", default="key"))
            keys.append(f"{event_name} {VK_NAMES.get(vk, f'VK{vk}')}")

    lines = [f"events: {len(bin_data['events'])}"]
    if mouse_dx or mouse_dy:
        lines.append(f"mouse move: dx={mouse_dx:+d}, dy={mouse_dy:+d}")
    if buttons:
        lines.append("mouse: " + ", ".join(buttons[:8]))
    if keys:
        lines.append("keys: " + ", ".join(keys[:8]))
    if len(buttons) > 8 or len(keys) > 8:
        lines.append("...")
    if len(lines) == 1:
        lines.append("no action in this bin")
    return lines


def _draw_text_block(
    image: np.ndarray,
    title: str,
    lines: List[str],
    x: int,
    y: int,
    width: int,
    height: int,
    color: Tuple[int, int, int],
) -> None:
    cv2.rectangle(image, (x, y), (x + width, y + height), (34, 34, 34), -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    cv2.putText(image, title, (x + 14, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    cursor_y = y + 68
    max_chars = max(20, (width - 28) // 10)
    for line in lines:
        chunks = [line[i : i + max_chars] for i in range(0, len(line), max_chars)] or [""]
        for chunk in chunks:
            if cursor_y > y + height - 18:
                return
            cv2.putText(
                image,
                chunk,
                (x + 14, cursor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            cursor_y += 26


def _resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = target_width / width
    return cv2.resize(frame, (target_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _video_duration_ns(cap: cv2.VideoCapture) -> int | None:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps and fps > 0 and frame_count and frame_count > 0:
        return int(frame_count / fps * 1e9)
    return None


def play_synchronized_viewer(
    video_path: str,
    ground_truth_mcap_path: str,
    predicted_mcap_path: str,
    cfg: BinningConfig,
    display_width: int = 960,
    speed: float = 1.0,
) -> None:
    """Play video with ground-truth and predicted MCAP action panels."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    with OWAMcapReader(ground_truth_mcap_path) as gt_reader, OWAMcapReader(predicted_mcap_path) as pred_reader:
        alignment = _get_alignment_info(gt_reader, pred_reader)

        duration_ns = _video_duration_ns(cap)
        if duration_ns is not None:
            alignment.end_pts = min(alignment.end_pts, duration_ns)
            alignment.duration_sec = (alignment.end_pts - alignment.start_pts) / 1e9

        bins = create_bins(alignment.start_pts, alignment.end_pts, cfg.bin_size_ns)
        gt_bins = bin_events(
            _normalize_events(gt_reader, alignment.src_offset, alignment.start_pts, alignment.end_pts),
            copy.deepcopy(bins),
        )
        pred_bins = bin_events(
            _normalize_events(pred_reader, alignment.dst_offset, alignment.start_pts, alignment.end_pts),
            copy.deepcopy(bins),
        )

    if not bins:
        raise ValueError("No bins to display after aligning video and MCAP files")

    panel_width = 430
    current_pts = alignment.start_pts
    paused = False
    wall_start = time.perf_counter()
    pts_start = current_pts

    logger.info("Viewer controls: q/esc quit, space pause, left/right one bin, a/d one second")

    while True:
        if not paused:
            elapsed = (time.perf_counter() - wall_start) * speed
            current_pts = pts_start + int(elapsed * 1e9)

        if current_pts >= alignment.end_pts:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, current_pts / 1e6))
        ok, frame = cap.read()
        if not ok:
            break

        frame = _resize_frame(frame, display_width)
        frame_h, frame_w = frame.shape[:2]
        canvas = np.zeros((frame_h, frame_w + panel_width, 3), dtype=np.uint8)
        canvas[:, :frame_w] = frame

        bin_idx = min(len(bins) - 1, max(0, int((current_pts - alignment.start_pts) // cfg.bin_size_ns)))
        t_sec = (current_pts - alignment.start_pts) / 1e9
        status = "PAUSED" if paused else f"{speed:.2f}x"
        header = f"{t_sec:7.2f}s / {alignment.duration_sec:7.2f}s  bin {bin_idx + 1}/{len(bins)}  {status}"
        cv2.rectangle(canvas, (0, 0), (frame_w, 38), (0, 0, 0), -1)
        cv2.putText(canvas, header, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        panel_x = frame_w
        half_h = frame_h // 2
        _draw_text_block(canvas, "Ground truth MCAP", _event_lines(gt_bins[bin_idx]), panel_x, 0, panel_width, half_h, (80, 220, 120))
        _draw_text_block(
            canvas,
            "Predicted MCAP",
            _event_lines(pred_bins[bin_idx]),
            panel_x,
            half_h,
            panel_width,
            frame_h - half_h,
            (80, 170, 255),
        )

        cv2.imshow("D2E video + MCAP comparison", canvas)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            wall_start = time.perf_counter()
            pts_start = current_pts
        elif key in (81, ord("a")):
            current_pts = max(alignment.start_pts, current_pts - (cfg.bin_size_ns if key == 81 else 1_000_000_000))
            paused = True
        elif key in (83, ord("d")):
            current_pts = min(alignment.end_pts - 1, current_pts + (cfg.bin_size_ns if key == 83 else 1_000_000_000))
            paused = True

    cap.release()
    cv2.destroyAllWindows()


def main():
    root = Path(__file__).resolve().parent
    default_video = root.parent / "open-world-agents/data/cyberpunk_test_trial2/trial2.mkv"
    default_ground_truth = root.parent / "open-world-agents/data/cyberpunk_test_trial2/trial2.mcap"
    parser = argparse.ArgumentParser(
        description="G-IDM evaluation: compare ground truth MCAP with predicted MCAP while viewing video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run evaluate_video.py video.mkv ground_truth.mcap predicted.mcap
    uv run evaluate_video.py video.mkv gt.mcap pred.mcap --bin-ms 100
    uv run evaluate_video.py video.mkv gt.mcap pred.mcap --output results.json --no-viewer
""",
    )
    parser.add_argument("--video", type=str, default=str(default_video), help="Path to the source video file")
    parser.add_argument("--ground_truth", type=str, default=str(default_ground_truth), help="Path to ground truth MCAP file")
    parser.add_argument("--predicted", type=str, default="predicted.mcap", help="Path to predicted MCAP file")
    parser.add_argument("--bin-ms", type=int, default=50, help="Bin size in milliseconds (default: 50)")
    parser.add_argument(
        "--empty-bins-as-correct",
        action="store_true",
        help="Treat empty bins as correct matches",
    )
    parser.add_argument("--output", "-o", type=str, help="Save JSON output to file")
    parser.add_argument("--no-viewer", action="store_true", help="Only compute metrics; do not open the synchronized viewer")
    parser.add_argument("--display-width", type=int, default=960, help="Displayed video width in pixels (default: 960)")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier (default: 1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    video_path = Path(args.video)
    gt_path = Path(args.ground_truth)
    pred_path = Path(args.predicted)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth MCAP not found: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predicted MCAP not found: {pred_path}")
    if args.speed <= 0:
        raise ValueError("--speed must be greater than 0")
    if args.display_width <= 0:
        raise ValueError("--display-width must be greater than 0")

    cfg = BinningConfig(
        bin_size_ns=args.bin_ms * 1_000_000,
        empty_bins_as_correct=args.empty_bins_as_correct,
    )

    logger.info(f"Evaluating: {gt_path} vs {pred_path}")
    results = compute_binned_metrics(str(gt_path), str(pred_path), cfg)

    # Print results
    output = json.dumps(results, indent=2)
    print(output)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    logger.success("Evaluation complete")

    if not args.no_viewer:
        logger.info(f"Opening synchronized viewer for: {video_path}")
        play_synchronized_viewer(
            str(video_path),
            str(gt_path),
            str(pred_path),
            cfg,
            display_width=args.display_width,
            speed=args.speed,
        )


if __name__ == "__main__":
    main()
