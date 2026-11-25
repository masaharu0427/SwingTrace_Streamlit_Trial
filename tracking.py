"""Inference with the original SwingTrace weights and editable 2-D trajectory."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
import threading
import urllib.request
from zipfile import is_zipfile

import numpy as np

Point = tuple[int, int]
Box = tuple[int, int, int, int]
MODEL_SHA256 = "fcda5469a9cede0d4745d15dc797cd2ebcca4bda8b4e7e717b0b8c00a56d8744"
MODEL_URL = (
    "https://raw.githubusercontent.com/nishsm/golf-club-detection-swingtrace/"
    "609af10e19bae6741e016c41caa966c6e6124faf/weights/best.pt"
)

HEAD = 1
SHAFT = 0
HANDS = 2
STRONG_HEAD = 0.28


@dataclass(frozen=True)
class Detection:
    head: Point | None
    confidence: float
    shaft_tip: Point | None = None
    heads: tuple[tuple[Point, float], ...] = ()
    shafts: tuple[Box, ...] = ()


@dataclass(frozen=True)
class Sample:
    point: Point | None
    source: str
    suspect: bool = False


def model_path() -> Path:
    path = Path(__file__).resolve().parent / "weights" / "best.pt"
    if path.exists():
        with path.open("rb") as existing:
            if file_digest(existing, "sha256").hexdigest() == MODEL_SHA256:
                return path
        raise RuntimeError("weights/best.pt が公開モデルと一致しません。READMEの手順で入れ直してください。")

    path.parent.mkdir(exist_ok=True)
    partial = path.with_suffix(".download")
    digest = sha256()
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response, partial.open("wb") as output:
            size = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > 50 * 1024 * 1024:
                    raise RuntimeError("モデルのダウンロードサイズが上限を超えました。")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != MODEL_SHA256 or not is_zipfile(partial):
            raise RuntimeError("ダウンロードしたモデルが公開モデルと一致しません。")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


class Detector:
    def __init__(self) -> None:
        from ultralytics import YOLO
        import torch

        if torch.get_num_threads() > 2:
            torch.set_num_threads(2)
        self.model = YOLO(str(model_path()), task="detect")
        if [str(self.model.names[i]) for i in range(len(self.model.names))] != ["0", "1", "3"]:
            raise RuntimeError("SwingTraceのクラス構成を確認できません。")
        self.lock = threading.Lock()

    def infer(self, frames: list[np.ndarray], size: int, offset=(0, 0)) -> list[Detection]:
        with self.lock:
            results = self.model.predict(
                frames, device="cpu", imgsz=size, conf=0.09,
                classes=[HEAD, SHAFT, HANDS], max_det=24, verbose=False,
            )
        detections = []
        for result in results:
            heads, shafts, hands = [], [], []
            ox, oy = offset
            for x1, y1, x2, y2, score, label in result.boxes.data.cpu().tolist():
                box = (round(x1 + ox), round(y1 + oy), round(x2 + ox), round(y2 + oy))
                center = (round((x1 + x2) / 2 + ox), round((y1 + y2) / 2 + oy))
                if int(label) == HEAD:
                    heads.append((center, score))
                elif int(label) == SHAFT and score >= 0.30:
                    shafts.append((box, score))
                elif int(label) == HANDS and score >= 0.28:
                    hands.append((center, score))
            tip = shaft_tip(shafts, hands, float(np.hypot(*result.orig_shape)))
            detections.append(make_detection(heads, tip, tuple(box for box, _ in shafts)))
        return detections

    def predict(self, frames: list[np.ndarray], image_size: int) -> list[Detection]:
        base = self.infer(frames, min(640, image_size))
        if image_size <= 640:
            return base

        # Keep the original detections as anchors so retries cannot propagate a false hit.
        anchors = [d.head if d.confidence >= 0.50 else None for d in base]
        missing = [i for i, detection in enumerate(base) if detection.confidence < STRONG_HEAD]
        for start in range(0, len(missing), 2):
            indices = missing[start:start + 2]
            retry = self.infer([frames[i] for i in indices], image_size)
            for i, secondary in zip(indices, retry):
                detected = base[i]
                tip = secondary.shaft_tip or detected.shaft_tip
                heads = detected.heads + secondary.heads
                center = tip
                if center is None:
                    before = anchors[i - 1] if i > 0 else None
                    after = anchors[i + 1] if i + 1 < len(anchors) else None
                    if before is not None and after is not None:
                        center = tuple(round((a + b) / 2) for a, b in zip(before, after))
                if center is not None and max((score for _, score in heads), default=0) < STRONG_HEAD:
                    frame = frames[i]
                    height, width = frame.shape[:2]
                    crop_width = max(1, round(width * 0.52))
                    crop_height = max(1, round(height * 0.52))
                    x = min(width - crop_width, max(0, center[0] - crop_width // 2))
                    y = min(height - crop_height, max(0, center[1] - crop_height // 2))
                    crop = frame[y:y + crop_height, x:x + crop_width]
                    zoomed = self.infer([crop], 640, (x, y))[0]
                    radius = 0.10 * float(np.hypot(width, height))
                    heads += tuple((point, score) for point, score in zoomed.heads
                                   if distance(point, center) <= radius)
                base[i] = make_detection(heads, tip, detected.shafts + secondary.shafts)
        return base


def distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def shaft_tip(shafts, hands, diagonal: float) -> Point | None:
    """The shaft end opposite the detected grip, when both boxes agree."""
    best = None
    best_cost = float("inf")
    for (x1, y1, x2, y2), shaft_confidence in shafts:
        length = distance((x1, y1), (x2, y2))
        if not 0.045 * diagonal <= length <= 0.48 * diagonal:
            continue
        corners = ((x1, y1), (x1, y2), (x2, y1), (x2, y2))
        for hand, hand_confidence in hands:
            ordered = sorted(range(4), key=lambda i: distance(hand, corners[i]))
            grip, alternate = ordered[:2]
            error = distance(hand, corners[grip])
            if error > min(0.055 * diagonal, max(0.018 * diagonal, 0.24 * length)):
                continue
            if (distance(hand, corners[alternate]) - error < 0.008 * diagonal
                    and distance(corners[grip], corners[alternate]) > 0.045 * diagonal):
                continue
            cost = error / length - 0.20 * (shaft_confidence + hand_confidence)
            if cost < best_cost:
                best = corners[3 - grip]
                best_cost = cost
    return best


def make_detection(heads, tip: Point | None, shafts: tuple[Box, ...] = ()) -> Detection:
    candidates = tuple(sorted(heads, key=lambda hit: hit[1], reverse=True)[:8])
    point, score = candidates[0] if candidates else (None, 0.0)
    return Detection(point, score, tip, candidates, shafts)


def estimate_path(
    detections: list[Detection], corrections: dict[int, Point | None],
    fps: float, width: int, height: int,
) -> list[Sample]:
    def clip(point: Point) -> Point:
        return min(width - 1, max(0, point[0])), min(height - 1, max(0, point[1]))

    def interpolate(a: Point, b: Point, ratio: float) -> Point:
        return round(a[0] + (b[0] - a[0]) * ratio), round(a[1] + (b[1] - a[1]) * ratio)

    diagonal = float(np.hypot(width, height))
    path = [Sample(clip(d.head), "head", d.confidence < 0.5)
            if d.head is not None and d.confidence >= STRONG_HEAD
            else Sample(None, "missing", True) for d in detections]
    for index, point in corrections.items():
        path[index] = Sample(clip(point), "manual") if point is not None else Sample(None, "excluded")

    # Discard isolated jumps when the surrounding positions agree.
    for index in range(1, len(path) - 1):
        prev, current, after = path[index - 1:index + 2]
        if current.source != "head" or prev.point is None or after.point is None:
            continue
        step = distance(prev.point, after.point)
        offset = distance(current.point, interpolate(prev.point, after.point, 0.5))
        if offset > max(0.045 * diagonal, 1.15 * step):
            path[index] = Sample(None, "missing", True)

    anchor_before = None
    nearby = max(2, round(fps * 0.16))
    references: list[Point | None] = [None] * len(path)
    bracketed = set()
    anchors = [i for i, sample in enumerate(path) if sample.point is not None or sample.source == "excluded"]
    right_anchors = iter(anchors)
    anchor_after = next(right_anchors, None)
    for i, sample in enumerate(path):
        if i == anchor_after:
            anchor_before = i
            anchor_after = next(right_anchors, None)
            continue
        left = path[anchor_before].point if anchor_before is not None and i - anchor_before <= nearby else None
        right = path[anchor_after].point if anchor_after is not None and anchor_after - i <= nearby else None
        if left is not None and right is not None:
            references[i] = interpolate(left, right, (i - anchor_before) / (anchor_after - anchor_before))
            bracketed.add(i)
        else:
            references[i] = left if left is not None else right

    previous_pass = list(path)
    for i, detection in enumerate(detections):
        if previous_pass[i].source != "missing":
            continue
        reference, tip = references[i], detection.shaft_tip
        # Consecutive shaft guesses can lock onto the same background line.
        tip_agrees = i in bracketed and tip is not None and distance(reference, tip) <= 0.045 * diagonal
        candidates = [
            (point, score) for point, score in detection.heads
            if reference is not None and (
                distance(reference, point) <= 0.055 * diagonal
                or (i in bracketed and score >= 0.18 and tip is not None
                    and distance(tip, point) <= 0.025 * diagonal
                    and distance(reference, point) <= 0.12 * diagonal)
            )
        ]
        if candidates:
            point, _ = max(candidates, key=lambda hit: hit[1])
            path[i] = Sample(clip(point), "head", True)
        elif tip_agrees:
            path[i] = Sample(clip(tip), "shaft", True)
        elif i in bracketed and detection.shafts:
            corners = [(x, y) for x1, y1, x2, y2 in detection.shafts
                       for x in (x1, x2) for y in (y1, y2)]
            closest = min(corners, key=lambda corner: distance(reference, corner))
            if distance(reference, closest) <= 0.03 * diagonal:
                path[i] = Sample(clip(closest), "shaft", True)

    gap_span = max(1, round(fps * 0.14))
    anchors = [i for i, sample in enumerate(path) if sample.point is not None or sample.source == "excluded"]
    for left, right in zip(anchors, anchors[1:]):
        first, last = path[left].point, path[right].point
        span = right - left
        if first is None or last is None or not 1 < span <= gap_span + 1:
            continue
        if distance(first, last) > diagonal * min(0.34, span * 0.11):
            # Fast, short gaps can still be joined when both sides show the same motion.
            before = path[left - 1].point if left > 0 else None
            after = path[right + 1].point if right + 1 < len(path) else None
            if (before is None or after is None or path[left].source not in ("head", "manual")
                    or path[right].source not in ("head", "manual")
                    or distance(first, last) > 0.43 * diagonal):
                continue
            gap_velocity = (np.asarray(last) - first) / span
            entry_velocity = np.asarray(first) - before
            exit_velocity = np.asarray(after) - last
            speed = np.linalg.norm(gap_velocity)
            if not all(
                0.45 * speed <= np.linalg.norm(velocity) <= 2.2 * speed
                and np.dot(velocity, gap_velocity) >= 0.8 * np.linalg.norm(velocity) * speed
                for velocity in (entry_velocity, exit_velocity)
            ):
                continue
        for index in range(left + 1, right):
            point = interpolate(first, last, (index - left) / span)
            path[index] = Sample(point, "interpolated", True)
    return path
