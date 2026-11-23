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


@dataclass(frozen=True)
class Detection:
    head: Point | None
    confidence: float
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

    def predict(self, frames: list[np.ndarray], image_size: int) -> list[Detection]:
        with self.lock:
            results = self.model.predict(
                frames, device="cpu", imgsz=image_size, conf=0.24,
                classes=[HEAD, SHAFT], max_det=12, verbose=False,
            )
        detections = []
        for result in results:
            head = None
            confidence = 0.0
            shafts = []
            for x1, y1, x2, y2, score, label in result.boxes.data.cpu().tolist():
                if int(label) == HEAD and score > confidence:
                    head = (round((x1 + x2) / 2), round((y1 + y2) / 2))
                    confidence = score
                if int(label) == SHAFT:
                    shafts.append(tuple(map(round, (x1, y1, x2, y2))))
            detections.append(Detection(head, confidence, tuple(shafts)))
        return detections


def estimate_path(
    detections: list[Detection], corrections: dict[int, Point | None],
    fps: float, width: int, height: int,
) -> list[Sample]:
    def clip(point: Point) -> Point:
        return min(width - 1, max(0, point[0])), min(height - 1, max(0, point[1]))

    def distance(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def interpolate(a: Point, b: Point, ratio: float) -> Point:
        return round(a[0] + (b[0] - a[0]) * ratio), round(a[1] + (b[1] - a[1]) * ratio)

    path = [Sample(clip(d.head), "head", d.confidence < 0.5) if d.head is not None
            else Sample(None, "missing", True) for d in detections]
    for index, point in corrections.items():
        path[index] = Sample(clip(point), "manual") if point is not None else Sample(None, "excluded")

    diagonal = float(np.hypot(width, height))
    for index in range(1, len(path) - 1):
        prev, current, after = path[index - 1:index + 2]
        if current.source == "head" and all(p.point is not None for p in (prev, after)):
            step = distance(prev.point, after.point)
            if distance(current.point, interpolate(prev.point, after.point, 0.5)) > max(
                0.035 * diagonal, 0.85 * step
            ):
                path[index] = Sample(current.point, "head", True)

    # Use nearby head detections or manual points as shaft-end references.
    # An explicit exclusion breaks the trajectory.
    anchors = [i for i, s in enumerate(path) if s.point is not None or s.source == "excluded"]
    shaft_span = max(1, round(fps * 0.16))
    gap_span = max(1, round(fps * 0.12))
    for left, right in zip(anchors, anchors[1:]):
        first, last = path[left].point, path[right].point
        if first is None or last is None or right - left <= 1 or right - left > shaft_span + 1:
            continue
        for index in range(left + 1, right):
            reference = interpolate(first, last, (index - left) / (right - left))
            corners = [
                clip((x, y))
                for x1, y1, x2, y2 in detections[index].shafts
                for x, y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2))
            ]
            if corners:
                guess = min(corners, key=lambda point: distance(point, reference))
                if distance(guess, reference) <= diagonal * 0.08:
                    path[index] = Sample(guess, "shaft", True)

    anchors = [i for i, s in enumerate(path) if s.point is not None or s.source == "excluded"]
    for left, right in zip(anchors, anchors[1:]):
        first, last = path[left].point, path[right].point
        if first is None or last is None or right - left > gap_span + 1:
            continue
        for index in range(left + 1, right):
            point = interpolate(first, last, (index - left) / (right - left))
            path[index] = Sample(point, "interpolated", True)
    return path
