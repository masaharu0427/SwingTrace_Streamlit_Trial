"""Normalize phone video and render an MP4 with a cumulative club-head trail."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np

from tracking import Sample
from trajectory import curve_segments

MAX_SECONDS = 30
MAX_FRAMES = 1800
WINDOWS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BLUE = (255, 126, 18)
RED = (36, 48, 255)


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frames: int


def ffmpeg() -> list[str]:
    return [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(ffmpeg() + args, capture_output=True, timeout=180,
                            creationflags=WINDOWS_FLAGS)
    if result.returncode:
        details = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"動画を変換できませんでした。{details[-500:]}")


@contextmanager
def open_video(path: Path):
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    try:
        if not capture.isOpened():
            raise RuntimeError("動画を開けませんでした。MP4またはMOVでお試しください。")
        yield capture
    finally:
        capture.release()


def normalize(source: Path, output: Path) -> VideoInfo:
    scale = "scale='min(1920,iw)':'min(1920,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1"
    run_ffmpeg([
        "-threads", "2", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
        "-t", str(MAX_SECONDS + 1), "-filter_threads", "2", "-vf", scale,
        "-fps_mode", "cfr", "-c:v", "libx264", "-threads:v", "2",
        "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", str(output),
    ])
    with open_video(output) as capture:
        info = VideoInfo(capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if info.frames < 1 or not np.isfinite(info.fps) or info.fps <= 0:
        raise RuntimeError("動画のフレーム情報を取得できませんでした。")
    if info.frames > MAX_FRAMES or info.frames / info.fps > MAX_SECONDS + 0.05:
        raise RuntimeError(f"{MAX_SECONDS}秒・{MAX_FRAMES:,}コマ以内のスイング動画に切り詰めてください。")
    return info


def analyze(path: Path, info: VideoInfo, detector, image_size: int, progress):
    detections = []
    with open_video(path) as capture:
        for start in range(0, info.frames, 4):
            frames = []
            for _ in range(min(4, info.frames - start)):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("動画の途中に読み取れないコマがありました。")
                frames.append(frame)
            detections.extend(detector.predict(frames, image_size))
            progress(len(detections), info.frames)
    return detections


def read_frame(path: Path, index: int) -> np.ndarray:
    with open_video(path) as capture:
        if index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    if not ok:
        raise RuntimeError("このコマを読み込めませんでした。")
    return frame


class Trail:
    def __init__(self, path: list[Sample], width: int, height: int, split: int | None, end: int):
        self.split = split
        self.end = end
        self.thickness = max(2, round(min(height, width) / 270))
        self.blue = np.zeros((height, width), dtype=np.uint8)
        self.red = np.zeros_like(self.blue)
        self.curves = curve_segments(path, end, width, height)

    def color(self, index: int) -> tuple[int, int, int]:
        return RED if self.split is not None and index > self.split else BLUE

    def extend(self, index: int) -> None:
        vertices = self.curves.get(index)
        if vertices is not None:
            mask = self.red if self.color(index) == RED else self.blue
            cv2.polylines(mask, [vertices], False, 255, self.thickness, cv2.LINE_AA, shift=4)

    def composite(self, frame: np.ndarray, sample: Sample, index: int, *, review=False) -> None:
        for mask, color in ((self.blue, BLUE), (self.red, RED)):
            indices = np.nonzero(mask)
            opacity = mask[indices].astype(np.float32)[:, None] / 255.0
            frame[indices] = frame[indices] * (1 - opacity) + np.asarray(color) * opacity
        if sample.point is None or index > self.end:
            return
        color = self.color(index)
        if review and sample.source == "manual":
            color = (255, 220, 40)
        elif review and sample.suspect:
            color = (30, 175, 255)
        radius = self.thickness + 4
        cv2.circle(frame, sample.point, radius + 1, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.circle(frame, sample.point, radius, color, 2, cv2.LINE_AA)


def preview_jpeg(source: Path, path: list[Sample], index: int, show_trail: bool,
                 *, split: int | None, end: int, review_sample: Sample) -> bytes:
    frame = read_frame(source, index)
    height, width = frame.shape[:2]
    trail = Trail(path, width, height, split, end)
    if show_trail:
        for i in range(1, min(index, end) + 1):
            trail.extend(i)
    trail.composite(frame, review_sample, index, review=True)
    ratio = min(1.0, 960 / max(height, width))
    if ratio < 1:
        frame = cv2.resize(frame, (round(width * ratio), round(height * ratio)),
                           interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("確認画像を作成できませんでした。")
    return jpeg.tobytes()


def export_video(source: Path, info: VideoInfo, path: list[Sample], target: Path, progress,
                 *, split: int | None, end: int) -> None:
    if len(path) != info.frames or not any(sample.point is not None for sample in path[:end + 1]):
        raise RuntimeError("保存できる軌道がありません。")
    trail = Trail(path, info.width, info.height, split, end)
    temporary = target.with_name("exporting.mp4")
    command = ffmpeg() + [
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{info.width}x{info.height}",
        "-framerate", f"{info.fps:.12g}", "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy", "-t", f"{info.frames / info.fps:.9f}",
        "-c:v", "libx264", "-threads:v", "2", "-preset", "veryfast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    try:
        with tempfile.TemporaryFile() as errors, open_video(source) as capture:
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors,
                                  creationflags=WINDOWS_FLAGS) as encoder:
                try:
                    for index, sample in enumerate(path):
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError("出力中に動画のコマを読み取れませんでした。")
                        trail.extend(index)
                        trail.composite(frame, sample, index)
                        encoder.stdin.write(frame.tobytes())
                        if index % 8 == 0 or index + 1 == info.frames:
                            progress(index + 1, info.frames)
                    encoder.stdin.close()
                    if encoder.wait(timeout=90):
                        raise RuntimeError("MP4エンコーダーが停止しました。")
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as error:
                    errors.seek(0)
                    detail = errors.read().decode(errors="replace").strip()
                    raise RuntimeError(f"MP4を書き出せませんでした。{detail[-500:]}") from error
                finally:
                    if encoder.poll() is None:
                        encoder.kill()
                        encoder.wait()
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
