"""Estimate swing landmarks and prepare a continuous, editable overlay."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from tracking import Sample


@dataclass(frozen=True)
class Timing:
    split: int | None
    end: int
    occlusion_found: bool


def estimate_timing(path: list[Sample], fps: float, height: int) -> Timing:
    valid = np.array([i for i, sample in enumerate(path) if sample.point is not None], dtype=int)
    fallback = Timing(None, int(valid[-1]) if len(valid) else len(path) - 1, False)
    if len(valid) < 5:
        return fallback

    first = int(valid[0])
    times = np.arange(first, int(valid[-1]) + 1)
    y = gaussian_filter1d(
        np.interp(times, valid, [path[i].point[1] for i in valid]),
        max(0.7, fps * 0.035), mode="nearest",
    )
    address = float(np.median(y[:max(3, round(fps * 0.16))]))
    tops, _ = find_peaks(-y, prominence=height * 0.13)
    for top in tops:
        lift = address - y[top]
        if lift < height * 0.16:
            continue
        descent = np.flatnonzero(y[top + 1:] >= y[top] + 0.78 * lift)
        if not len(descent):
            continue
        impact = top + 1 + int(descent[0])
        if not np.any(np.abs(valid - (top + first)) <= max(1, round(fps * 0.08))):
            continue
        reversals, _ = find_peaks(y[top:impact], prominence=height * 0.006)
        split = first + int(top) + (int(reversals[0]) if len(reversals) else 0)
        near_impact = np.flatnonzero(y[impact:] >= y[top] + 0.90 * lift)
        after = first + impact + (int(near_impact[0]) if len(near_impact) else 0)
        gap_size = max(3, round(fps * 0.20))
        end = fallback.end
        occlusion_found = end < len(path) - 1
        for left, right in zip(valid, valid[1:]):
            if left >= after and right - left - 1 >= gap_size:
                end, occlusion_found = int(left), True
                break
        return Timing(split, max(split, end), occlusion_found)
    return fallback


def smooth_path(path: list[Sample], fps: float, width: int, height: int) -> list[Sample]:
    """Smooth in contiguous stretches while leaving clicked positions exact."""
    updated = list(path)
    valid = [i for i, sample in enumerate(path) if sample.point is not None]
    max_gap = max(1, round(fps * 0.16))
    diagonal = float(np.hypot(width, height))
    for left, right in zip(valid, valid[1:]):
        gap = right - left - 1
        between = path[left + 1:right]
        confirmed_ends = path[left].source == path[right].source == "manual"
        gap_limit = max(max_gap, round(fps * 0.36)) if confirmed_ends else max_gap
        if not 0 < gap <= gap_limit or any(sample.source == "excluded" for sample in between):
            continue
        start, end = np.array(path[left].point), np.array(path[right].point)
        if np.linalg.norm(end - start) > diagonal * (0.55 if confirmed_ends else 0.16):
            continue

        # Preserve the curvature supported by the adjacent observed positions.
        indices = [left, right]
        if left and path[left - 1].point is not None:
            indices.insert(0, left - 1)
        if right + 1 < len(path) and path[right + 1].point is not None:
            indices.append(right + 1)
        interpolate = PchipInterpolator(indices, [path[i].point for i in indices], axis=0)
        for i in range(left + 1, right):
            updated[i] = Sample(tuple(np.rint(interpolate(i)).astype(int)), "interpolated", True)

    active = np.array([i for i, sample in enumerate(updated) if sample.point is not None])
    for group in np.split(active, np.flatnonzero(np.diff(active) > 1) + 1):
        if len(group) < 3:
            continue
        original = np.asarray([updated[i].point for i in group], dtype=float)
        filtered = gaussian_filter1d(original, np.clip(fps * 0.047, 0.85, 3.5), axis=0)
        for offset, i in enumerate(group[1:-1], 1):
            sample = updated[i]
            if sample.source == "manual":
                continue
            delta = filtered[offset] - original[offset]
            limit = diagonal * (0.009 if sample.suspect else 0.0045)
            scale = min(1.0, limit / max(float(np.linalg.norm(delta)), 1e-9))
            point = original[offset] + delta * scale
            updated[i] = Sample(tuple(np.rint(point).astype(int)), sample.source, sample.suspect)
    return updated


def curve_segments(path: list[Sample], end: int, width: int, height: int) -> dict[int, np.ndarray]:
    """Build short subpixel curves for the preview and MP4 to share."""
    active = np.array([i for i, sample in enumerate(path[:end + 1]) if sample.point is not None])
    curves = {}
    for group in np.split(active, np.flatnonzero(np.diff(active) > 1) + 1):
        if len(group) < 2:
            continue
        points = np.array([path[i].point for i in group])
        curve = CubicSpline(group, points, axis=0, bc_type="natural")
        for i in group[1:]:
            vertices = curve(np.linspace(i - 1, i, 10))
            # Limit spline overshoot around sharp turns.
            pair = points[i - group[0] - 1:i - group[0] + 1]
            lo = np.maximum(pair.min(axis=0) - 3, 0)
            hi = np.minimum(pair.max(axis=0) + 3, [width - 1, height - 1])
            curves[int(i)] = np.rint(np.clip(vertices, lo, hi) * 16).astype(np.int32)
    return curves
