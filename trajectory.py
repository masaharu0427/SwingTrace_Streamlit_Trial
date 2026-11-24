"""Estimate swing landmarks and prepare a continuous, editable overlay."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
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


def smooth_path(path: list[Sample], fps: float, width: int, height: int, end: int) -> list[Sample]:
    """Fit a smooth curve to detected points within the chosen drawing range."""
    updated = list(path)
    stop = min(end + 1, len(path))
    breaks = [-1, *(i for i in range(stop) if path[i].source == "excluded"), stop]
    for start, finish in zip(breaks, breaks[1:]):
        known = [i for i in range(start + 1, finish) if path[i].point is not None]
        if len(known) < 2:
            continue
        first, last = known[0], known[-1]
        frames = np.arange(first, last + 1)
        observed = np.asarray([path[i].point for i in known], dtype=float)
        if len(known) < 3 or len(frames) < 4:
            fitted = PchipInterpolator(known, observed, axis=0)(frames)
        else:
            fitted = fit_points(path, known, frames, observed, fps, float(np.hypot(width, height)))
        fitted = np.rint(np.clip(fitted, [0, 0], [width - 1, height - 1])).astype(int)
        for index, point in zip(frames, fitted):
            original = path[index]
            if original.source == "manual":
                continue
            updated[index] = Sample(tuple(map(int, point)), original.source, original.suspect) if original.point is not None else (
                Sample(tuple(map(int, point)), "interpolated", True)
            )
    return updated


def fit_points(path, known, frames, observed, fps, diagonal):
    first, count = int(frames[0]), len(frames)
    local_known = np.asarray(known) - first
    observations = np.zeros((count, 2))
    observations[local_known] = observed
    weights = np.zeros(count)
    for index in known:
        sample = path[index]
        weights[index - first] = (
            (0.55 if sample.suspect else 1.0) if sample.source == "head" else
            0.25 if sample.source == "shaft" else 0.08
        )
    # Trust observed head positions more where fast motion is visible in successive frames.
    frame_steps = np.diff(local_known)
    speeds = np.linalg.norm(np.diff(observed, axis=0), axis=1) / frame_steps
    speeds[frame_steps != 1] = 0
    adjacent_speed = np.maximum(np.r_[0, speeds], np.r_[speeds, 0])
    for offset, index in enumerate(known):
        if path[index].source == "head":
            boost = 1 + 8 * min(1, (adjacent_speed[offset] * fps / diagonal) ** 2)
            weights[index - first] *= boost
    base_weights = weights.copy()

    # Fix the visible endpoints and all corrections without resampling the clicks.
    fixed = sorted({0, count - 1, *(i - first for i in known if path[i].source == "manual")})
    free = np.setdiff1d(np.arange(count), fixed, assume_unique=True)
    points = observations.copy()
    if not len(free):
        return points

    difference = diags([-1., 3., -3., 1.], [0, 1, 2, 3],
                       shape=(count - 3, count), format="csc")
    strength = 110 * np.clip(fps / 30, 0.65, 6) ** 6
    stiffness = strength * (difference.T @ difference)
    fixed_term = stiffness[free, :][:, fixed] @ points[fixed]
    free_stiffness = stiffness[free, :][:, free]

    for _ in range(2):
        system = free_stiffness + diags(weights[free], format="csc")
        points[free] = spsolve(system, weights[free, None] * observations[free] - fixed_term)
        # A detection that briefly moves from the center to the heel should not bend the path.
        residual = np.linalg.norm(points[free] - observations[free], axis=1)
        robust = np.clip(diagonal * 0.012 / np.maximum(residual, 1e-6), 0.60, 1.0)
        weights[free] = base_weights[free] * robust
    return points


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
            curves[int(i)] = np.rint(np.clip(vertices, [0, 0], [width - 1, height - 1]) * 16).astype(np.int32)
    return curves
