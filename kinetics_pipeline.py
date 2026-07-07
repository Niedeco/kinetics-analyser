#!/usr/bin/env python3
"""
kinetics_pipeline.py

Automatic extraction of enzyme-kinetic initial rates from continuous UV-Vis
absorbance traces (Cary 60 CSV export at a single wavelength).

Workflow
--------
1. Read the Cary kinetics CSV (time/min, absorbance).
2. Detect measurement-interrupt events (cuvette removal: negative spikes;
   cuvette reinsertion / pipetting: large point-to-point jumps) and dilate
   them by a small buffer to eliminate instrument settle-down artefacts.
3. Split the clean trace into candidate segments; drop segments shorter than
   `--min-seg-sec` (these are instrument settle stubs, not real measurements).
4. Classify each surviving segment as BLANK or ENZYME. Default policy is
   alternating (first = blank, second = enzyme, ...) because this matches the
   usual workflow (substrate baseline, then add enzyme, repeat for next well).
   The classification is cross-checked against slope magnitude.
5. Fit the initial-rate window in each segment:
     * BLANK   -> linear fit over the whole (trimmed) segment.
     * ENZYME  -> automatic linear-region selection: starting from the head,
                  pick the longest window with R^2 >= --r2-thresh and slope
                  within --slope-tol of the maximum short-window slope. This
                  handles the substrate-depletion curvature that appears in
                  later enzyme traces.
6. Pair consecutive (BLANK, ENZYME) segments and compute
        v_net = slope_enzyme - slope_blank   [AU min^-1]
7. Write a semicolon-separated results table and a diagnostic PNG with all
   segments, fitted lines, and classification labels.

CLI
---
    python kinetics_pipeline.py <input.csv> [-o <outdir>] [--min-seg-sec 20]
        [--head-trim-sec 3] [--tail-trim-sec 1] [--r2-thresh 0.995]
        [--slope-tol 0.10] [--force-alternating/--no-force-alternating]

Author: Cornel Niederhauser, 2026
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_cary_csv(path: Path) -> pd.DataFrame:
    """Read the numeric data block of a Cary 60 kinetics CSV export.

    The file is in latin-1 with CRLF line endings, a two-line header
    ("Sample 1,," and "Time (min),Abs,"), numeric rows, then an empty line
    followed by a text metadata footer. We stop reading at the first row
    after the numeric block that cannot be parsed as two floats.
    """
    raw = path.read_bytes().decode("latin-1")
    rows: list[tuple[float, float]] = []
    started = False
    for i, line in enumerate(raw.splitlines()):
        if i < 2:
            continue
        parts = line.strip().split(",")
        if len(parts) < 2 or parts[0] == "":
            if started:
                break
            continue
        try:
            t = float(parts[0])
            a = float(parts[1])
            rows.append((t, a))
            started = True
        except ValueError:
            if started:
                break
            continue
    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    df = pd.DataFrame(rows, columns=["time_min", "abs"])
    return df


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def detect_segments(
    t: np.ndarray,
    a: np.ndarray,
    jump_thresh: float = 0.03,
    neg_spike_thresh: float = -0.05,
    dilate_points: int = 25,
    min_seg_sec: float = 20.0,
) -> list[tuple[int, int]]:
    """Return a list of (start_idx, end_idx_exclusive) for clean segments.

    A point is marked as disturbed if:
      - the absolute point-to-point change in A exceeds `jump_thresh`, OR
      - A drops below `neg_spike_thresh` (cuvette removal artefact).
    Disturbed regions are dilated by `dilate_points` samples on each side
    to remove instrument settle-down after pipetting or reinsertion.
    Segments shorter than `min_seg_sec` are discarded.
    """
    n = len(a)
    if n < 3:
        return []

    da = np.abs(np.diff(a))
    disturbed = np.zeros(n, dtype=bool)
    disturbed[:-1] |= da > jump_thresh
    disturbed[1:]  |= da > jump_thresh
    disturbed      |= a < neg_spike_thresh
    disturbed      = binary_dilation(disturbed, iterations=dilate_points)

    clean = ~disturbed
    d = np.diff(clean.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1
    if clean[0]:
        starts = np.r_[0, starts]
    if clean[-1]:
        ends = np.r_[ends, n]

    # Filter by duration
    dt_min = min_seg_sec / 60.0
    segs = [(int(s), int(e)) for s, e in zip(starts, ends)
            if (t[e - 1] - t[s]) >= dt_min]
    return segs


# ---------------------------------------------------------------------------
# Linear fitting
# ---------------------------------------------------------------------------

def _cumulative_linreg(x: np.ndarray, y: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised: for every prefix x[:k+1] (k = min_idx, ..., n-1) return
    arrays of (slope, intercept, R^2). Uses cumulative sums so the cost
    is O(n), not O(n^2) like a polyfit-per-window sweep.

    The k-th element corresponds to the fit over x[:k+1] / y[:k+1].
    """
    n = len(x)
    if n < 2:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    cs_x  = np.cumsum(x)
    cs_y  = np.cumsum(y)
    cs_xx = np.cumsum(x * x)
    cs_xy = np.cumsum(x * y)
    cs_yy = np.cumsum(y * y)
    n_arr = np.arange(1, n + 1, dtype=float)
    # Avoid division warnings for the trivial first window.
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = n_arr * cs_xx - cs_x * cs_x
        slope = np.where(denom > 0,
                         (n_arr * cs_xy - cs_x * cs_y) / denom,
                         np.nan)
        intercept = np.where(n_arr > 0,
                             (cs_y - slope * cs_x) / n_arr,
                             np.nan)
        ss_tot = cs_yy - (cs_y * cs_y) / n_arr
        # ss_res = sum((y - (slope*x + intercept))^2)
        #        = sum(y^2) - 2*slope*sum(x*y) - 2*intercept*sum(y)
        #          + slope^2*sum(x^2) + 2*slope*intercept*sum(x)
        #          + n*intercept^2
        ss_res = (cs_yy
                  - 2.0 * slope * cs_xy
                  - 2.0 * intercept * cs_y
                  + slope * slope * cs_xx
                  + 2.0 * slope * intercept * cs_x
                  + n_arr * intercept * intercept)
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return slope, intercept, r2


@dataclass
class LinearFit:
    slope: float              # AU per minute
    intercept: float
    r2: float
    n_points: int
    t_start: float            # minutes
    t_end: float              # minutes
    window_sec: float
    meets_r2_threshold: bool = True


def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, R^2) for a simple linear regression."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(m), float(b), r2


def fit_enzyme_linear_robust(
    t: np.ndarray, a: np.ndarray,
    head_trim_sec: float, tail_trim_sec: float,
    min_window_sec: float = 8.0,
    max_slope_dev: float = 0.05,
    curvature_sig_sigma: float = 4.0,
) -> LinearFit:
    """Noise-robust linear-region detection via global quadratic.

    Fits a quadratic over the full trimmed enzyme segment, then
    decides whether the curvature is BOTH:
      (a) statistically resolvable: |a|/SE(a) > curvature_sig_sigma
      (b) practically meaningful: the slope changes by more than
          `max_slope_dev` (default 5%) across the segment

    AND in the saturation direction (slope decreasing over time).

    If any of these checks fail, the full segment is kept (no
    truncation). Otherwise the window is truncated to the point where
    the predicted slope deviates from the initial slope by exactly
    `max_slope_dev`.

    A final OLS line is fit on the chosen window. This is more robust
    on noisy low-concentration traces than the R^2-based window-finder
    because R^2 thresholds become meaningless when the noise dominates
    the signal.
    """
    t0 = t[0] + head_trim_sec / 60.0
    t1 = t[-1] - tail_trim_sec / 60.0
    mask = (t >= t0) & (t <= t1)
    if mask.sum() < 10:
        mask = np.ones_like(t, dtype=bool)
    x, y = t[mask], a[mask]
    n = len(x)
    if n < 20:
        m, b, r2 = _linreg(x, y)
        return LinearFit(
            slope=m, intercept=b, r2=r2, n_points=n,
            t_start=float(x[0]), t_end=float(x[-1]),
            window_sec=float((x[-1] - x[0]) * 60.0),
            meets_r2_threshold=True,
        )

    dt_min = float(np.mean(np.diff(x)))
    # The 5-point floor matches the legacy r2_window method's floor;
    # below this a 2-coefficient linear fit becomes unreliable. The
    # user-specified `min_window_sec` decides the actual window length
    # whenever it implies more than 5 points.
    min_pts = max(int(min_window_sec / 60.0 / dt_min), 5)
    min_pts = min(min_pts, n)

    # Fit a quadratic over the whole window; centre x at start so that
    # the linear coefficient = initial slope.
    xp = x - x[0]
    try:
        coeffs, cov = np.polyfit(xp, y, 2, cov=True)
    except (ValueError, np.linalg.LinAlgError):
        m, b, r2 = _linreg(x, y)
        return LinearFit(
            slope=m, intercept=b, r2=r2, n_points=n,
            t_start=float(x[0]), t_end=float(x[-1]),
            window_sec=float((x[-1] - x[0]) * 60.0),
            meets_r2_threshold=True,
        )
    a_q, b_q, _ = coeffs
    se_a = float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else 0.0
    z_a = abs(a_q) / se_a if se_a > 0 else 0.0

    xp_end = float(xp[-1])
    if abs(b_q) > 1e-12:
        deviation_end = abs(2.0 * a_q * xp_end / b_q)
    else:
        deviation_end = 0.0

    # Direction: the linear coefficient `b_q` is the initial slope of
    # the quadratic at xp=0. If a_q has the opposite sign of b_q, the
    # slope is decreasing over time, which is saturation.
    decreasing = (a_q * b_q < 0)

    if not decreasing or z_a < curvature_sig_sigma or deviation_end < max_slope_dev:
        end_idx = n
    else:
        xp_max = max_slope_dev * abs(b_q) / (2.0 * abs(a_q))
        end_idx = int(np.searchsorted(xp, xp_max))
        end_idx = max(min_pts, min(end_idx, n))

    m, b, r2 = _linreg(x[:end_idx], y[:end_idx])
    return LinearFit(
        slope=m, intercept=b, r2=r2, n_points=int(end_idx),
        t_start=float(x[0]), t_end=float(x[end_idx - 1]),
        window_sec=float((x[end_idx - 1] - x[0]) * 60.0),
        meets_r2_threshold=True,   # this method has no R^2 threshold
    )


def fit_blank(
    t: np.ndarray, a: np.ndarray,
    head_trim_sec: float, tail_trim_sec: float,
) -> LinearFit:
    """Linear fit over the trimmed blank segment (whole window)."""
    t0 = t[0] + head_trim_sec / 60.0
    t1 = t[-1] - tail_trim_sec / 60.0
    mask = (t >= t0) & (t <= t1)
    if mask.sum() < 5:
        mask = np.ones_like(t, dtype=bool)
    x, y = t[mask], a[mask]
    m, b, r2 = _linreg(x, y)
    return LinearFit(
        slope=m, intercept=b, r2=r2, n_points=int(mask.sum()),
        t_start=float(x[0]), t_end=float(x[-1]),
        window_sec=float((x[-1] - x[0]) * 60.0),
    )


def _extend_start_to_minimum(
    t_full: np.ndarray, a_full: np.ndarray, seg_start: int,
    max_lookback_sec: float = 3.0,
    smooth_thresh: float = 0.02,
) -> int:
    """Walk backward from a detected segment start to recover the data
    trimmed by the post-spike dilation buffer, and return the index of
    the absorbance minimum in the recovered region.

    Physical rationale: after a cuvette manipulation, the raw trace
    shows (i) the large spike / ringing, (ii) a settle where A drops
    to its post-manipulation minimum, and (iii) the enzyme-catalysed
    rise. The dilation in `detect_segments` is intentionally generous
    so residual ringing never enters a segment, but on well-behaved
    recoveries this also trims genuine settle+early-rise data. Walking
    backward point-by-point, stopping at the first |dA| above
    `smooth_thresh`, identifies the spike edge; the absorbance minimum
    in the walked region is then the physical onset of the enzymatic
    reaction.

    If `seg_start == 0` or no smoother point is found, the original
    index is returned unchanged.
    """
    if seg_start == 0:
        return 0
    # Local sampling interval
    if len(t_full) < 2:
        return seg_start
    dt_min = float(np.mean(np.diff(t_full[:min(len(t_full), 200)])))
    if dt_min <= 0:
        return seg_start
    max_lookback_pts = max(1, int(max_lookback_sec / 60.0 / dt_min))
    earliest = max(0, seg_start - max_lookback_pts)

    i = seg_start
    while i > earliest:
        if abs(a_full[i] - a_full[i - 1]) > smooth_thresh:
            break
        i -= 1

    region = a_full[i:seg_start + 1]
    if region.size == 0:
        return seg_start
    min_offset = int(np.argmin(region))
    return i + min_offset


def fit_enzyme_initial_rate(
    t: np.ndarray, a: np.ndarray,
    head_trim_sec: float, tail_trim_sec: float,
    r2_thresh: float = 0.995,
    min_window_sec: float = 5.0,
    slope_tol: float = 0.10,
) -> LinearFit:
    """Pick the initial-rate window for a curving enzyme trace.

    Strategy
    --------
    1. Apply the fixed head/tail trims.
    2. Compute a reference initial velocity `slope_ref` from the first
       ~2 s of the trimmed segment. This is the closest we can get to
       the true t=0 velocity.
    3. Compute slope, intercept and R^2 for every prefix window using
       a single O(n) vectorised cumulative-sum pass.
    4. Among windows >= `min_window_sec` satisfying
           R^2 >= r2_thresh  AND  slope >= (1 - slope_tol) * slope_ref,
       return the LONGEST.
    5. If no window meets both, return the highest-R^2 candidate
       (ties broken by longer window) and set
       `meets_r2_threshold = False` for the caller to flag.
    """
    t0 = t[0] + head_trim_sec / 60.0
    t1 = t[-1] - tail_trim_sec / 60.0
    mask = (t >= t0) & (t <= t1)
    if mask.sum() < 10:
        mask = np.ones_like(t, dtype=bool)
    x, y = t[mask], a[mask]

    # Reference initial velocity
    ref_dt_min = min(2.0, min_window_sec) / 60.0
    ref_end = int(np.searchsorted(x, x[0] + ref_dt_min))
    ref_end = max(ref_end, 5)
    slope_ref, _, _ = _linreg(x[:ref_end], y[:ref_end])

    # Minimum window length in points
    min_pts = int(np.searchsorted(x, x[0] + min_window_sec / 60.0))
    min_pts = max(min_pts, 5)
    if min_pts >= len(x):
        m, b, r2 = _linreg(x, y)
        return LinearFit(
            slope=m, intercept=b, r2=r2, n_points=len(x),
            t_start=float(x[0]), t_end=float(x[-1]),
            window_sec=float((x[-1] - x[0]) * 60.0),
            meets_r2_threshold=(r2 >= r2_thresh),
        )

    # Vectorised: slope/intercept/R^2 for every prefix in one O(n) pass
    slopes, intercepts, r2s = _cumulative_linreg(x, y)
    valid_lo = min_pts - 1   # k-th element corresponds to a window of length k+1
    if abs(slope_ref) < 1e-9:
        slope_mask = np.ones_like(slopes, dtype=bool)
    else:
        slope_mask = slopes >= (1.0 - slope_tol) * slope_ref
    r2_mask = r2s >= r2_thresh
    in_range = np.zeros_like(slopes, dtype=bool)
    in_range[valid_lo:] = True
    valid = slope_mask & r2_mask & in_range & np.isfinite(slopes)
    if valid.any():
        i_end = int(np.flatnonzero(valid)[-1])
        meets = True
    else:
        cand = in_range & np.isfinite(r2s)
        if not cand.any():
            m, b, r2 = _linreg(x[:min_pts], y[:min_pts])
            return LinearFit(
                slope=m, intercept=b, r2=r2, n_points=int(min_pts),
                t_start=float(x[0]), t_end=float(x[min_pts - 1]),
                window_sec=float((x[min_pts - 1] - x[0]) * 60.0),
                meets_r2_threshold=(r2 >= r2_thresh),
            )
        idxs = np.flatnonzero(cand)
        i_end = int(idxs[np.argmax(r2s[idxs])])
        meets = False

    return LinearFit(
        slope=float(slopes[i_end]),
        intercept=float(intercepts[i_end]),
        r2=float(r2s[i_end]),
        n_points=i_end + 1,
        t_start=float(x[0]),
        t_end=float(x[i_end]),
        window_sec=float((x[i_end] - x[0]) * 60.0),
        meets_r2_threshold=meets,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_segments(
    segments: list[tuple[int, int]],
    t: np.ndarray, a: np.ndarray,
    force_alternating: bool = True,
) -> list[str]:
    """Label each segment as 'blank' or 'enzyme'.

    `force_alternating=True` assumes strict blank, enzyme, blank,
    enzyme... ordering by position.

    `force_alternating=False` uses **ΔA ratio-gap clustering**.
    For each segment, the absolute change in absorbance |A_end -
    A_start| is computed over the post-spike, post-tail interior.
    Real enzyme reactions accumulate enough product to produce a
    rise of 0.03 AU or more across a 20-60 s segment, even at the
    lowest substrate concentration. Substrate decomposition (the
    blank) gives 0.000-0.015 AU even at the highest substrate
    concentration we use. There is therefore typically a clean 2x
    or larger ratio gap between the two populations.

    Algorithm:
    1. Compute |dA| for each segment's interior.
    2. Sort the |dA| values; ignore the noise-floor cluster (any
       |dA| below 0.005, where photometer noise dominates and
       ratios become meaningless).
    3. Among the remaining values, find the largest consecutive
       ratio. If it is at least 2x, split at the geometric mean.
    4. As a fallback, classify any segment with |dA| > 0.015 AU
       AND slope > 0.025 AU/min as enzyme.
    """
    if force_alternating:
        return ["blank" if i % 2 == 0 else "enzyme"
                for i in range(len(segments))]

    if not segments:
        return []

    if len(t) < 2:
        return ["blank"] * len(segments)
    dt_min = float(np.mean(np.diff(t[:min(len(t), 200)])))
    head_skip = max(1, int(2.0 / 60.0 / max(dt_min, 1e-9)))
    tail_skip = max(1, int(1.0 / 60.0 / max(dt_min, 1e-9)))

    dAs = np.zeros(len(segments))
    slopes = np.zeros(len(segments))
    for k, (s, e) in enumerate(segments):
        s_in = s + head_skip
        e_in = e - tail_skip
        if e_in - s_in < 5:
            continue
        dAs[k] = abs(float(a[e_in - 1] - a[s_in]))
        m, _, _ = _linreg(t[s_in:e_in], a[s_in:e_in])
        slopes[k] = abs(m)

    # Find the largest ratio gap in dA values that are above the
    # noise floor (0.005 AU). This avoids being misled by ratios
    # between very small numbers in the noise floor cluster.
    NOISE_FLOOR = 0.005
    GAP_FACTOR = 2.0
    DA_FALLBACK = 0.015     # AU; if no clear gap found
    SLOPE_FALLBACK = 0.025  # AU/min

    above_noise = np.sort(dAs[dAs >= NOISE_FLOOR])
    threshold_dA = DA_FALLBACK
    if len(above_noise) >= 2:
        ratios = above_noise[1:] / above_noise[:-1]
        kg = int(np.argmax(ratios))
        if ratios[kg] >= GAP_FACTOR:
            threshold_dA = float(np.sqrt(above_noise[kg] * above_noise[kg + 1]))

    return [
        "enzyme" if (dAs[k] > threshold_dA and slopes[k] > SLOPE_FALLBACK)
        else "blank"
        for k in range(len(segments))
    ]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse_file(
    input_path: Path,
    outdir: Path,
    min_seg_sec: float,
    head_trim_sec: float,
    tail_trim_sec: float,
    r2_thresh: float,
    slope_tol: float,
    force_alternating: bool,
    extend_to_minimum: bool = True,
    extend_lookback_sec: float = 3.0,
    extend_smooth_thresh: float = 0.02,
) -> pd.DataFrame:
    df = load_cary_csv(input_path)
    t = df["time_min"].to_numpy()
    a = df["abs"].to_numpy()

    segments = detect_segments(t, a, min_seg_sec=min_seg_sec)
    if not segments:
        raise RuntimeError("No segments detected. Check input file / thresholds.")

    labels = classify_segments(segments, t, a, force_alternating=force_alternating)

    # Segment-start refinement: walk each enzyme segment backward to the
    # post-spike absorbance minimum. This recovers the data the dilation
    # buffer trimmed off and anchors the fit at the physical onset.
    if extend_to_minimum:
        refined = []
        for (s, e), lab in zip(segments, labels):
            if lab == "enzyme":
                new_s = _extend_start_to_minimum(
                    t, a, s,
                    max_lookback_sec=extend_lookback_sec,
                    smooth_thresh=extend_smooth_thresh,
                )
                refined.append((new_s, e))
            else:
                refined.append((s, e))
        segments = refined

    # Fit every segment
    fits: list[LinearFit] = []
    for (s, e), lab in zip(segments, labels):
        ts, ys = t[s:e], a[s:e]
        if lab == "blank":
            fits.append(fit_blank(ts, ys, head_trim_sec, tail_trim_sec))
        else:
            fits.append(fit_enzyme_initial_rate(
                ts, ys, head_trim_sec, tail_trim_sec,
                r2_thresh=r2_thresh, slope_tol=slope_tol,
            ))

    # Pair up and build results table
    rows = []
    pair_no = 0
    i = 0
    while i < len(segments):
        lab_i = labels[i]
        if lab_i == "blank" and i + 1 < len(segments) and labels[i + 1] == "enzyme":
            pair_no += 1
            fb, fe = fits[i], fits[i + 1]
            rows.append({
                "pair":            pair_no,
                "seg_blank":       i + 1,
                "seg_enzyme":      i + 2,
                "t_blank_start":   fb.t_start,
                "t_blank_end":     fb.t_end,
                "slope_blank":     fb.slope,
                "r2_blank":        fb.r2,
                "window_blank_s":  fb.window_sec,
                "t_enzyme_start":  fe.t_start,
                "t_enzyme_end":    fe.t_end,
                "slope_enzyme":    fe.slope,
                "r2_enzyme":       fe.r2,
                "window_enzyme_s": fe.window_sec,
                "slope_net":       fe.slope - fb.slope,
            })
            i += 2
        else:
            # Unpaired / unexpected order -- log and skip one
            rows.append({
                "pair":            None,
                "seg_blank":       (i + 1) if lab_i == "blank" else None,
                "seg_enzyme":      (i + 1) if lab_i == "enzyme" else None,
                "t_blank_start":   fits[i].t_start if lab_i == "blank" else None,
                "t_blank_end":     fits[i].t_end   if lab_i == "blank" else None,
                "slope_blank":     fits[i].slope   if lab_i == "blank" else None,
                "r2_blank":        fits[i].r2      if lab_i == "blank" else None,
                "window_blank_s":  fits[i].window_sec if lab_i == "blank" else None,
                "t_enzyme_start":  fits[i].t_start if lab_i == "enzyme" else None,
                "t_enzyme_end":    fits[i].t_end   if lab_i == "enzyme" else None,
                "slope_enzyme":    fits[i].slope   if lab_i == "enzyme" else None,
                "r2_enzyme":       fits[i].r2      if lab_i == "enzyme" else None,
                "window_enzyme_s": fits[i].window_sec if lab_i == "enzyme" else None,
                "slope_net":       None,
            })
            i += 1

    results = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{input_path.stem}_kinetics.csv"
    results.to_csv(out_csv, sep=";", index=False, float_format="%.6g")

    # Diagnostic plots: overview + per-pair zoom
    _plot_diagnostic(t, a, segments, labels, fits, outdir / f"{input_path.stem}_kinetics.png")
    _plot_pairs_zoom(t, a, segments, labels, fits, outdir / f"{input_path.stem}_pairs_zoom.png")

    return results


def _plot_diagnostic(t, a, segments, labels, fits, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(t, a, color="lightgrey", linewidth=0.6, label="raw trace")
    pair = 0
    for (s, e), lab, fit in zip(segments, labels, fits):
        color = "tab:blue" if lab == "blank" else "tab:red"
        ax.plot(t[s:e], a[s:e], color=color, linewidth=0.9, alpha=0.6)
        xs = np.array([fit.t_start, fit.t_end])
        ys = fit.slope * xs + fit.intercept
        ax.plot(xs, ys, color="black", linewidth=2.0)
        if lab == "enzyme":
            pair += 1
            ax.annotate(f"#{pair}", xy=(xs.mean(), ys.mean()),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color="black")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Absorbance at 380 nm")
    ax.set_title(f"{out_png.stem}  (blue = blank, red = enzyme, black = fit)")
    ax.set_ylim(-0.05, min(0.7, max(0.1, a.max() * 1.05)))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_pairs_zoom(t, a, segments, labels, fits, out_png: Path) -> None:
    """Per-pair zoomed subplots. Each enzyme segment is shown alone with
    its fit overlaid and the fit window shaded, so the user can verify
    where the linear region starts and ends for every pair."""
    # Collect pairs
    pair_data = []
    i = 0
    while i < len(segments) - 1:
        if labels[i] == "blank" and labels[i + 1] == "enzyme":
            pair_data.append((i, i + 1))
            i += 2
        else:
            i += 1
    n_pairs = len(pair_data)
    if n_pairs == 0:
        return

    ncols = 5
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.0 * ncols, 2.4 * nrows),
                             squeeze=False)
    for pair_no, (ib, ie) in enumerate(pair_data, 1):
        ax = axes[(pair_no - 1) // ncols][(pair_no - 1) % ncols]
        s_e, e_e = segments[ie]
        fe = fits[ie]
        fb = fits[ib]

        # Plot the full enzyme segment
        ax.plot(t[s_e:e_e], a[s_e:e_e], color="tab:red",
                linewidth=0.9, alpha=0.8, label="enzyme trace")

        # Shade the fit window
        ax.axvspan(fe.t_start, fe.t_end, color="black", alpha=0.08)

        # Plot the fit line, slightly extended for visibility
        pad = 0.02 * (fe.t_end - fe.t_start)
        xs = np.array([fe.t_start - pad, fe.t_end + pad])
        ys = fe.slope * xs + fe.intercept
        ax.plot(xs, ys, color="black", linewidth=1.8, label="fit")

        # Mark the onset (start of fit window)
        ax.axvline(fe.t_start, color="black", linewidth=0.8,
                   linestyle=":", alpha=0.6)

        ax.set_title(f"#{pair_no}  v_E={fe.slope:.3f}  v_B={fb.slope:+.4f}  "
                     f"R²={fe.r2:.3f}  w={fe.window_sec:.1f}s",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for k in range(n_pairs, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle(f"{out_png.stem}  —  per-pair fits "
                 f"(red = raw enzyme trace, black = fit, "
                 f"shaded = fit window)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automatic enzyme-kinetic initial-rate extraction from "
                    "a Cary 60 continuous UV-Vis trace (CSV export).",
    )
    p.add_argument("input", type=Path, help="Cary CSV file")
    p.add_argument("-o", "--outdir", type=Path, default=Path("."))
    p.add_argument("--min-seg-sec",    type=float, default=20.0)
    p.add_argument("--head-trim-sec",  type=float, default=0.0,
                   help="Extra trim after the dilation buffer. Default 0: "
                        "start the fit as soon as the rise begins.")
    p.add_argument("--tail-trim-sec",  type=float, default=1.0)
    p.add_argument("--r2-thresh",      type=float, default=0.9999)
    p.add_argument("--slope-tol",      type=float, default=0.10)
    p.add_argument("--force-alternating",    dest="force_alt", action="store_true", default=True)
    p.add_argument("--no-force-alternating", dest="force_alt", action="store_false")
    p.add_argument("--extend-to-minimum",    dest="extend", action="store_true",  default=True,
                   help="Extend each enzyme segment's start backward to the "
                        "post-spike absorbance minimum, so the fit begins at "
                        "the physical onset of the rise (default: on).")
    p.add_argument("--no-extend-to-minimum", dest="extend", action="store_false")
    p.add_argument("--extend-lookback-sec", type=float, default=3.0,
                   help="Max seconds to walk backward from the dilation-buffered "
                        "segment start when searching for the absorbance minimum.")
    p.add_argument("--extend-smooth-thresh", type=float, default=0.02,
                   help="Point-to-point |dA| threshold that marks the spike edge "
                        "during the backward walk. Should be smaller than the "
                        "`detect_segments` jump threshold (0.03).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = analyse_file(
        input_path=args.input,
        outdir=args.outdir,
        min_seg_sec=args.min_seg_sec,
        head_trim_sec=args.head_trim_sec,
        tail_trim_sec=args.tail_trim_sec,
        r2_thresh=args.r2_thresh,
        slope_tol=args.slope_tol,
        force_alternating=args.force_alt,
        extend_to_minimum=args.extend,
        extend_lookback_sec=args.extend_lookback_sec,
        extend_smooth_thresh=args.extend_smooth_thresh,
    )
    paired = results.dropna(subset=["slope_net"])
    print(f"Input: {args.input}")
    print(f"Pairs detected: {len(paired)} / rows in table: {len(results)}")
    print()
    cols = ["pair", "t_blank_start", "slope_blank", "r2_blank",
            "t_enzyme_start", "slope_enzyme", "r2_enzyme",
            "window_enzyme_s", "slope_net"]
    with pd.option_context("display.float_format", "{:.5f}".format,
                           "display.max_columns", None, "display.width", 200):
        print(paired[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())