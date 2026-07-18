"""Tests for V_DIFF (v1) — DBSCAN-based bad-change detector."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.bcd import V_DIFF


RNG = np.random.default_rng(0)

# Unix epoch base so timestamps are plausible
_T0 = 1_700_000_000


def _old(n_windows=20, base=50.0, noise=3.0, win=10):
    """Stable old-build series: n_windows * win points."""
    n = n_windows * win
    vals = base + RNG.normal(0, noise, n)
    return pd.Series(vals, index=np.arange(_T0, _T0 + n * 60, 60))


def _new(n=10, base=50.0, noise=3.0):
    """New-build series of n points."""
    t_start = _T0 + 20 * 10 * 60
    vals = base + RNG.normal(0, noise, n)
    return pd.Series(vals, index=np.arange(t_start, t_start + n * 60, 60))


# ---------------------------------------------------------------------------
# Precondition gates
# ---------------------------------------------------------------------------

def test_win_below_60_returns_false():
    d = V_DIFF(win=30)
    assert d.detect(_old(), _new()) is False


def test_none_old_returns_false():
    d = V_DIFF(win=600)
    assert d.detect(None, _new(n=20)) is False


def test_too_few_old_points_returns_false():
    d = V_DIFF(win=600)
    short_old = pd.Series([50.0] * 5, index=np.arange(_T0, _T0 + 5 * 60, 60))
    assert d.detect(short_old, _new(n=20)) is False


def test_none_new_returns_false():
    d = V_DIFF(win=600)
    assert d.detect(_old(), None) is False


def test_too_few_new_points_returns_false():
    d = V_DIFF(win=600)
    short_new = pd.Series([200.0] * 3, index=np.arange(_T0, _T0 + 3 * 60, 60))
    assert d.detect(_old(), short_new) is False


def test_new_build_mean_below_avg_min_returns_false():
    d = V_DIFF(avgMin=100.0, win=600)
    new_ts = _new(n=10, base=5.0)   # mean well below 100
    assert d.detect(_old(), new_ts) is False


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def test_clear_regression_detected():
    """New-build mean far above old-build — should be detected."""
    d = V_DIFF(eps=0.3, minPts=2, avgMin=20.0, giniMax=0.5, win=600)
    baseline_ts = _old(n_windows=30, base=50.0, noise=3.0, win=10)
    new_ts = _new(n=10, base=300.0, noise=5.0)
    assert d.detect(baseline_ts, new_ts) is True


def test_healthy_deploy_not_detected():
    """New build at same level as old build — no anomaly."""
    d = V_DIFF(eps=0.3, minPts=2, avgMin=20.0, win=600)
    baseline_ts = _old(n_windows=30, base=50.0, noise=3.0, win=10)
    new_ts = _new(n=10, base=52.0, noise=3.0)
    assert d.detect(baseline_ts, new_ts) is False


# ---------------------------------------------------------------------------
# Post-checks (slope and GINI suppression)
# ---------------------------------------------------------------------------

def test_slope_suppression():
    """A sharply falling new-build signal should be suppressed."""
    d = V_DIFF(eps=0.3, minPts=2, avgMin=20.0, slopeMin=-5.0, giniMax=0.9, win=600)
    baseline_ts = _old(n_windows=30, base=50.0, noise=2.0, win=10)
    # New build starts high but falls steeply — slope should be very negative
    t_start = _T0 + 30 * 10 * 60
    vals = np.linspace(500, 30, 10)
    new_ts = pd.Series(vals, index=np.arange(t_start, t_start + 10 * 60, 60))
    assert d.detect(baseline_ts, new_ts) is False


def test_gini_suppression():
    """A spiky (high-GINI) new-build window should be suppressed."""
    d = V_DIFF(eps=0.3, minPts=2, avgMin=20.0, giniMax=0.3, win=600)
    baseline_ts = _old(n_windows=30, base=50.0, noise=2.0, win=10)
    # Most values are zero with one massive spike — GINI will be high
    t_start = _T0 + 30 * 10 * 60
    vals = np.array([0.0] * 9 + [500.0])
    new_ts = pd.Series(vals, index=np.arange(t_start, t_start + 10 * 60, 60))
    assert d.detect(baseline_ts, new_ts) is False
