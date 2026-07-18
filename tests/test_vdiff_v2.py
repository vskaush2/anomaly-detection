"""Tests for V_DIFF_V2 — gate-based bad-change detector."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.bcd import V_DIFF_V2


RNG = np.random.default_rng(1)

_T0 = 1_700_000_000          # base Unix timestamp
_T_NEW = _T0 + 7 * 24 * 3600 # new-build starts 7 days later


def _old(n=10_080, base=50.0, noise=4.0):
    """7-day minute-level stable baseline_ts."""
    vals = base + RNG.normal(0, noise, n)
    return pd.Series(vals, index=np.arange(_T0, _T0 + n * 60, 60))


def _new(n=30, base=50.0, noise=3.0):
    """New-build window of n points."""
    vals = base + RNG.normal(0, noise, n)
    return pd.Series(vals, index=np.arange(_T_NEW, _T_NEW + n * 60, 60))


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------

def test_none_new_returns_false():
    assert V_DIFF_V2().detect(_old(), None) is False


def test_empty_new_returns_false():
    assert V_DIFF_V2().detect(_old(), pd.Series([], dtype=float)) is False


# ---------------------------------------------------------------------------
# Gate 1: Absolute floor
# ---------------------------------------------------------------------------

def test_floor_gate_blocks_low_new_build():
    """New-build values below avg_min should not trigger an alert."""
    d = V_DIFF_V2(avg_min=100.0)
    new_ts = _new(n=30, base=5.0)   # mean << 100
    assert d.detect(_old(), new_ts) is False


def test_floor_gate_passes_when_above_threshold():
    """Values well above avg_min should pass through to the next gate."""
    d = V_DIFF_V2(avg_min=10.0, threshold=0.1, adaptive_pct=False,
                  suppress_similar_spikes=False, correlated_check=False)
    baseline_ts = _old(base=50.0, noise=3.0)
    new_ts = _new(base=800.0, noise=5.0)
    assert d.detect(baseline_ts, new_ts) is True


# ---------------------------------------------------------------------------
# Gate 2: Percentile rank
# ---------------------------------------------------------------------------

def test_percentile_gate_blocks_moderate_elevation():
    """A modest increase that doesn't exceed p99 of baseline_ts should be suppressed."""
    d = V_DIFF_V2(avg_min=10.0, adaptive_pct=False, pct_gate_low_cv=0.99,
                  suppress_similar_spikes=False, correlated_check=False)
    # Old build has occasional spikes so p99 is high
    rng_local = np.random.default_rng(42)
    vals = np.full(10_080, 50.0)
    spike_idx = rng_local.choice(10_080, size=200, replace=False)
    vals[spike_idx] = 900.0
    baseline_ts = pd.Series(vals, index=np.arange(_T0, _T0 + 10_080 * 60, 60))
    # New build is elevated but still below the p99 spike level
    new_ts = _new(base=80.0, noise=3.0)
    assert d.detect(baseline_ts, new_ts) is False


# ---------------------------------------------------------------------------
# Gate 3: Spike suppression
# ---------------------------------------------------------------------------

def test_spike_suppression_on_spiky_baseline_ts():
    """When the baseline_ts is spiky and new spike is within historical range, suppress."""
    rng_local = np.random.default_rng(7)
    vals = np.full(10_080, 50.0)
    spike_idx = rng_local.choice(10_080, size=int(10_080 * 0.08), replace=False)
    vals[spike_idx] = 300.0
    baseline_ts = pd.Series(vals, index=np.arange(_T0, _T0 + 10_080 * 60, 60))
    # New spike is within 1.2x of the historical spike baseline_ts
    new_ts = _new(base=320.0, noise=5.0)
    d = V_DIFF_V2(avg_min=10.0, adaptive_pct=False, pct_gate_low_cv=0.0,
                  correlated_check=False, spike_range_multiplier=1.5,
                  baseline_spikiness_gini_threshold=0.1)
    assert d.detect(baseline_ts, new_ts) is False


def test_spike_suppression_skipped_when_disabled():
    """With suppress_similar_spikes=False the spike gate is bypassed."""
    rng_local = np.random.default_rng(8)
    vals = np.full(10_080, 50.0)
    spike_idx = rng_local.choice(10_080, size=int(10_080 * 0.08), replace=False)
    vals[spike_idx] = 300.0
    baseline_ts = pd.Series(vals, index=np.arange(_T0, _T0 + 10_080 * 60, 60))
    new_ts = _new(base=2000.0, noise=10.0)
    d = V_DIFF_V2(avg_min=10.0, adaptive_pct=False, pct_gate_low_cv=0.0,
                  correlated_check=False, suppress_similar_spikes=False,
                  threshold=5.0)
    assert d.detect(baseline_ts, new_ts) is True


# ---------------------------------------------------------------------------
# Gate 4: Correlated check
# ---------------------------------------------------------------------------

def test_correlated_check_suppresses_service_wide_event():
    """When both old and new builds are elevated, the correlated gate suppresses."""
    baseline_ts_ts_base = _old(base=50.0, noise=3.0)
    new_ts = _new(base=150.0, noise=5.0)

    # Old build is also elevated in the connew_ts window
    connew_ts_vals = np.full(30, 130.0)
    connew_ts_idx = np.arange(_T_NEW - 4 * 60, _T_NEW - 4 * 60 + 30 * 60, 60)
    connew_ts_old = pd.Series(connew_ts_vals, index=connew_ts_idx)
    baseline_ts = pd.concat([baseline_ts_ts_base, connew_ts_old]).sort_index()

    d = V_DIFF_V2(avg_min=10.0, adaptive_pct=False, pct_gate_low_cv=0.0,
                  suppress_similar_spikes=False, correlated_check=True,
                  correlation_ratio_threshold=2.0)
    assert d.detect(baseline_ts, new_ts) is False


def test_correlated_check_disabled_allows_detection():
    """With correlated_check=False, service-wide elevation is not suppressed."""
    baseline_ts_ts_base = _old(base=50.0, noise=3.0)
    new_ts = _new(base=2000.0, noise=10.0)
    connew_ts_vals = np.full(30, 130.0)
    connew_ts_idx = np.arange(_T_NEW - 4 * 60, _T_NEW - 4 * 60 + 30 * 60, 60)
    connew_ts_old = pd.Series(connew_ts_vals, index=connew_ts_idx)
    baseline_ts = pd.concat([baseline_ts_ts_base, connew_ts_old]).sort_index()

    d = V_DIFF_V2(avg_min=10.0, adaptive_pct=False, pct_gate_low_cv=0.0,
                  suppress_similar_spikes=False, correlated_check=False,
                  threshold=5.0)
    assert d.detect(baseline_ts, new_ts) is True


# ---------------------------------------------------------------------------
# Gate 5: Robust z-score
# ---------------------------------------------------------------------------

def test_z_score_gate_blocks_borderline_elevation():
    """A spike that clears earlier gates but has a low z-score should be blocked."""
    d = V_DIFF_V2(avg_min=10.0, threshold=100.0, adaptive_pct=False,
                  pct_gate_low_cv=0.0, suppress_similar_spikes=False,
                  correlated_check=False)
    baseline_ts = _old(base=50.0, noise=4.0)
    new_ts = _new(base=55.0, noise=2.0)   # only slightly above baseline_ts
    assert d.detect(baseline_ts, new_ts) is False


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_clear_regression_detected():
    """Strong sustained spike clears all gates and is detected."""
    d = V_DIFF_V2(threshold=10.0, avg_min=10.0)
    baseline_ts = _old(base=50.0, noise=3.0)
    new_ts = _new(base=800.0, noise=10.0)
    assert d.detect(baseline_ts, new_ts) is True


def test_healthy_deploy_not_detected():
    """Stable new build at the same level as old build — no alert."""
    d = V_DIFF_V2(threshold=10.0, avg_min=10.0)
    baseline_ts = _old(base=50.0, noise=3.0)
    new_ts = _new(base=51.0, noise=3.0)
    assert d.detect(baseline_ts, new_ts) is False


def test_no_baseline_ts_data():
    """With no old-build data the detector still runs without error."""
    d = V_DIFF_V2(avg_min=10.0)
    new_ts = _new(base=50.0, noise=3.0)
    result = d.detect(None, new_ts)
    assert isinstance(result, bool)
