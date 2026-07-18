"""Shared utilities for BCD detectors."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


class GINI:
    """
    GINI coefficient utilities (0 = perfectly equal, 1 = perfectly unequal).

    calc()   — pairwise difference method, O(n²), includes zeros. Used by V_DIFF.
    calc_v2() — Lorenz curve method, O(n log n), configurable zero handling. Used by V_DIFF_V2.
    """

    @staticmethod
    def calc(x: ArrayLike) -> float:
        return 0.5 * np.abs(np.subtract.outer(x, x)).mean() / np.mean(x)

    @staticmethod
    def calc_v2(values: ArrayLike, include_zeros: bool = False) -> float:
        """
        Args:
            include_zeros: If True, zeros are included. Use True for spikiness detection
                on sparse data where [0, 0, 0, 100] should show high inequality.
        """
        v = np.array(values, dtype=float)
        if not include_zeros:
            v = v[v > 0]
        if len(v) < 2:
            return 0.0
        v = np.sort(v)
        n = len(v)
        v_sum = np.sum(v)
        if v_sum == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * v) - (n + 1) * v_sum) / (n * v_sum)


class SLOPE:
    @staticmethod
    def calc(x: ArrayLike) -> float:
        """Fit a line to x and return the slope (least squares)."""
        idx = list(range(len(x)))
        coeffs = np.polyfit(idx, x, 1)
        return coeffs[-2]


class SpikeBaseline:
    """
    Historical spike baseline calculation.

    Used by V_DIFF_V2 to determine whether a current spike is within the
    normal historical range for a given metric.
    """

    METHODS = ['max', 'p90', 'p95', 'p99', 'top3_median', 'top5_median', 'spike_p90']

    @staticmethod
    def calculate(method: str, data: np.ndarray) -> float:
        """
        Args:
            method: One of 'max', 'p90', 'p95', 'p99', 'top3_median',
                    'top5_median' (recommended), 'spike_p90'.
            data: Historical baseline values (numpy array).

        Returns:
            Baseline spike value as a float.
        """
        def _top_n_median(d: np.ndarray, n: int) -> float:
            top_n = np.sort(d)[-n:] if len(d) >= n else d
            return np.median(top_n)

        def _spike_p90(d: np.ndarray) -> float:
            spike_threshold = np.percentile(d, 95)
            spike_values = d[d >= spike_threshold]
            if len(spike_values) >= 5:
                return np.percentile(spike_values, 90)
            elif len(spike_values) > 0:
                return np.median(spike_values)
            return np.percentile(d, 99)

        dispatch = {
            'max':         lambda d: np.max(d),
            'p90':         lambda d: np.percentile(d, 90),
            'p95':         lambda d: np.percentile(d, 95),
            'p99':         lambda d: np.percentile(d, 99),
            'top3_median': lambda d: _top_n_median(d, 3),
            'top5_median': lambda d: _top_n_median(d, 5),
            'spike_p90':   lambda d: _spike_p90(d),
        }

        if method not in dispatch:
            raise ValueError(
                f"Unknown method: '{method}'. Valid options: {SpikeBaseline.METHODS}"
            )
        return dispatch[method](data)


def denoise_mean(x: pd.Series, n: float = 0.3) -> pd.Series:
    """Clip values to within n * mean_absolute_deviation of the mean."""
    mean = np.mean(x)
    diff_mean = np.mean(np.abs(x - mean))
    return np.clip(x, mean - n * diff_mean, mean + n * diff_mean)


def denoise_rolling_median(x: pd.Series, win: int) -> pd.Series:
    """Smooth a series with a rolling median, forward-filling the warmup window."""
    denoised = x.rolling(win).median()
    return denoised.fillna(denoised.iloc[win])


def de_autofill(x: pd.Series | None) -> None:
    """
    Remove consecutive duplicate values produced by monitoring system auto-fill.

    Example: [0, 0, 65, 65, 65, 65, 66] → [0, 0, 65, 0, 0, 0, 66]
    """
    if x is None:
        return
    prev = 0
    for i in range(len(x.values)):
        v = x.values[i]
        if v > 0:
            if v == prev:
                x.values[i] = 0
            else:
                prev = v
