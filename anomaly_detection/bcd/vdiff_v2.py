"""V_DIFF_V2: Statistical bad-change detector using old-build baseline comparison.

Detects metric regressions by running a sequence of statistical gates against
a 7-day old-build baseline. Designed to reduce false positives on services
that are naturally spiky or highly variable.

Gates (in order)
----------------
1. Absolute floor      — new build must sustain values above a minimum threshold.
2. Percentile rank     — current value must exceed a high percentile of the baseline
                         (adaptive: p99.9 for high-variance baselines, p99 otherwise).
3. Spike suppression   — suppress if the service regularly has spikes of similar
                         magnitude (based on baseline GINI and historical spike level).
4. Correlated check    — suppress if the old build is also elevated concurrently
                         (indicates a service-wide event, not a build regression).
5. Z-score gate        — robust z-score using MAD-based standard deviation.
"""

import logging

import numpy as np
import pandas as pd

from anomaly_detection.bcd._utils import GINI, SpikeBaseline

logger = logging.getLogger('vdiff_v2')


class V_DIFF_V2:
    def __init__(
        self,
        threshold: float = 10.0,
        avg_min: float = 10.0,
        correlated_check: bool = True,
        adaptive_pct: bool = True,
        suppress_similar_spikes: bool = True,
        spike_range_multiplier: float = 1.2,
        baseline_spike_calculation_method: str = 'top5_median',
        baseline_spikiness_gini_threshold: float = 0.4,
        baseline_variance_cv_threshold: float = 0.45,
        correlation_ratio_threshold: float = 2.0,
        concurrent_spike_min: float = 50.0,
        min_baseline_coverage: float = 0.10,
        floor_window_high: int = 5,
        floor_window_low: int = 3,
        pct_gate_high_cv: float = 0.999,
        pct_gate_low_cv: float = 0.99,
        cv_high_threshold: float = 0.8,
        min_gini_observations: int = 10,
    ) -> None:
        """
        Core parameters
        ---------------
        threshold : float
            Robust z-score threshold for anomaly detection. Default 10.0.
        avg_min : float
            Floor threshold. New build must sustain values at or above this to alert.
            Also used as the boundary for the adaptive floor window: if the old build's
            p99 >= avg_min the longer floor window is used. Default 10.0.
        correlated_check : bool
            Suppress when the old build is also elevated (service-wide event). Default True.
        adaptive_pct : bool
            Use stricter percentile gate (p99.9) for high-variance baselines. Default True.
        suppress_similar_spikes : bool
            Suppress when the new spike is within the service's normal spike range. Default True.
        spike_range_multiplier : float
            Tolerance above historical spike baseline (1.2 = up to 20% above). Default 1.2.
        baseline_spike_calculation_method : str
            How to compute the historical spike level. Options: 'max', 'p90', 'p95', 'p99',
            'top3_median', 'top5_median' (recommended), 'spike_p90'. Default 'top5_median'.

        Tuning parameters
        -----------------
        baseline_spikiness_gini_threshold : float
            GINI threshold above which a baseline is classified as spiky. Default 0.4.
        baseline_variance_cv_threshold : float
            CV threshold above which a baseline is classified as high-variance. Default 0.45.
        correlation_ratio_threshold : float
            new_median / old_median ratio below which both builds are considered
            concurrently elevated. Default 2.0.
        concurrent_spike_min : float
            Both builds must exceed this value for the concurrent spike check. Default 50.0.
        min_baseline_coverage : float
            Minimum fraction of non-NaN baseline points required. Below this, spike
            suppression is skipped and the longer floor window is used. Default 0.10.
        floor_window_high : int
            Floor window (minutes) for historically elevated or sparse baselines. Default 5.
        floor_window_low : int
            Floor window (minutes) for clean baselines. Default 3.
        pct_gate_high_cv : float
            Percentile gate for high-variance baselines. Default 0.999 (p99.9).
        pct_gate_low_cv : float
            Percentile gate for normal baselines. Default 0.99 (p99).
        cv_high_threshold : float
            CV threshold above which the stricter percentile gate is applied. Default 0.8.
        min_gini_observations : int
            Minimum baseline observations needed to compute GINI for spike suppression.
            Default 10.
        """
        self.threshold = threshold
        self.avg_min = avg_min
        self.correlated_check = correlated_check
        self.adaptive_pct = adaptive_pct
        self.suppress_similar_spikes = suppress_similar_spikes
        self.spike_range_multiplier = spike_range_multiplier
        self.baseline_spike_calculation_method = baseline_spike_calculation_method
        self.baseline_spikiness_gini_threshold = baseline_spikiness_gini_threshold
        self.baseline_variance_cv_threshold = baseline_variance_cv_threshold
        self.correlation_ratio_threshold = correlation_ratio_threshold
        self.concurrent_spike_min = concurrent_spike_min
        self.min_baseline_coverage = min_baseline_coverage
        self.floor_window_high = floor_window_high
        self.floor_window_low = floor_window_low
        self.pct_gate_high_cv = pct_gate_high_cv
        self.pct_gate_low_cv = pct_gate_low_cv
        self.cv_high_threshold = cv_high_threshold
        self.min_gini_observations = min_gini_observations

    def detect(self, baseline_ts: pd.Series, new_ts: pd.Series) -> bool:
        """
        Parameters
        ----------
        baseline_ts : pd.Series
            Old-build metric time series (typically 7-day history). Index should be
            Unix timestamps in seconds for correlated-check accuracy.
        new_ts : pd.Series
            New-build metric time series (recent post-deploy window). Same index format.

        Returns
        -------
        bool
            True if a regression is detected.
        """
        if baseline_ts is not None and not isinstance(baseline_ts, pd.Series):
            raise TypeError(f"baseline_ts must be a pd.Series, got {type(baseline_ts).__name__}")
        if new_ts is not None and not isinstance(new_ts, pd.Series):
            raise TypeError(f"new_ts must be a pd.Series, got {type(new_ts).__name__}")

        if new_ts is None or len(new_ts) < 1:
            return False

        new_ts = new_ts.fillna(0).astype(float)
        current_val = new_ts.iloc[-1]

        logger.info(
            'detect() start: new_ts_len=%d, baseline_ts_len=%s, current_val=%.1f',
            len(new_ts), len(baseline_ts) if baseline_ts is not None else 0, current_val,
        )

        # Baseline statistics (with safe defaults when baseline is missing/empty)
        baseline_median, baseline_std, pct_rank, old_p99 = 0.0, 1.0, 1.0, 0.0
        all_baseline = np.array([0.0])
        has_sufficient_coverage = False
        cv = 0.0

        if baseline_ts is not None and len(baseline_ts) > 0:
            has_sufficient_coverage = (~baseline_ts.isna()).sum() / len(baseline_ts) >= self.min_baseline_coverage
            baseline_ts = baseline_ts.fillna(0).astype(float)
            all_baseline = baseline_ts.values
            baseline_median = np.median(all_baseline)
            mad = np.median(np.abs(all_baseline - baseline_median))
            # Scale MAD by 1.4826 to approximate a normal-distribution std
            baseline_std = mad * 1.4826 if mad > 0 else max(1.0, baseline_median * 0.1)
            pct_rank = np.searchsorted(np.sort(all_baseline), current_val, side='right') / len(all_baseline)
            old_p99 = np.percentile(all_baseline, 99)
            cv = baseline_std / baseline_median if baseline_median > 0 else 0.0

        logger.info(
            'Baseline: median=%.1f, std=%.1f, cv=%.2f, p99=%.1f, pct_rank=%.3f, coverage_ok=%s',
            baseline_median, baseline_std, cv, old_p99, pct_rank, has_sufficient_coverage,
        )

        # Gate 1: Absolute floor
        high_confidence = old_p99 >= self.avg_min
        no_baseline = not has_sufficient_coverage
        floor_window = self.floor_window_high if (high_confidence or no_baseline) else self.floor_window_low
        recent_window = min(floor_window, len(new_ts))
        floor_values = new_ts.iloc[-recent_window:].values
        recent_median = np.median(floor_values)
        floor_passed = recent_median >= self.avg_min and current_val >= self.avg_min

        if not floor_passed:
            reason = (
                f'recent_median({recent_median:.1f}) < avg_min({self.avg_min:.1f})'
                if recent_median < self.avg_min
                else f'current_val({current_val:.1f}) < avg_min({self.avg_min:.1f})'
            )
            logger.info('Floor gate FAIL: %s', reason)
            return False

        # Gate 2: Percentile rank
        pct_gate = (
            (self.pct_gate_high_cv if cv > self.cv_high_threshold else self.pct_gate_low_cv)
            if self.adaptive_pct
            else self.pct_gate_low_cv
        )
        if pct_rank < pct_gate:
            logger.info('Percentile gate FAIL: pct_rank(%.3f) < gate(%.3f)', pct_rank, pct_gate)
            return False

        # Gate 3: Spike suppression
        if self.suppress_similar_spikes and len(all_baseline) >= self.min_gini_observations and has_sufficient_coverage:
            baseline_gini = GINI.calc_v2(all_baseline, include_zeros=True)
            baseline_is_spiky = (
                baseline_gini > self.baseline_spikiness_gini_threshold
                or cv > self.baseline_variance_cv_threshold
            )
            historical_spike_baseline = SpikeBaseline.calculate(
                self.baseline_spike_calculation_method, all_baseline
            )
            spike_ceiling = historical_spike_baseline * self.spike_range_multiplier
            within_historical_range = current_val <= spike_ceiling

            if baseline_is_spiky and within_historical_range:
                reason = (
                    f'gini({baseline_gini:.3f}) > threshold({self.baseline_spikiness_gini_threshold:.3f})'
                    if baseline_gini > self.baseline_spikiness_gini_threshold
                    else f'cv({cv:.2f}) > threshold({self.baseline_variance_cv_threshold:.2f})'
                )
                logger.info('Spike suppression SKIP: %s and current(%.1f) <= ceiling(%.1f)',
                            reason, current_val, spike_ceiling)
                return False

        # Gate 4: Correlated check (suppress service-level events)
        if self.correlated_check and baseline_ts is not None and len(baseline_ts) >= 1:
            current_time = new_ts.index[-1]

            recent_cutoff = current_time - 5 * 60
            recent_baseline = baseline_ts[baseline_ts.index >= recent_cutoff]
            if len(recent_baseline) >= 1:
                new_5_window = new_ts.iloc[-min(5, len(new_ts)):]
                baseline_5med = np.median(recent_baseline.values)
                new_5med = float(np.median(new_5_window.values))
                corr_ratio = new_5med / baseline_5med if baseline_5med > 0 else float('inf')
                if baseline_5med > 0 and corr_ratio < self.correlation_ratio_threshold:
                    logger.info('Correlated check SKIP: ratio(%.2f) < threshold(%.1f)',
                                corr_ratio, self.correlation_ratio_threshold)
                    return False

            concurrent_baseline = baseline_ts[abs(baseline_ts.index - current_time) <= 2 * 60]
            if len(concurrent_baseline) >= 1:
                baseline_point = concurrent_baseline.iloc[-1]
                new_point = new_ts.iloc[-1]
                conc_ratio = new_point / baseline_point if baseline_point > 0 else float('inf')
                conc_suppressed = (
                    baseline_point > self.concurrent_spike_min
                    and new_point > self.concurrent_spike_min
                    and conc_ratio < self.correlation_ratio_threshold
                )
                if conc_suppressed:
                    logger.info('Concurrent spike SKIP: both above min(%.1f), ratio(%.2f)',
                                self.concurrent_spike_min, conc_ratio)
                    return False

        # Gate 5: Robust z-score
        z_score = (current_val - baseline_median) / baseline_std
        if z_score <= self.threshold:
            logger.info('Z-score gate FAIL: z_score(%.1f) <= threshold(%.1f)', z_score, self.threshold)
            return False

        logger.info(
            'Anomaly detected: val=%.1f, z_score=%.1f, pct_rank=%.3f, recent_median=%.1f',
            current_val, z_score, pct_rank, recent_median,
        )
        return True
