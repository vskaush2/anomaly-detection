"""V_DIFF: Bad-change detector based on version difference.

Detects metric regressions in a newly deployed build by comparing a short
post-deploy window against a sliding window of the old build's history.

Algorithm outline
-----------------
1. De-noise and scale both time series.
2. Slice the old-build series into fixed-width windows; the new-build window
   is treated as an additional data point.
3. Run DBSCAN over the window means. If the new-build window is an outlier
   (label == -1) and its mean exceeds the old-build mean, proceed.
4. Slope check — suppress if the signal is sharply falling (not a regression).
5. GINI check — suppress if the signal is highly bursty/spiky.
"""

import time
import logging

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from anomaly_detection.bcd._utils import (
    GINI, SLOPE,
    denoise_rolling_median, denoise_mean,
    de_autofill,
)

logger = logging.getLogger('vdiff')


class V_DIFF:
    def __init__(self, eps=0.3, minPts=2, avgMin=20.0, slopeMin=-30.0, giniMax=0.3, win=600):
        """
        Parameters
        ----------
        eps : float
            DBSCAN epsilon (0–1.0 after MinMax scaling). Default 0.3.
        minPts : int
            DBSCAN min_samples. Default 2.
        avgMin : float
            Minimum mean of the new-build window to proceed. Default 20.0.
        slopeMin : float
            Suppress if the new-build trend slope is below this value. Default -30.0.
        giniMax : float
            Suppress if the new-build GINI coefficient exceeds this value. Default 0.3.
        win : int
            Window size in seconds (resolution assumed to be 1 point/minute). Default 600.
        """
        self.eps = eps
        self.minPts = minPts
        self.avgMin = avgMin
        self.slopeMin = slopeMin
        self.giniMax = giniMax
        self.win = win

    def detect(self, baseline_ts, new_ts):
        """
        Parameters
        ----------
        baseline_ts : pd.Series
            Metric time series for the old build.
        new_ts : pd.Series
            Metric time series for the new build.

        Returns
        -------
        bool
            True if a regression is detected.
        """
        if self.win < 60:
            logger.info('win parameter is less than 60: %d', self.win)
            return False

        count = int(self.win / 60)

        if baseline_ts is None or len(baseline_ts) < count:
            logger.info('Too few data points for old build (%d), skip detect',
                        0 if baseline_ts is None else len(baseline_ts))
            return False

        if new_ts is None or len(new_ts) < count:
            logger.info('Too few data points for new build (%d), skip detect',
                        0 if new_ts is None else len(new_ts))
            return False

        baseline_ts.fillna(0, inplace=True)
        new_ts.fillna(0, inplace=True)
        de_autofill(baseline_ts)
        de_autofill(new_ts)

        new_ts = new_ts.iloc[-count:]

        if np.mean(new_ts) < self.avgMin:
            logger.info('New-build mean %.2f is below avgMin %.2f, skip detect',
                        np.mean(new_ts), self.avgMin)
            return False

        denoisedOld = denoise_rolling_median(baseline_ts, int(count / 3))
        denoisedNew = denoise_mean(new_ts)
        denoisedAll = np.append(denoisedOld, denoisedNew).reshape(-1, 1)
        scaledAll = MinMaxScaler().fit_transform(denoisedAll).flatten()
        scaledOld = scaledAll[:len(baseline_ts)]
        scaledNew = scaledAll[-len(new_ts):]

        ary = [np.mean(scaledNew)]
        i = len(scaledOld)
        while i >= count:
            ary.append(np.mean(scaledOld[i - count:i]))
            i -= count

        if len(ary) < 3:
            logger.info('Too few slices for DBSCAN: %d', len(ary))
            return False

        t0 = time.time()
        labels = DBSCAN(eps=self.eps, min_samples=self.minPts).fit(
            np.array(ary).reshape(-1, 1)
        ).labels_
        logger.debug('DBSCAN (%.3fs) labels=%s values=%s', time.time() - t0, labels, ary)

        if labels[0] == -1 and np.mean(scaledNew) > np.mean(scaledOld):
            slope = SLOPE.calc(new_ts)
            if slope < self.slopeMin:
                logger.info('Slope %.2f < slopeMin %.2f, suppressing', slope, self.slopeMin)
                return False

            trim = np.trim_zeros(new_ts.values, 'f')
            g = GINI.calc(trim)
            if g > self.giniMax:
                logger.info('GINI %.2f > giniMax %.2f, suppressing', g, self.giniMax)
                return False

            return True

        return False
