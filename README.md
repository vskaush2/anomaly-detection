# anomaly_detection

[![Tests](https://github.com/vskaush2/anomaly-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/vskaush2/anomaly-detection/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Anomaly detection algorithms for production metric monitoring. The core focus is **BCD (Bad Change Detection)** — detecting metric regressions caused by a software deployment, as opposed to organic service load changes or infrastructure events.

## Algorithms

### BCD — Bad Change Detection

Compares a short post-deploy metric window against the old build's historical baseline_ts and returns `True` if a regression is detected.

#### V_DIFF (v1)

DBSCAN-based detector. Slices the old-build history into equal-width windows and treats the new-build window as a candidate outlier. Post-filters with a slope check (suppress if the signal is already recovering) and a GINI check (suppress if the new-build window is highly bursty).

Best for: smooth, low-variance metrics where the post-deploy window is expected to be cleanly distinguishable from history.

#### V_DIFF_V2

Gate-based detector. Runs five sequential statistical checks against a long baseline_ts:

1. **Absolute floor** — new build must sustain values above a minimum threshold.
2. **Percentile rank** — current value must exceed p99 (or p99.9 for high-variance baseline_tss).
3. **Spike suppression** — suppress if the baseline_ts is naturally spiky and the new spike is within the historical range.
4. **Correlated check** — suppress if the old build is also elevated concurrently (service-level event, not a build regression).
5. **Robust z-score** — MAD-based z-score gate.

Best for: noisy or spiky production metrics where simpler outlier methods produce too many false positives.

## Quick start

```python
import pandas as pd
from anomaly_detection.bcd import V_DIFF, V_DIFF_V2

# baseline_ts: old-build metric time series (pd.Series, index = Unix timestamps in seconds)
# new_ts: new-build metric time series (pd.Series, same index format)

# V_DIFF (v1)
detector = V_DIFF(eps=0.3, minPts=2, avgMin=20.0, win=600)
detected = detector.detect(baseline_ts, new_ts)

# V_DIFF_V2
detector_v2 = V_DIFF_V2(threshold=10.0, avg_min=10.0, correlated_check=True)
detected_v2 = detector_v2.detect(baseline_ts, new_ts)
```

Both `detect()` methods return a `bool`: `True` if a regression is detected.

## Project structure

```
anomaly_detection/
├── anomaly_detection/
│   ├── __init__.py
│   └── bcd/
│       ├── __init__.py
│       ├── _utils.py           # GINI, SLOPE, SpikeBaseline, denoise helpers
│       ├── vdiff.py            # V_DIFF (v1)
│       └── vdiff_v2.py         # V_DIFF_V2
├── experiment/
│   └── bcd_vdiff.ipynb         # Demo notebook with synthetic data
├── tests/
│   ├── test_vdiff.py           # V_DIFF (v1) tests
│   └── test_vdiff_v2.py        # V_DIFF_V2 tests
└── pyproject.toml
```

## Setup with uv

[uv](https://docs.astral.sh/uv/) is the recommended way to manage the environment. It resolves and installs dependencies significantly faster than pip.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create environment and install dependencies

```bash
# Runtime dependencies only
uv venv --seed
uv pip install -e .

# Include dev dependencies (pytest + matplotlib for the notebook)
uv pip install -e ".[dev]"
```

> `--seed` includes `pip` in the venv. This is required if you open the notebook in PyCharm, which calls `pip` internally to install `ipykernel`.

### Run the tests

```bash
uv run pytest
```

### Launch the experiment notebook

```bash
uv run jupyter notebook experiment/bcd_vdiff.ipynb
```

> `jupyter` is not listed as a project dependency. Install it into the environment with `uv pip install jupyter` before launching.

## Dependencies

| Package | Version | Used by |
|---|---|---|
| `numpy` | 2.3.2 | All detectors |
| `pandas` | 2.3.1 | All detectors |
| `scikit-learn` | 1.7.2 | V_DIFF v1 (DBSCAN, MinMaxScaler) |
| `matplotlib` | 3.10.5 | Experiment notebook (dev) |
| `pytest` | 9.0.3 | Tests (dev) |

## License

Copyright 2026 Vivek Kaushik. Licensed under the [Apache License, Version 2.0](LICENSE).
