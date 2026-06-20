"""Shared fixtures for the mitl test suite.

Synthetic, HAI-shaped telemetry only — no dataset download — so the suite is
self-contained and fast (mitl/CLAUDE.md: "Use HAI 22.04 sample fixtures (not
full dataset) to keep tests self-contained").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mitl.calibrate import BaselineCalibrator
from mitl.datasets.hai import build_hai_spec


def _normal_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Normal operation: turbine speed PV (SIT01) tracks its setpoint (AutoSD),
    every tag inside spec bounds. Used to calibrate a baseline with non-zero
    variance for P2_SIT01 (so 'frozen' is detectable as var << baseline)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    speed = 3000.0 + 30.0 * np.sin(t / 5.0) + rng.normal(0, 2.0, n)
    return pd.DataFrame(
        {
            "timestamp": t,
            "P2_AutoSD": speed,                              # speed setpoint
            "P2_SIT01": speed + rng.normal(0, 1.0, n),      # speed PV tracks setpoint
            "P4_ST_PS": 50.0 + 5.0 * np.sin(t / 7.0) + rng.normal(0, 0.5, n),
            "attack": 0,
        }
    )


@pytest.fixture
def hai_spec():
    return build_hai_spec()


@pytest.fixture
def normal_frame():
    return _normal_frame()


@pytest.fixture
def baseline(normal_frame, hai_spec):
    # warmup_fraction=0.5 -> 100 warm-up rows of varying SIT01 -> non-zero baseline var.
    return BaselineCalibrator(warmup_fraction=0.5).fit(normal_frame, hai_spec)
