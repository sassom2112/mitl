"""Constraint-projection correctness — the executable form of the MITL thesis.

mitl/CLAUDE.md test targets:
  - ConstraintProjector.evaluate() flags physically impossible windows
  - C3 (tracking / AP27 replay): turbine speed setpoint (P2_AutoSD) ramps while
    the speed PV (P2_SIT01) is frozen. That is physically impossible but
    statistically normal — exactly the attack a density model cannot see and
    the reason MITL exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mitl.datasets.hai import (
    c1_saturation_bounds,
    c3_tracking_invariant_p2sc,
    hai_constraints,
)
from mitl.project import ConstraintProjector


def _replay_window(n: int = 30, bucket: int = 1) -> pd.DataFrame:
    """AP27-style replay: AutoSD ramps (active speed change) but SIT01 is frozen.
    P4_ST_PS held constant so this window isolates C3 from the C4 cross-layer rule."""
    return pd.DataFrame(
        {
            "P2_AutoSD": np.linspace(3000.0, 3120.0, n),   # range 120 RPM > 50 -> active ramp
            "P2_SIT01": np.full(n, 3000.0),                # frozen actuator -> variance 0
            "P4_ST_PS": np.full(n, 50.0),
            "bucket": bucket,
        }
    )


def _clean_window(n: int = 30, bucket: int = 0, seed: int = 1) -> pd.DataFrame:
    """Normal: AutoSD ramps AND SIT01 tracks it (high variance). The discriminating
    case for C3 — a ramp alone must not fire; only a ramp with a frozen PV does."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(3000.0, 3100.0, n)
    return pd.DataFrame(
        {
            "P2_AutoSD": ramp + rng.normal(0, 0.5, n),
            "P2_SIT01": ramp + rng.normal(0, 0.5, n),      # PV follows setpoint -> not frozen
            "P4_ST_PS": np.full(n, 50.0),
            "bucket": bucket,
        }
    )


def test_c3_flags_frozen_actuator_under_ramp(baseline, hai_spec):
    viols = c3_tracking_invariant_p2sc(_replay_window(), baseline, hai_spec)
    assert len(viols) == 1
    assert viols[0].constraint_id == "C3"
    assert viols[0].severity == "definite"


def test_c3_silent_when_pv_tracks_setpoint(baseline, hai_spec):
    # A genuine ramp with the PV tracking it must NOT be flagged (no false positive).
    assert c3_tracking_invariant_p2sc(_clean_window(), baseline, hai_spec) == []


def test_c1_flags_value_above_spec_ceiling(baseline, hai_spec):
    # P2_SIT01 spec ceiling is 3200 RPM; 5000 is physically impossible.
    window = pd.DataFrame({"P2_SIT01": [3000.0, 5000.0, 3000.0]})
    viols = c1_saturation_bounds(window, baseline, hai_spec)
    assert any(v.constraint_id == "C1" for v in viols)


def test_c1_silent_within_bounds(baseline, hai_spec):
    window = pd.DataFrame({"P2_SIT01": [3000.0, 3100.0, 2950.0]})
    assert c1_saturation_bounds(window, baseline, hai_spec) == []


def test_projector_flags_replay_window_not_clean(baseline, hai_spec):
    """End-to-end through ConstraintProjector.evaluate(): the replay window is
    flagged (with C3 in the reason), the clean window is not."""
    df = pd.concat([_clean_window(bucket=0), _replay_window(bucket=1)], ignore_index=True)
    projector = ConstraintProjector(hai_spec, baseline, hai_constraints(), window_col="bucket")
    by_bucket = {r.bucket_ts: r for r in projector.evaluate(df)}

    assert by_bucket[0].flagged is False
    assert by_bucket[1].flagged is True
    assert "C3" in by_bucket[1].flag_reason()
