"""API-stability tripwire.

mitl/CLAUDE.md: ics-sim-anomaly-detection depends on this package, so the public
signatures of BaselineCalibrator, ConstraintProjector, etapr_report,
build_hai_spec, and hai_constraints must stay fixed — any change is a MAJOR
version bump. This test fails loudly the moment one of them drifts.
"""
from __future__ import annotations

import inspect

from mitl import BaselineCalibrator, ConstraintProjector, etapr_report
from mitl.datasets.hai import build_hai_spec, hai_constraints


def _params(func) -> list[str]:
    return list(inspect.signature(func).parameters)


def test_baseline_calibrator_signature_frozen():
    assert _params(BaselineCalibrator.__init__) == ["self", "warmup_fraction", "skip_cols"]
    assert _params(BaselineCalibrator.fit) == ["self", "df", "spec"]


def test_constraint_projector_signature_frozen():
    assert _params(ConstraintProjector.__init__) == [
        "self",
        "spec",
        "baseline",
        "constraints",
        "window_col",
    ]
    assert _params(ConstraintProjector.evaluate) == ["self", "df"]


def test_etapr_report_signature_frozen():
    assert _params(etapr_report) == ["true_labels", "pred_labels", "label", "buffer_steps"]


def test_hai_builders_callable_with_no_args():
    spec = build_hai_spec()
    assert spec.dataset_name == "HAI-22.04"

    constraints = hai_constraints()
    assert [c.constraint_id for c in constraints] == ["C1", "C2", "C3", "C4"]
