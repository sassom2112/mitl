"""eTaPR metric correctness.

mitl/CLAUDE.md test target: "etapr_report() returns correct eTaPR F1 on known
input." These are pure functions on binary arrays, so the expected values are
exact — no fixtures, no randomness in the assertions.
"""
from __future__ import annotations

import numpy as np

from mitl.metrics import (
    etapr_f1,
    etapr_report,
    extract_segments,
    time_aware_recall,
)


def test_extract_segments_inclusive_pairs():
    labels = np.array([0, 1, 1, 0, 0, 1])
    assert extract_segments(labels) == [(1, 2), (5, 5)]


def test_extract_segments_edges():
    assert extract_segments(np.array([1, 1, 1])) == [(0, 2)]   # trailing run closes
    assert extract_segments(np.array([0, 0, 0])) == []


def test_perfect_detection_scores_one():
    y = np.array([0, 1, 1, 0, 1, 0])
    assert etapr_f1(y, y) == (1.0, 1.0, 1.0)


def test_false_positive_only_zeros_precision():
    # No true events, one predicted cluster: nothing missed (recall 1) but the
    # prediction overlaps no real event (precision 0) -> F1 0.
    eta_p, eta_r, f1 = etapr_f1(np.array([0, 0, 0, 0]), np.array([0, 1, 1, 0]))
    assert eta_p == 0.0
    assert eta_r == 1.0
    assert f1 == 0.0


def test_lead_time_buffer_credits_early_warning():
    true = [(100, 110)]
    # Prediction inside [event_start - buffer, event_end] counts as a detection.
    assert time_aware_recall(true, [(50, 60)], buffer_steps=60) == 1.0
    # Prediction entirely before the buffer window is a miss.
    assert time_aware_recall(true, [(0, 30)], buffer_steps=60) == 0.0


def test_report_exposes_both_metric_families_in_range():
    rng = np.random.default_rng(0)
    y_true = (rng.random(200) > 0.7).astype(int)
    y_pred = (rng.random(200) > 0.7).astype(int)
    rep = etapr_report(y_true, y_pred, label="unit", buffer_steps=30)

    # Both standard and eTaPR are reported so the gap (MITL paper Table 2) is visible.
    for key in ("std_f1", "etap", "etar", "etapr_f1", "n_true_events", "n_pred_clusters"):
        assert key in rep
    for key in ("std_f1", "etap", "etar", "etapr_f1"):
        assert 0.0 <= rep[key] <= 1.0
    assert rep["label"] == "unit"
    assert rep["buffer_steps"] == 30
