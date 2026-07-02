# mitl — Manual-in-the-Loop Constraint Projection for ICS Anomaly Detection

> **Thesis:** If you cannot add signal intelligence, add a manual in the loop.

`mitl` encodes physical system invariants from ICS engineering documentation and detects attacks that statistical ML misses — because it enforces cross-layer structural properties that no density model can observe.

Companion to [CATT](https://github.com/sassom2112/catt-ccs) (Constrained-Adversarial Tabular Telemetry, AISec @ CCS 2026):
- **CATT** shows constraint projection exposes *inflated evasion* in adversarial NIDS
- **MITL** shows constraint projection closes *detection gaps* in ICS anomaly detection

Same constraint mechanism. Two sides of the same coin.

---

## Install

```bash
pip install mitl
# With AWS Bedrock LLM extraction:
pip install "mitl[bedrock]"
```

---

## The Problem

A replay attack in an ICS captures legitimate Modbus traffic and retransmits it verbatim. Every packet is valid. The statistical distribution of protocol fields is normal. A network ML classifier cannot distinguish it from normal traffic — by design. Our best supervised model (LightGBM, 48 features) hits a **49.9% replay recall ceiling** regardless of tuning.

The attack is not statistically anomalous. It is **physically impossible**. The replayed commands freeze the valve actuator at a captured position while network traffic volume *increases*. Two independent signals — elevated network commands co-occurring with frozen actuator state — are physically contradictory. Only a layer that knows what the process spec says can detect it.

---

## Quickstart

### Option A — Python API

```python
from mitl import BaselineCalibrator, ConstraintProjector, etapr_report
from mitl.datasets.hai import build_hai_spec, hai_constraints
import pandas as pd

# 1. Load your ICS data
train_df = pd.read_csv("hai_train.csv")
test_df  = pd.read_csv("hai_test.csv")

# 2. Build spec from the engineering manual (hand-coded or LLM-extracted)
spec = build_hai_spec()   # HAI 22.04 — 19 tags, 5 control loops

# 3. Calibrate behavioral baseline (no attack labels needed)
baseline = BaselineCalibrator(warmup_fraction=0.15).fit(train_df, spec)

# 4. Project constraints over test windows
projector = ConstraintProjector(spec, baseline, hai_constraints(), window_col="bucket")
results   = projector.evaluate(test_df)

# 5. Evaluate with eTaPR (HAI's mandated metric)
y_true = test_df.groupby("bucket")["attack"].max().values
y_pred = [int(r.flagged) for r in results]
report = etapr_report(y_true, y_pred, label="MITL-Calibrated", buffer_steps=60)
print(f"eTaPR F1: {report['etapr_f1']:.3f}")
```

### Option B — CLI

```bash
# Ingest a new engineering manual via Bedrock Claude
mitl ingest manual.pdf --name my-system --system "water treatment PLC"

# See what was extracted and what needs human review
mitl review my-system

# Approve the bounds a human has spot-checked
mitl approve my-system --tags P2_RTR P1_FIT01

# Run detection against a sensor CSV
mitl evaluate sensor_data.csv --spec my-system --output report.json
```

---

## Manual Library

The library stores extracted `ConstraintSpec` objects at `~/.mitl/library/`.
Each spec carries page-level provenance: every constraint knows exactly which
table row or block diagram figure it came from.

```
~/.mitl/library/
  HAI-22.04/entry.json      # 19 tags, 5 loops, status=approved
  my-plc-v3/entry.json      # LLM-extracted, 31 tags, status=pending review
```

### Review workflow for a new manual

```bash
mitl ingest scada_manual.pdf --name plant-a --system "gas compression"
# → Bedrock Claude extracts tag bounds + control loop topology
# → Spec saved with confidence=0.85 (LLM), status=pending

mitl review plant-a
# ⚠ 4 tags need review: P3_FIT02, P3_FCV01, P4_TIT01, P4_PIT01
#   Run: mitl approve plant-a --tags P3_FIT02 P3_FCV01 P4_TIT01 P4_PIT01

mitl approve plant-a --tags P3_FIT02 P3_FCV01 P4_TIT01 P4_PIT01
# ✓ plant-a: 31/31 items reviewed, status=approved, confidence=1.000

mitl evaluate live_telemetry.csv --spec plant-a
# eTaPR F1: 0.941  (C3 fires on all SP/PV divergences, C1 on 2 sensor injections)
```

---

## Four Constraints

Every spec produces four constraint classes automatically:

| ID | Name | Source in manual | Fires on |
|----|------|-----------------|----------|
| C1 | Saturation bounds | Data points table (explicit min/max per tag) | Sensor injection outside physical range |
| C2 | Rate limiter invariant | Block diagrams showing Rate Limiter/Ramp blocks | Step-change in control output |
| C3 | SP/PV tracking | Loop description + attack scenario table | Frozen PV while SP ramps (replay/spoofing) |
| C4 | Cross-layer coupling | Architecture diagram cross-layer edges | Upstream command change with frozen downstream PV |

For **HAI 22.04**, C3 encodes AP27: *if AutoSD (speed setpoint) changes, SIT01 (speed sensor) must track*. Frozen SIT01 during active AutoSD ramp = sensor spoofing. Source: Figure 11, p.10 + attack scenario AP27, p.27.

---

## Results

### ICSSim v2 (water treatment)

| Method | Attack Labels | Replay Recall |
|--------|--------------|--------------|
| Network ML (LightGBM) | Yes | 49.9% |
| MITL-Static (spec only) | **No** | 25.0% |
| Cross-layer ML | Yes | 95.4% |
| **MITL-Calibrated** (spec + warm-up) | **No** | **100.0%** |

The 49% ceiling is a structural property of network-only models, not a tuning problem. MITL closes it entirely without a single attack label.

### HAI 22.04 (steam turbine HIL)

Evaluated with eTaPR (Enhanced Time-series Aware Precision and Recall), the HAI benchmark metric. Results pending full Kaggle run — see the [research repo](https://github.com/sassom2112/ics-sim-anomaly-detection).

---

## LLM-Assisted Extraction (Bedrock)

The "any manual" claim: `mitl ingest` sends the PDF to Claude via AWS Bedrock and returns a structured `ConstraintSpec`. No domain engineer needs to hand-code constraints for every new dataset.

```python
import boto3
from mitl.extract.bedrock import extract_constraint_spec

client = boto3.client("bedrock-runtime", region_name="us-east-1")
spec   = extract_constraint_spec("plant_manual.pdf", client)
print(spec.describe())
# ConstraintSpec(plant-manual / Rev 3.2)
#   Tags: 47  Loops: 8  Confidence: 0.87
```

The confidence score tells you how completely the manual was parsed. Hand-coded specs have confidence 1.0; LLM-extracted start at 0.85 and reach 1.0 after `mitl approve`.

---

## Reference Encodings

`mitl.datasets` contains two fully hand-coded reference implementations:

- `mitl.datasets.hai` — HAI 22.04 steam turbine (19 tags, 5 loops, from Technical Manual v4.0)
- `mitl.datasets.icssim` — ICSSim v2 water treatment (PLC register layout + fill/drain spec)

These are the baselines for the ablation in the paper: LLM extraction vs. hand-coded.

---

## Architecture

```
PDF Manual  ──► SpecReader (Bedrock LLM or hand-authored)
                    │
                    ▼ ConstraintSpec (tags + loops + page/figure provenance)
Train CSV   ──► BaselineCalibrator (warm-up 15%, no labels)
                    │
                    ▼ BehavioralBaseline (per-tag mean/var/rate)
Test CSV    ──► ConstraintProjector (C1–C4 applied per window)
                    │
                    ▼ WindowConstraintResult → eTaPR report
                         └── to_feature_row() → AutoML feature matrix
```

---

## Citation

```bibtex
@inproceedings{mitl2026,
  title     = {{Manual-in-the-Loop (MITL)}: Specification-Derived Constraint
               Projection for {ICS} Anomaly Detection},
  author    = {Anonymous},
  booktitle = {Proceedings of the ACM Workshop on Artificial Intelligence
               and Security (AISec)},
  year      = {2026},
  note      = {Companion to CATT (AISec @ CCS 2026)}
}
```

---

## Related

- [ics-sim-anomaly-detection](https://github.com/sassom2112/ics-sim-anomaly-detection) — Research repo: all sprint experiments, paper LaTeX, Kaggle notebook
- [catt-ccs](https://github.com/sassom2112/catt-ccs) — CATT: constraint projection on the evasion side (AISec @ CCS 2026)
