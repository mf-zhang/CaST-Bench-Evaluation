# CaST-Bench Metrics Reference

This document describes the meaning and exact computation of every output metric in `metrics`.
Code entry points: `src/evaluate_benchmark.py` + `src/metrics_core.py`.

---

## Evaluation Overview

Each QA item contains:
- **GT**: the correct answer option (A/B/C/D) + a number of evidence instances (each instance has a time span + per-second bboxes).
- **Pred**: the model's predicted answer option + a number of predicted instances (same format).

Evaluation pipeline:
1. **Greedy 1:1 matching**: for each item, greedily match predicted instances to GT instances, sorting candidate pairs by vIoU descending and assigning one at a time (each pred / GT is matched at most once).
2. **Compute temporal metrics** (time spans only).
3. **Compute spatio-temporal metrics** (time span × spatial bbox).
4. **Judge answer correctness + Faithful/Spurious**.
5. **Average over all items** to get the final scores.

---

## Hyperparameters (defaults)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `eps_overlap` | 2 | Minimum number of overlapping seconds between pred and GT time spans for a valid match (filters noise). |
| `tau_t` | 0.5 | tIoU hit threshold for Temporal Recall. |
| `tau_st` | 0.1 | vIoU hit threshold for ST Recall and Faithful. |

---

## Core Operator Definitions

### tIoU (Temporal IoU, time-span IoU)

```
tIoU(pred, gt) = |pred ∩ gt| / |pred ∪ gt|
```

Time spans are measured in whole seconds, inclusive of endpoints.

### sIoU (Spatial IoU, single-frame bbox IoU)

Standard 2D IoU: intersection area / union area of two axis-aligned rectangles.

### vIoU (Coverage-Aware Spatio-Temporal IoU)

```
vIoU(pred_i, gt_j) = tIoU(pred_i, gt_j)  ×  mean_sIoU(T_ov)
```

where `T_ov` is the intersection of the pred and GT time spans (a list of whole seconds), and
`mean_sIoU` is the average of the per-second sIoU over `T_ov`.

If `|T_ov| < eps_overlap`, it returns 0 directly (avoiding noise from a single overlapping second).

---

## Temporal Metrics (time span only, ignoring bbox)

### `qa_temporal_r1_tau` — Temporal Recall @ τ_t

**Meaning**: what fraction of GT instances the model "hits" on the time span (tIoU ≥ τ_t = 0.5).

**Computation**:
```
For each item:
  For each GT instance j:
    if there exists a matched pred_i with tIoU(pred_i, gt_j) >= tau_t → hit
  r1_tau = number of hit GT instances / total GT instances

Average over all items: avg(r1_tau)
```

---

### `qa_temporal_m_tiou_tponly` — mean tIoU (TP pairs only)

**Meaning**: mean tIoU over successfully matched pred-GT pairs only, ignoring unmatched GT (missed detections).

**Computation**:
```
For each item:
  TP_set = {(pred_i, gt_j) | matched}
  if TP_set is empty → 0
  else → mean(tIoU(pred_i, gt_j) for all TP pairs)

Average over all items: avg(...)
```

> **Note**: the denominator here is the number of TP pairs; it does not penalize missed detections, so it is optimistic.

---

### `qa_temporal_m_tiou_pergt` — mean tIoU (per GT)

**Meaning**: same as above, but the denominator is the total number of GT instances (missed detections count as 0), which is stricter.

**Computation**:
```
For each item:
  tiou_sum = sum(tIoU(pred_i, gt_j) for all matched (i,j))
  m_tiou_pergt = tiou_sum / max(1, len(gt_insts))

Average over all items: avg(...)
```

---

## Spatio-Temporal Metrics (time span + spatial bbox)

### `qa_st_r1_tau_st` — ST Recall @ τ_st

**Meaning**: what fraction of GT instances the model "hits" on spatio-temporal localization (vIoU ≥ τ_st = 0.1).

**Computation**:
```
For each item:
  For each GT instance j:
    if there exists a matched pred_i with vIoU(pred_i, gt_j) >= tau_st → hit
  r1_tau_st = number of hits / total GT instances

Average over all items: avg(r1_tau_st)
```

---

### `qa_st_m_viou_tponly` — mean vIoU (TP pairs only)

**Meaning**: mean vIoU over matched pairs only; does not penalize missed detections.

**Computation**:
```
For each item:
  if no matches → 0
  else → mean(vIoU(pred_i, gt_j) for all matched pairs)

Average over all items: avg(...)
```

---

### `qa_st_m_viou_pergt` — mean vIoU (per GT) [primary spatio-temporal metric]

**Meaning**: denominator is the total number of GT instances, missed detections count as 0; comprehensively reflects spatio-temporal localization quality.

**Computation**:
```
For each item:
  viou_sum = sum(vIoU(pred_i, gt_j) for all matched pairs)
  m_viou_pergt = viou_sum / max(1, len(gt_insts))

Average over all items: avg(...)
```

> This is the **most central** spatio-temporal metric: it penalizes spatial inaccuracy (low sIoU), temporal offset (low tIoU), and missed detections (dividing by the total GT count).

---

## QA Answer Accuracy Metrics

### `qa_accuracy` — QA Accuracy

**Meaning**: the fraction of items where the predicted answer matches the GT answer (letter match A/B/C/D in MCQ mode).

**Computation**:
```
For each item:
  qa_acc = 1.0 if pred_choice == gt_choice else 0.0

Average over all items: avg(qa_acc)
```

---

### `qa_faithful_answer_accuracy` — Faithful Rate

**Meaning**: the fraction of items that are answered correctly **and** have at least one matched evidence instance with vIoU ≥ τ_st. That is, "answered correctly, with verifiable visual spatio-temporal evidence backing it up".

**Computation**:
```
For each item:
  any_hit_st = any(vIoU(pred_i, gt_j) >= tau_st for matched pairs)
  faithful = 1.0 if (qa_acc == 1.0 AND any_hit_st) else 0.0

Average over all items: avg(faithful)
```

---

### `qa_spurious_answer_rate` — Spurious Rate

**Meaning**: the fraction of items that are answered correctly **but** have no matched instance with vIoU ≥ τ_st. That is, "answered correctly, but cannot prove with visual evidence that the video was genuinely understood" (possibly relying on text priors or a lucky guess).

**Computation**:
```
For each item:
  spurious = 1.0 if (qa_acc == 1.0 AND NOT any_hit_st) else 0.0

Average over all items: avg(spurious)
```

**Key invariant**: `Faithful Rate + Spurious Rate = QA Accuracy` (the two are complementary — there is no "neither" case).

---

## Metric Summary and Interpretation

| Metric | Dimension | Penalizes missed detections | Higher is better |
|--------|-----------|-----------------------------|------------------|
| `qa_temporal_r1_tau` | Temporal | Yes (per GT) | ✓ |
| `qa_temporal_m_tiou_tponly` | Temporal | No (TP only) | ✓ |
| `qa_temporal_m_tiou_pergt` | Temporal | Yes (per GT) | ✓ |
| `qa_st_r1_tau_st` | Spatio-temporal | Yes (per GT) | ✓ |
| `qa_st_m_viou_tponly` | Spatio-temporal | No (TP only) | ✓ |
| `qa_st_m_viou_pergt` | Spatio-temporal | Yes (per GT) | ✓ |
| `qa_accuracy` | Answer | — | ✓ |
| `qa_faithful_answer_accuracy` | Answer + spatio-temporal evidence | — | ✓ |
| `qa_spurious_answer_rate` | Correct answer without evidence support | — | ✗ (lower is better) |

**Recommended core metrics to report** (highlighted in the paper):
1. `qa_accuracy` — the model's question-answering ability.
2. `qa_faithful_answer_accuracy` — the fraction of items the model genuinely "understood from the video".
3. `qa_st_m_viou_pergt` — spatio-temporal localization quality.
4. `qa_temporal_m_tiou_pergt` — temporal localization quality.
