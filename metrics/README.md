# CaST-Bench Evaluation — `metrics`

Official evaluation code for **CaST-Bench**: Causal Chain-Grounded Spatio-Temporal Reasoning in Video Question Answering.

Evaluates **MCQ** (multiple-choice) predictions: answer accuracy plus temporal /
spatio-temporal grounding of the supporting evidence.


## Quick Start

`evaluate.sh` is preconfigured to run the bundled example — the 80-QA ground truth in
[`example/castbench_hf.jsonl`](example/castbench_hf.jsonl) paired with the Claude
Sonnet 4.6 predictions in [`../inference/`](../inference/). No dataset download needed:

```bash
bash evaluate.sh
```

Expected output (a compact JSON summary; the full per-metric breakdown is written to `--out_dir`):

```json
{
  "model_name": "claude-sonnet-4-6",
  "qa_accuracy": 0.75,
  "qa_faithful": 0.35,
  "qa_temporal_iou": 0.29454464601982977,
  "qa_spatiotemporal_iou": 0.07931422757156253
}
```

Pass `--debug_print` for the verbose per-metric breakdown (Temporal/ST recall, bbox-size analysis, progress bar).


To evaluate your own model, edit the variables at the top of `evaluate.sh`
(`GT_PATH`, `PRED_PATH`, `MODEL_NAME`, and optionally `START_INDEX` / `END_INDEX` to score a
subset), then run `bash evaluate.sh` again. Output files for a sliced run gain a suffix,
e.g. `results_overall_0_2066.json`.

---

## Input Formats

### Ground Truth — `castbench_hf.jsonl`

The HuggingFace release format: one QA per line. `evidence` is a **JSON-encoded string**
(a list of evidence entries, each with per-second bounding boxes under `bboxes_in_range`):

```jsonl
{"video": "sav_000110.mp4", "question": "Given that the employee picks up the tray, what will they do next?", "options": {"A": "Turn back to the table", "B": "Take the tray to storage", "C": "...", "D": "...", "E": "...", "F": "Walk away with the tray"}, "answer": "F", "evidence": "[{\"evidence_start_time\": \"00:01\", \"evidence_end_time\": \"00:02\", \"evidence_instance_id\": \"person_vzg\", \"evidence_rationale\": \"The employee picks up the tray.\", \"bboxes_in_range\": {\"1\": \"[138,322,459,1089]\", \"2\": \"[58,288,344,1087]\"}}]"}
```

Evidence entries are grouped by `evidence_instance_id` into per-instance timelines for matching.

### Prediction JSONL

Point `--pred` to a `.jsonl` file (as emitted by the inference runner). Each line has a
`prompt` field and a `predict` field holding the model's JSON response:

```jsonl
{"prompt": "...Question: <q>\nOptions:\n...", "predict": "<think>...</think>\n{\"answer_choice\": \"F\", \"instances\": [{\"instance_name\": \"person_vzg\", \"evidences\": [{\"evidence_start_time\": \"00:01\", \"evidence_end_time\": \"00:02\", \"evidence_rationale\": \"...\", \"bboxes_in_time_range\": {\"00:01\": \"[138,322,459,1089]\"}}]}]}"}
```

Predictions are joined to GT QAs **by question text** (extracted from the prompt), so the
prediction file need not be in the same order as the GT. If a question can't be matched it
falls back to positional alignment. The `<think>...</think>` block is stripped automatically.

---

## Metrics

All metrics are computed per QA and averaged across all QAs.

| Metric | Description |
|--------|-------------|
| **QA Accuracy** | Exact-match on `answer_choice` |
| **Temporal Recall@τ_t** | Fraction of GT instances whose matched prediction has tIoU ≥ τ_t (default 0.5) |
| **IM-tIoU** | Mean tIoU per GT instance; unmatched = 0 |
| **ST Recall@τ_st** | Fraction of GT instances whose matched prediction has vIoU ≥ τ_st (default 0.1) |
| **IM-vIoU** | Mean vIoU per GT instance; unmatched = 0 |
| **Faithful Rate** | QA correct AND mean predicted vIoU ≥ τ_st |
| **Spurious Rate** | QA correct AND no single matched instance has vIoU ≥ τ_st |

**vIoU** (coverage-aware Video IoU):
```
vIoU(p, g) = tIoU(p, g) × mean_{t ∈ overlap} sIoU(p_t, g_t)
```

Instance matching is greedy 1:1 by descending spatial IoU over overlapping seconds, requiring at least `eps_overlap` overlapping seconds.

---

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--gt` | required | Path to GT `castbench_hf.jsonl` |
| `--pred` | required | Path to the predictions `.jsonl` file |
| `--model_name` | `model` | Model name shown in the JSON summary |
| `--eps_overlap` | `1` | Min overlapping seconds for a valid instance match |
| `--tau_t` | `0.5` | tIoU threshold for Temporal Recall |
| `--tau_st` | `0.3` | vIoU threshold for ST Recall / Faithful / Spurious |
| `--use_coverage_aware_score` | `false` | Use tIoU×sIoU for matching (vs. plain mean sIoU) |
| `--start_index` | None | Evaluate QAs from this index (inclusive) |
| `--end_index` | None | Evaluate QAs up to this index (exclusive) |
| `--out_dir` | `outputs` | Output directory |
| `--ignore_missing_preds` | `false` | Skip missing QAs instead of scoring them 0 |
| `--debug_print` | off | Print the verbose per-metric breakdown instead of just the JSON summary |

---

## Output Files

Results are written to `--out_dir`:

- **`results_overall.json`** — aggregated metrics across all evaluated QAs
- **`results_per_qa.json`** — per-QA breakdown with temporal, spatiotemporal, and per-QA scores

When `--start_index` / `--end_index` are used, filenames include a suffix: `results_overall_0_2066.json`.

### `results_overall.json` schema

```json
{
  "overall_per_qa": {
    "n_qas": 2066,
    "qa_accuracy": 0.5257,
    "qa_temporal_r1_tau": 0.2329,
    "qa_temporal_m_tiou_pergt": 0.2153,
    "qa_st_r1_tau_st": 0.0814,
    "qa_st_m_viou_pergt": 0.0246,
    "qa_faithful_answer_accuracy": 0.0760,
    "qa_spurious_answer_rate": 0.4226
  },
  "missing_pred_files": 0,
  "config": { "eps_overlap": 2, "tau_t": 0.5, "tau_st": 0.1, "..." : "..." }
}
```

---

## Directory Layout

```
metrics/
├── src/
│   ├── evaluate_benchmark.py   # main evaluation entrypoint
│   └── metrics_core.py         # tIoU, vIoU, greedy matching
├── evaluate.sh                 # single-model eval
├── example/                    # runnable example: 80-QA GT (paired with inference/ predictions)
└── outputs/                    # evaluation results per model (created at runtime)
```
