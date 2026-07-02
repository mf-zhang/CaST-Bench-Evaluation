# CaST-Bench Inference

This directory holds the inference **prompt** and a **sample model output**.

| File | Description |
|------|-------------|
| `prompt.md` | The exact MCQ inference prompt used for CaST-Bench. It asks the model to pick an answer (A–F) and ground it with per-instance, per-second bounding-box evidence, returned as JSON. |
| `sonnet-4.6_predictions.jsonl` | Example output: Claude Sonnet 4.6's responses to this prompt on 80 QAs. One JSON object per line — `{"prompt": <full prompt>, "predict": <model response>}`. |

## Generate your own predictions

There is no bundled runner — use `prompt.md` with any video-capable model. For each QA,
fill in the `{question}` and `{A}`–`{F}` placeholders, send it together with the video, and
collect the model's JSON response. Write one `{"prompt", "predict"}` object per line to a
`.jsonl` file, exactly like `sonnet-4.6_predictions.jsonl`. The metrics suite joins each
prediction to its ground-truth QA by the question text in `prompt`, so the file order does
not need to match the GT.

## Score predictions

Feed a predictions `.jsonl` to the metrics suite — see [`../metrics/`](../metrics/). The same
Sonnet 4.6 file is bundled there as a runnable example.
