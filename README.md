# CaST-Bench — Evaluation & Inference

Official evaluation code for **CaST-Bench**: *Causal Chain-Grounded Spatio-Temporal
Reasoning in Video Question Answering*.

🏠 [Homepage](https://woven-by-toyota.github.io/CaST-Bench/) &nbsp;|&nbsp; 📄 [Paper](https://arxiv.org/abs/2605.23216) &nbsp;|&nbsp; 🤗 [Dataset](https://huggingface.co/datasets/wovenbytoyota-vai/CaST-Bench) &nbsp;|&nbsp; 🏆 [Leaderboard](https://wovenbytoyota-vai-cast-bench-leaderboard.hf.space/)

This repository contains two pieces:

| Directory | What it does |
|-----------|--------------|
| [`inference/`](inference/) | The exact inference [`prompt.md`](inference/prompt.md) plus a sample model output. |
| [`metrics/`](metrics/) | Scores MCQ predictions with the official CaST-Bench metrics. Model-agnostic — works on any predictions file in the supported format. |

## Dataset

The ground-truth annotations and videos are **not** included in this repository.
Download them from the CaST-Bench dataset release on Hugging Face:

> 🤗 **Dataset:** https://huggingface.co/datasets/wovenbytoyota-vai/CaST-Bench

Use the videos to generate model predictions, and the ground-truth JSON as the `--gt`
input to the metrics. Compare your results against published baselines on the
[CaST-Bench Leaderboard](https://wovenbytoyota-vai-cast-bench-leaderboard.hf.space/).

## Quick Start

1. **Generate predictions** — run any video-capable model with
   [`inference/prompt.md`](inference/prompt.md) and write one prediction per line to a
   `.jsonl` (see [`inference/README.md`](inference/README.md)). A sample output from Claude
   Sonnet 4.6 is bundled at [`inference/predictions/sonnet-4.6_predictions.jsonl`](inference/predictions/sonnet-4.6_predictions.jsonl).

2. **Evaluate** (see [`metrics/README.md`](metrics/README.md)) — `evaluate.sh` is
   preconfigured to run the bundled example, no dataset download needed:

   ```bash
   cd metrics
   bash evaluate.sh
   ```

## Metrics

See [`metrics/METRICS.md`](metrics/METRICS.md) for exact definitions. In brief:

- **QA Accuracy** — MCQ answer correctness.
- **IM-tIoU** — Instance-Matched temporal IoU between predicted and GT evidence.
- **IM-vIoU** — coverage-aware spatio-temporal IoU (`tIoU × mean_sIoU`).
- **Faithful Rate** — correct answer **and** well-grounded evidence (vIoU ≥ τ_st).
- **Spurious Rate** — correct answer with no instance reaching vIoU ≥ τ_st.

## License

[MIT](LICENSE).

## Citation

```bibtex
@inproceedings{zhang2026castbench,
  title = {CaST-Bench: Benchmarking Causal Chain-Grounded Spatio-Temporal Reasoning for Video Question Answering},
  author = {Zhang, Mingfang and Pan, Jingjing and Kumar, Ashutosh and Saini, Rajat and Erdogan, Mustafa and Yang, Hsuan-Kung and Kang, Caixin and Huang, Yifei and Sato, Yoichi and Kong, Quan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2026}
}
```
