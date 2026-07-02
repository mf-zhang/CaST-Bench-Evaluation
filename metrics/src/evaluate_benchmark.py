# evaluate_benchmark.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
import ast
import argparse
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from metrics_core import (
    build_timelines_from_schema,
    greedy_match,
    parse_box,
    score_modified_viou,
    score_coverage_aware_viou,
    compute_temporal_metrics,
    compute_spatiotemporal_metrics,
    viou_star,
)

# When False (default), only the final JSON summary is printed. Set via --debug_print.
VERBOSE = False


def vprint(*a, **k):
    if VERBOSE:
        print(*a, **k)


@dataclass
class TemporalMetrics:
    r1_tau: float
    m_tiou_tponly: float
    m_tiou_pergt: float


@dataclass
class SpatioTemporalMetrics:
    r1_tau_st: float
    m_viou_tponly: float
    m_viou_pergt: float


@dataclass
class QA6Metrics:
    qa_accuracy: float
    temporal_grounding_accuracy: float
    spatiotemporal_grounding_accuracy: float
    faithful_answer_accuracy: float
    spurious_answer_rate: float


@dataclass
class QAResult:
    video_id: str
    qa_id: str
    temporal: TemporalMetrics
    spatiotemporal: SpatioTemporalMetrics
    tp: int
    fp: int
    fn: int
    six_metrics: QA6Metrics
    avg_gt_bbox_area: float


@dataclass
class OverallPerQA:
    n_qas: int
    qa_temporal_r1_tau: float
    qa_temporal_m_tiou_tponly: float
    qa_temporal_m_tiou_pergt: float
    qa_st_r1_tau_st: float
    qa_st_m_viou_tponly: float
    qa_st_m_viou_pergt: float
    qa_accuracy: float
    qa_faithful_answer_accuracy: float
    qa_spurious_answer_rate: float


def compute_bbox_area(bbox_str: str) -> float:
    if not isinstance(bbox_str, str):
        return 0.0
    bbox_str = bbox_str.strip()
    if not (bbox_str.startswith("[") and bbox_str.endswith("]")):
        return 0.0
    try:
        parts = bbox_str[1:-1].split(",")
        if len(parts) != 4:
            return 0.0
        x1, y1, x2, y2 = [float(p.strip()) for p in parts]
        return abs(x2 - x1) * abs(y2 - y1)
    except Exception:
        return 0.0


def compute_avg_gt_bbox_area(gt_qa_json: Dict[str, Any]) -> float:
    all_areas = []
    for inst in gt_qa_json.get("instances", []) or []:
        for ev in inst.get("evidences", []) or []:
            bboxes_in_time = ev.get("bboxes_in_time_range", {})
            if isinstance(bboxes_in_time, dict):
                for bbox_str in bboxes_in_time.values():
                    area = compute_bbox_area(bbox_str)
                    if area > 0:
                        all_areas.append(area)
    return (sum(all_areas) / len(all_areas)) if all_areas else 0.0


def normalize_time_key(k: str) -> str:
    k = str(k).strip()
    if ":" in k:
        parts = k.split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            mm, ss = int(parts[0]), int(parts[1])
            return f"{mm:02d}:{ss:02d}"
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            total = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return f"{total // 60:02d}:{total % 60:02d}"
        try:
            ss = int(parts[-1])
            return f"00:{ss:02d}"
        except Exception:
            return k
    try:
        ss = int(float(k))
        return f"00:{ss:02d}"
    except Exception:
        return k


def normalize_gt_qa(qa: dict):
    if isinstance(qa.get("answer"), dict):
        ans = qa["answer"]
        if "answer_choice" in ans and "answer_choice" not in qa:
            qa["answer_choice"] = ans["answer_choice"]
    for inst in qa.get("instances", []) or []:
        if "instance_id" in inst and "instance" not in inst:
            inst["instance"] = inst["instance_id"]
        for ev in inst.get("evidences", []) or []:
            bbt = ev.get("bboxes_in_time_range", {})
            if isinstance(bbt, dict):
                new_map = {normalize_time_key(tk): box for tk, box in bbt.items()}
                ev["bboxes_in_time_range"] = new_map


def load_gt_hf_jsonl(path: str):
    """Load castbench_hf.jsonl (HuggingFace release format) as GT.

    Each line has: video, question, options, answer, evidence (a JSON-encoded string).
    Returns (gt_map, ordered_keys, unique_videos). Predictions are joined to these QAs
    by question text (see load_predictions_jsonl), so line order need not match.
    """
    gt_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    ordered_keys: List[Tuple[str, str]] = []
    unique_videos: set = set()

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            video_field = row.get("video", "") or ""
            video_id = video_field[:-4] if video_field.endswith(".mp4") else video_field
            qa_id = f"qa_{i:06d}"

            # Parse evidence: JSON-encoded string → flat list
            evidence_raw = row.get("evidence", "[]") or "[]"
            evidence_list = json.loads(evidence_raw) if isinstance(evidence_raw, str) else evidence_raw

            # Group the flat evidence list by instance id
            inst_map: Dict[str, List[Dict]] = {}
            for ev in (evidence_list or []):
                inst_id = ev.get("evidence_instance_id") or ev.get("instance_id") or ""
                bboxes_raw = ev.get("bboxes_in_range") or ev.get("bboxes_in_time_range") or {}
                normalized_bboxes = {normalize_time_key(k): v for k, v in bboxes_raw.items()}
                inst_map.setdefault(inst_id, []).append({
                    "evidence_start_time": ev.get("evidence_start_time", "00:00"),
                    "evidence_end_time": ev.get("evidence_end_time", "00:00"),
                    "evidence_rationale": ev.get("evidence_rationale", ""),
                    "bboxes_in_time_range": normalized_bboxes,
                })

            instances = [{"instance": inst_id, "evidences": evs}
                         for inst_id, evs in inst_map.items()]

            qa = {
                "video_id": video_id,
                "qa_id": qa_id,
                "question": row.get("question", ""),
                "options": row.get("options", {}),
                "answer_choice": row.get("answer", ""),
                "instances": instances,
            }
            if video_id:
                unique_videos.add(video_id)
            gt_map[(video_id, qa_id)] = qa
            ordered_keys.append((video_id, qa_id))

    vprint(f"[INFO] GT loaded: {len(ordered_keys)} QAs from {len(unique_videos)} videos.")
    return gt_map, ordered_keys, unique_videos


def is_bbox_map(obj: dict) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if not re.fullmatch(r"\d{2}:\d{2}", k):
            return False
        if not isinstance(v, str):
            return False
        if not re.fullmatch(r"\[\d+,\s*\d+,\s*\d+,\s*\d+\]", v):
            return False
    return True


def extract_top_level_json(predict_text: str):
    """Extract the top-level prediction JSON from raw model output text.

    Steps: strip </think> chain-of-thought, then scan for balanced {} blocks,
    preferring any candidate that contains 'instances' or 'answer_choice'.
    """
    if not predict_text:
        return None
    if "</think>" in predict_text:
        predict_text = predict_text.split("</think>", 1)[1]
    predict_text = predict_text.strip()

    candidates = []
    start = None
    depth = 0
    for i, c in enumerate(predict_text):
        if c == '{':
            if start is None:
                start = i
            depth += 1
        elif c == '}':
            if start is not None:
                depth -= 1
                if depth == 0:
                    frag = predict_text[start:i + 1]
                    try:
                        obj = json.loads(frag)
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except Exception:
                        try:
                            obj = ast.literal_eval(frag)
                            if isinstance(obj, dict):
                                candidates.append(obj)
                        except Exception:
                            pass
                    start = None
    if not candidates:
        vprint("[DEBUG] Could not extract a JSON object from the prediction text. predict_text:", predict_text)
        return None
    for obj in candidates:
        if "instances" in obj or "answer_choice" in obj:
            return obj
    return candidates[0]


def normalize_prediction_obj(raw_obj: dict) -> dict:
    if is_bbox_map(raw_obj):
        return {"answer_choice": None, "instances": []}

    def to_mmss(t: str) -> str:
        if t is None:
            return ""
        t = str(t).strip()
        if re.fullmatch(r"\d{2}:\d{2}", t):
            return t
        if ":" in t:
            parts = t.split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                total = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return f"{total // 60:02d}:{total % 60:02d}"
        # Pure numeric or float string: treat as bare seconds
        if re.fullmatch(r'\d+(\.\d+)?', t):
            return f"00:{int(float(t)):02d}"
        try:
            sec = int(float(t))
            return f"00:{sec:02d}"
        except Exception:
            return t

    answer_choice = raw_obj.get("answer_choice")
    if isinstance(answer_choice, str):
        ac = answer_choice.strip().upper()
        answer_choice = ac if (len(ac) == 1 and "A" <= ac <= "Z") else None
    else:
        answer_choice = None

    norm_insts = []
    insts_in = raw_obj.get("instances", [])
    if isinstance(insts_in, list):
        for inst in insts_in:
            if not isinstance(inst, dict):
                continue
            canonical = inst.get("instance_name") or inst.get("instance") or inst.get("instance_id") or ""
            norm_evidences = []
            for ev in (inst.get("evidences", []) or []):
                if not isinstance(ev, dict):
                    continue
                bbt = ev.get("bboxes_in_time_range", {})
                norm_bbt = {to_mmss(tk): box for tk, box in bbt.items()} if isinstance(bbt, dict) else {}
                norm_evidences.append({
                    "evidence_start_time": to_mmss(ev.get("evidence_start_time")),
                    "evidence_end_time": to_mmss(ev.get("evidence_end_time")),
                    "evidence_rationale": ev.get("evidence_rationale", ""),
                    "bboxes_in_time_range": norm_bbt
                })
            norm_insts.append({
                "instance": canonical,
                "instance_name": canonical,
                "evidences": norm_evidences
            })

    return {
        "answer_choice": answer_choice,
        "instances": norm_insts
    }


def build_resolution_map_from_videos(video_dir: str,
                                      video_ids: List[str]) -> Dict[str, Tuple[int, int]]:
    """Read actual (W, H) from video files using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for --bbox-format with n1000 scaling. "
                          "Install with: pip install opencv-python-headless")
    res_map: Dict[str, Tuple[int, int]] = {}
    missing = []
    for vid in video_ids:
        path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.isfile(path):
            missing.append(vid)
            continue
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            res_map[vid] = (w, h)
        else:
            missing.append(vid)
    if missing:
        print(f"[WARN] Could not read resolution for {len(missing)} videos: {missing[:5]}{'...' if len(missing)>5 else ''}")
    vprint(f"[INFO] Resolution map built from videos: {len(res_map)} entries.")
    return res_map


def apply_bbox_format(instances: List[Dict[str, Any]],
                      fmt: str,
                      video_id: str,
                      res_map: Dict[str, Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Transform pred bbox coordinates to match GT pixel xyxy convention.

    fmt options:
      xyxy_px    — no-op (already pixel xyxy)
      yxyx_px    — swap [y1,x1,y2,x2] → [x1,y1,x2,y2], keep pixel scale
      xyxy_n1000 — scale from 0-1000 to pixels, keep xyxy order
      yxyx_n1000 — swap yxyx→xyxy, then scale from 0-1000 to pixels
    """
    if fmt == "xyxy_px":
        return instances
    W, H = res_map.get(video_id, (1920, 1080))
    result = []
    for inst in (instances or []):
        new_evs = []
        for ev in (inst.get("evidences", []) or []):
            new_bbt: Dict[str, Any] = {}
            for ts, box in (ev.get("bboxes_in_time_range", {}) or {}).items():
                if isinstance(box, str):
                    parsed = parse_box(box)
                    if parsed is None:
                        new_bbt[ts] = box
                        continue
                    box = list(parsed)
                if not (isinstance(box, (list, tuple)) and len(box) == 4):
                    new_bbt[ts] = box
                    continue
                a, b, c, d = [float(x) for x in box]
                if "yxyx" in fmt:
                    a, b, c, d = b, a, d, c   # [y1,x1,y2,x2] → [x1,y1,x2,y2]
                if "n1000" in fmt:
                    a = round(a * W / 1000)
                    b = round(b * H / 1000)
                    c = round(c * W / 1000)
                    d = round(d * H / 1000)
                new_bbt[ts] = [int(a), int(b), int(c), int(d)]
            new_evs.append({**ev, "bboxes_in_time_range": new_bbt})
        result.append({**inst, "evidences": new_evs})
    return result


def _extract_question_from_prompt(prompt: str) -> str:
    """Extract the question text from a prompt (between 'Question:' and 'Options:')."""
    m = re.search(r'Question:\s*(.*?)\s*\nOptions:', prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_predictions_jsonl(path: str,
                           ordered_keys: List[Tuple[str, str]],
                           gt_map: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load a predictions .jsonl file (one prediction per line). Each line has a
    "predict" field containing the model's JSON response and a "prompt" field.

    Predictions are joined to GT QAs by question text (extracted from the prompt and
    matched against the GT question field), so the prediction file order need not match
    the GT order. Falls back to positional alignment when a question can't be matched."""
    # Build question → (video_id, qa_id) lookup from GT
    question_to_key: Dict[str, Tuple[str, str]] = {}
    for key, qa in gt_map.items():
        q = (qa.get("question") or "").strip()
        if q:
            question_to_key[q] = key

    pred_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    skipped = 0
    matched_by_question = 0
    matched_by_position = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                outer = json.loads(line)
            except Exception:
                skipped += 1
                continue
            predict_text = outer.get("predict") or ""
            embedded = extract_top_level_json(predict_text)
            if not embedded or not isinstance(embedded, dict) or is_bbox_map(embedded):
                core = {"answer_choice": None, "instances": []}
            else:
                core = normalize_prediction_obj(embedded)

            # Join by question text first; fall back to positional alignment.
            question = _extract_question_from_prompt(outer.get("prompt") or "")
            key = question_to_key.get(question) if question else None
            if key is not None:
                matched_by_question += 1
            elif i < len(ordered_keys):
                key = ordered_keys[i]
                matched_by_position += 1
            else:
                skipped += 1
                continue

            video_id, qa_id = key
            pred_map[key] = {
                "video_id": video_id,
                "qa_id": qa_id,
                "answer_choice": core.get("answer_choice"),
                "instances": core.get("instances", []),
            }

    vprint(f"[INFO] Predictions parsed: {len(pred_map)} "
          f"(matched by question {matched_by_question}, by position {matched_by_position}); skipped {skipped} lines.")
    return pred_map


def aggregate_overall_per_qa(qa_results: List[QAResult], debug_print: bool = False) -> OverallPerQA:
    def avg(seq): return sum(seq) / len(seq) if seq else 0.0

    if debug_print:
        # BBox size analysis
        bbox_threshold = 10000  # px²
        large_bbox_qas = [r for r in qa_results if r.avg_gt_bbox_area >= bbox_threshold]
        small_bbox_qas = [r for r in qa_results if r.avg_gt_bbox_area < bbox_threshold]

        print(f"\n{'='*80}")
        print(f"[BBox size analysis] Grouped by average GT bbox area (threshold: {bbox_threshold:.0f} px²):")
        print(f"  Large bbox (>={bbox_threshold:.0f}px²): {len(large_bbox_qas)} QAs ({len(large_bbox_qas)/len(qa_results)*100:.1f}%)")
        print(f"  Small bbox (<{bbox_threshold:.0f}px²): {len(small_bbox_qas)} QAs ({len(small_bbox_qas)/len(qa_results)*100:.1f}%)")
        if large_bbox_qas:
            print(f"  Large-bbox group QA Accuracy: {avg([r.six_metrics.qa_accuracy for r in large_bbox_qas])*100:.2f}%")
        if small_bbox_qas:
            print(f"  Small-bbox group QA Accuracy: {avg([r.six_metrics.qa_accuracy for r in small_bbox_qas])*100:.2f}%")
        if large_bbox_qas and small_bbox_qas:
            diff = avg([r.six_metrics.qa_accuracy for r in large_bbox_qas]) - avg([r.six_metrics.qa_accuracy for r in small_bbox_qas])
            print(f"  Difference: {diff*100:+.2f}% (large bbox - small bbox)")
        print(f"{'='*80}\n")

    if debug_print:
        all_m_viou_pergt = [r.spatiotemporal.m_viou_pergt for r in qa_results]
        final_m_viou_pergt = avg(all_m_viou_pergt)
        print(f"\n{'='*80}")
        print(f"[Final summary] m_vIoU(per-GT) computation:")
        print(f"  Evaluated {len(qa_results)} QAs in total")
        print(f"  First 10 QAs' m_vIoU(per-GT): {[f'{v*100:.2f}%' for v in all_m_viou_pergt[:10]]}")
        print(f"  Stats: min={min(all_m_viou_pergt)*100:.2f}%, max={max(all_m_viou_pergt)*100:.2f}%, mean={final_m_viou_pergt*100:.2f}%")
        print(f"  Non-zero count: {sum(1 for v in all_m_viou_pergt if v > 0)}/{len(all_m_viou_pergt)}")
        print(f"  Final result = {sum(all_m_viou_pergt):.6f} / {len(qa_results)} = {final_m_viou_pergt:.6f}")
        print(f"{'='*80}\n")

    return OverallPerQA(
        n_qas=len(qa_results),
        qa_temporal_r1_tau=avg([r.temporal.r1_tau for r in qa_results]),
        qa_temporal_m_tiou_tponly=avg([r.temporal.m_tiou_tponly for r in qa_results]),
        qa_temporal_m_tiou_pergt=avg([r.temporal.m_tiou_pergt for r in qa_results]),
        qa_st_r1_tau_st=avg([r.spatiotemporal.r1_tau_st for r in qa_results]),
        qa_st_m_viou_tponly=avg([r.spatiotemporal.m_viou_tponly for r in qa_results]),
        qa_st_m_viou_pergt=avg([r.spatiotemporal.m_viou_pergt for r in qa_results]),
        qa_accuracy=avg([r.six_metrics.qa_accuracy for r in qa_results]),
        qa_faithful_answer_accuracy=avg([r.six_metrics.faithful_answer_accuracy for r in qa_results]),
        qa_spurious_answer_rate=avg([r.six_metrics.spurious_answer_rate for r in qa_results]),
    )


def evaluate_one_qa(pred_json: Dict[str, Any],
                    gt_qa_json: Dict[str, Any],
                    eps_overlap: int,
                    tau_t: float,
                    tau_st: float,
                    use_coverage_aware_score: bool) -> QAResult:
    scorer = score_coverage_aware_viou if use_coverage_aware_score else score_modified_viou
    pred_insts = build_timelines_from_schema(pred_json.get("instances", []))
    gt_insts = build_timelines_from_schema(gt_qa_json.get("instances", []))
    matches = greedy_match(pred_insts, gt_insts, scorer, eps_overlap)
    tp = len(matches)
    fp = max(0, len(pred_insts) - tp)
    fn = max(0, len(gt_insts) - tp)
    t_metrics = compute_temporal_metrics(matches, pred_insts, gt_insts, tau_t)
    st_metrics = compute_spatiotemporal_metrics(matches, pred_insts, gt_insts, tau_st)

    gt_choice = gt_qa_json.get("answer_choice")
    pred_choice = pred_json.get("answer_choice")
    qa_acc = 1.0 if (pred_choice is not None and gt_choice is not None and str(pred_choice) == str(gt_choice)) else 0.0
    tga = float(t_metrics.r1_tau)
    stga = float(st_metrics.r1_tau_st)
    # faithful: answer correct AND at least one matched pred-gt pair has vIoU >= tau_st
    # spurious: answer correct AND no matched pair reaches vIoU >= tau_st
    # Both use the same any_hit_st check so they are complementary.
    any_hit_st = any(
        viou_star(pred_insts[m.pred_idx], gt_insts[m.gt_idx]) >= tau_st
        for m in matches
    ) if matches else False
    faithful_B = 1.0 if (qa_acc == 1.0 and any_hit_st) else 0.0
    spurious = 1.0 if (qa_acc == 1.0 and not any_hit_st) else 0.0

    qa6 = QA6Metrics(
        qa_accuracy=qa_acc,
        temporal_grounding_accuracy=tga,
        spatiotemporal_grounding_accuracy=stga,
        faithful_answer_accuracy=faithful_B,
        spurious_answer_rate=spurious,
    )
    video_id = gt_qa_json.get("video_id", pred_json.get("video_id", ""))
    qa_id = gt_qa_json.get("qa_id", pred_json.get("qa_id", ""))
    avg_bbox_area = compute_avg_gt_bbox_area(gt_qa_json)
    return QAResult(video_id=video_id, qa_id=qa_id,
                    temporal=TemporalMetrics(**asdict(t_metrics)),
                    spatiotemporal=SpatioTemporalMetrics(**asdict(st_metrics)),
                    tp=tp, fp=fp, fn=fn,
                    six_metrics=qa6,
                    avg_gt_bbox_area=avg_bbox_area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True,
                    help="Path to the GT castbench_hf.jsonl file (HuggingFace release format).")
    ap.add_argument("--pred", required=True,
                    help="Path to the predictions .jsonl file (one prediction per line).")
    ap.add_argument("--model_name", type=str, default="model",
                    help="Model name shown in the summary output.")
    ap.add_argument("--eps_overlap", type=int, default=1)
    ap.add_argument("--tau_t", type=float, default=0.5)
    ap.add_argument("--tau_st", type=float, default=0.3)
    ap.add_argument("--use_coverage_aware_score", type=str, default="false")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--ignore_missing_preds", type=str, default="false")
    ap.add_argument("--start_index", type=int, default=None,
                    help="Only evaluate QAs from start_index onward (0-based).")
    ap.add_argument("--end_index", type=int, default=None,
                    help="Only evaluate QAs up to end_index (exclusive).")
    ap.add_argument("--debug_print", action="store_true",
                    help="Print the verbose per-metric breakdown instead of just the JSON summary.")
    ap.add_argument("--bbox-format", type=str, default="xyxy_px",
                    choices=["xyxy_px", "yxyx_px", "xyxy_n1000", "yxyx_n1000"],
                    help="Pred bbox coordinate convention: xyxy_px (default), yxyx_px, xyxy_n1000, yxyx_n1000.")
    ap.add_argument("--video-dir", type=str, default=None,
                    help="Directory containing <video_id>.mp4 files. Required when --bbox-format "
                         "includes n1000 scaling (xyxy_n1000 or yxyx_n1000).")

    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.debug_print

    if not os.path.isfile(args.pred):
        raise ValueError("--pred must be a predictions .jsonl file")

    use_cov = args.use_coverage_aware_score.lower() in {"1", "true", "yes", "y"}
    ignore_missing = args.ignore_missing_preds.lower() in {"1", "true", "yes", "y"}

    # ── Load GT ──────────────────────────────────────────────────────────────────
    gt_map, ordered_keys, unique_videos = load_gt_hf_jsonl(args.gt)
    for qa in gt_map.values():
        normalize_gt_qa(qa)

    # ── Load Predictions ─────────────────────────────────────────────────────────
    pred_map = load_predictions_jsonl(args.pred, ordered_keys, gt_map)

    # Apply index slicing
    start_idx = None
    end_idx = None
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(ordered_keys)
        start_idx = max(0, min(start_idx, len(ordered_keys)))
        end_idx = max(start_idx, min(end_idx, len(ordered_keys)))
        ordered_keys_to_evaluate = ordered_keys[start_idx:end_idx]
        vprint(f"[INFO] Evaluating QAs from index {start_idx} to {end_idx-1} (total {len(ordered_keys_to_evaluate)} QAs)")
    else:
        ordered_keys_to_evaluate = ordered_keys
        vprint(f"[INFO] Evaluating all {len(ordered_keys_to_evaluate)} QAs")

    bbox_fmt = args.bbox_format
    res_map: Dict[str, Tuple[int, int]] = {}
    if bbox_fmt != "xyxy_px":
        needs_scale = "n1000" in bbox_fmt
        if needs_scale:
            if not args.video_dir:
                raise ValueError(
                    f"--bbox-format {bbox_fmt} requires --video-dir pointing to the .mp4 files.")
            res_map = build_resolution_map_from_videos(args.video_dir, list(unique_videos))
        vprint(f"[INFO] Bbox format: {bbox_fmt}  (res_map covers {len(res_map)} videos)")

    qa_results: List[QAResult] = []
    missing_pred = 0
    for (vid, qid) in tqdm(ordered_keys_to_evaluate, desc="Evaluating QAs", disable=not VERBOSE):
        gtqa = gt_map.get((vid, qid))
        if gtqa is None:
            continue
        pred = pred_map.get((vid, qid))
        if pred is None:
            for (pvid, pqid), pobj in pred_map.items():
                if pqid == qid:
                    pred = pobj
                    break
        if pred is None:
            missing_pred += 1
            if ignore_missing:
                continue
            pred = {"video_id": vid, "qa_id": qid, "instances": []}
        if bbox_fmt != "xyxy_px":
            transformed = apply_bbox_format(pred.get("instances", []), bbox_fmt, vid, res_map)
            pred = {**pred, "instances": transformed}
        res = evaluate_one_qa(pred, gtqa,
                              eps_overlap=args.eps_overlap,
                              tau_t=args.tau_t,
                              tau_st=args.tau_st,
                              use_coverage_aware_score=use_cov)
        qa_results.append(res)

    overall = aggregate_overall_per_qa(qa_results, debug_print=args.debug_print)
    os.makedirs(args.out_dir, exist_ok=True)

    suffix = f"_{start_idx}_{end_idx}" if (start_idx is not None and end_idx is not None) else ""
    perqa_path = os.path.join(args.out_dir, f"results_per_qa{suffix}.json")
    overall_path = os.path.join(args.out_dir, f"results_overall{suffix}.json")

    with open(perqa_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "video_id": r.video_id,
                "qa_id": r.qa_id,
                "temporal": asdict(r.temporal),
                "spatiotemporal": asdict(r.spatiotemporal),
                "tp": r.tp, "fp": r.fp, "fn": r.fn,
                "qa6": asdict(r.six_metrics)
            } for r in qa_results
        ], f, ensure_ascii=False, indent=2)

    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_per_qa": asdict(overall),
            "missing_pred_files": missing_pred,
            "config": {
                "eps_overlap": args.eps_overlap,
                "tau_t": args.tau_t,
                "tau_st": args.tau_st,
                "use_coverage_aware_score": use_cov,
                "pred_path": args.pred
            }
        }, f, ensure_ascii=False, indent=2)

    if ignore_missing:
        vprint(f"[INFO] Ignored {missing_pred} missing preds (not included in averages).")
    elif missing_pred:
        vprint(f"[WARN] Missing prediction files for {missing_pred} QA(s); treated as empty preds (score=0).")

    summary = {
        "model_name": args.model_name,
        "qa_accuracy": overall.qa_accuracy,
        "qa_faithful": overall.qa_faithful_answer_accuracy,
        "qa_temporal_iou": overall.qa_temporal_m_tiou_pergt,
        "qa_spatiotemporal_iou": overall.qa_st_m_viou_pergt,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
