from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import re

# =========================
# Utilities: time & geometry
# =========================

def to_seconds(mmss):
    """
    Robust mm:ss -> seconds.
    Accepts:
      mm:ss
      m:s
      ss
      extra colons/text: takes the first groups of digits
      a single group of digits: treated as seconds
    Falls back to 0 on failure.
    """
    if mmss is None:
        return 0
    s = str(mmss).strip()
    if not s:
        return 0
    # Pure numeric (possibly float): treat as bare seconds
    if re.fullmatch(r'\d+(\.\d+)?', s):
        return int(float(s))
    # Extract groups of digits (colon-separated time format)
    nums = re.findall(r'\d+', s)
    if len(nums) >= 3:
        # HH:MM:SS → total seconds
        return int(nums[0]) * 3600 + int(nums[1]) * 60 + int(nums[2])
    if len(nums) >= 2:
        # MM:SS
        return int(nums[0]) * 60 + int(nums[1])
    if len(nums) == 1:
        return int(nums[0])
    return 0

def parse_box(s) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(s, (list, tuple)) and len(s) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in s]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    parts = s[1:-1].split(",")
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(p.strip()) for p in parts]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def iou2d(a: Optional[Tuple[float, float, float, float]],
          b: Optional[Tuple[float, float, float, float]]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = area(a) + area(b) - inter
    return inter / ua if ua > 0 else 0.0

def inter_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    (s1, e1), (s2, e2) = a, b
    s, e = max(s1, s2), min(e1, e2)
    return max(0, e - s + 1)

def union_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    (s1, e1), (s2, e2) = a, b
    return (max(e1, e2) - min(s1, s2) + 1)

def seconds_range_inclusive(s: int, e: int) -> List[int]:
    if e < s:
        return []
    return list(range(s, e + 1))

# =========================
# Instance timeline builders
# =========================

@dataclass
class InstanceTimeline:
    name: str
    start: int
    end: int
    boxes: Dict[int, Optional[Tuple[float, float, float, float]]]

def build_instance_timeline(instance_obj: Dict[str, Any]) -> InstanceTimeline:
    """
    Merge multiple evidences of an instance into a single timeline:
    - time range = [min start, max end], inclusive seconds
    - boxes[t] = bbox from the evidence covering t; if multiple, choose larger-area (deterministic)
    - if no evidence covers t, boxes[t] = None
    """
    name = instance_obj.get("instance", "")
    evs = instance_obj.get("evidences", []) or []
    if not evs:
        return InstanceTimeline(name=name, start=0, end=-1, boxes={})

    starts, ends = [], []
    per_second_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for ev in evs:
        s = to_seconds(ev["evidence_start_time"])
        e = to_seconds(ev["evidence_end_time"])
        starts.append(s); ends.append(e)
        bmap: Dict[str, str] = ev.get("bboxes_in_time_range", {}) or {}
        for k, v in bmap.items():
            t = to_seconds(k)
            if s <= t <= e:
                bb = parse_box(v)
                if bb is None:
                    continue
                per_second_boxes.setdefault(t, []).append(bb)

    S, E = min(starts), max(ends)
    boxes: Dict[int, Optional[Tuple[float, float, float, float]]] = {}
    for t in seconds_range_inclusive(S, E):
        if t not in per_second_boxes:
            boxes[t] = None
        else:
            cand = per_second_boxes[t]
            best = max(cand, key=area)
            boxes[t] = best

    return InstanceTimeline(name=name, start=S, end=E, boxes=boxes)

def build_timelines_from_schema(instances_list: List[Dict[str, Any]]) -> List[InstanceTimeline]:
    return [build_instance_timeline(x) for x in (instances_list or [])]

# =========================
# Overlap & scores
# =========================

def overlap_seconds(pi: InstanceTimeline, gj: InstanceTimeline) -> List[int]:
    s = max(pi.start, gj.start)
    e = min(pi.end, gj.end)
    return seconds_range_inclusive(s, e) if e >= s else []

def mean_siou_over_overlap(pi: InstanceTimeline, gj: InstanceTimeline,
                           Tov: List[int]) -> float:
    if not Tov:
        return 0.0
    total = 0.0
    for t in Tov:
        total += iou2d(pi.boxes.get(t), gj.boxes.get(t))
    return total / len(Tov)

def tiou_over_union(pi: InstanceTimeline, gj: InstanceTimeline) -> float:
    inter = inter_len((pi.start, pi.end), (gj.start, gj.end))
    uni = union_len((pi.start, pi.end), (gj.start, gj.end))
    return inter / uni if uni > 0 else 0.0

def score_modified_viou(pi: InstanceTimeline, gj: InstanceTimeline,
                        eps_overlap: int) -> Tuple[float, int]:
    """Only mean 2D IoU on overlapping seconds; returns (score, |Tov|)."""
    Tov = overlap_seconds(pi, gj)
    if len(Tov) < eps_overlap:
        return (0.0, 0)
    return (mean_siou_over_overlap(pi, gj, Tov), len(Tov))

def score_coverage_aware_viou(pi: InstanceTimeline, gj: InstanceTimeline,
                              eps_overlap: int) -> Tuple[float, int]:
    """Coverage-aware vIoU = (overlap/union) * mean_sIoU(overlap)."""
    Tov = overlap_seconds(pi, gj)
    if len(Tov) < eps_overlap:
        return (0.0, 0)
    cov = tiou_over_union(pi, gj)
    ms = mean_siou_over_overlap(pi, gj, Tov)
    return (cov * ms, len(Tov))

def tiou(pi: InstanceTimeline, gj: InstanceTimeline) -> float:
    inter = inter_len((pi.start, pi.end), (gj.start, gj.end))
    uni = union_len((pi.start, pi.end), (gj.start, gj.end))
    return inter / uni if uni > 0 else 0.0

def viou_star(pi: InstanceTimeline, gj: InstanceTimeline) -> float:
    """Coverage-aware vIoU used in ST metrics."""
    Tov = overlap_seconds(pi, gj)
    if not Tov:
        return 0.0
    cov = tiou_over_union(pi, gj)
    ms = mean_siou_over_overlap(pi, gj, Tov)
    return cov * ms

# =========================
# Greedy 1:1 matching
# =========================

@dataclass
class MatchPair:
    pred_idx: int
    gt_idx: int
    score: float
    tov_len: int

def greedy_match(preds: List[InstanceTimeline],
                 gts: List[InstanceTimeline],
                 scorer,
                 eps_overlap: int) -> List[MatchPair]:
    candidates: List[MatchPair] = []
    for i, pi in enumerate(preds):
        for j, gj in enumerate(gts):
            score, tov = scorer(pi, gj, eps_overlap)
            if score > 0.0 and tov > 0:
                candidates.append(MatchPair(i, j, score, tov))

    candidates.sort(key=lambda x: (-x.score, -x.tov_len, x.pred_idx, x.gt_idx))
    used_pred, used_gt = set(), set()
    matches: List[MatchPair] = []
    for c in candidates:
        if c.pred_idx in used_pred or c.gt_idx in used_gt:
            continue
        matches.append(c)
        used_pred.add(c.pred_idx)
        used_gt.add(c.gt_idx)
    return matches

# =========================
# Metrics
# =========================

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

def compute_temporal_metrics(matches: List[MatchPair],
                             preds: List[InstanceTimeline],
                             gts: List[InstanceTimeline],
                             tau_t: float) -> TemporalMetrics:
    tiou_vals: Dict[Tuple[int,int], float] = {}
    for m in matches:
        tiou_vals[(m.pred_idx, m.gt_idx)] = tiou(preds[m.pred_idx], gts[m.gt_idx])

    hits = 0
    for j in range(len(gts)):
        ok = any(m.gt_idx == j and tiou_vals[(m.pred_idx,m.gt_idx)] >= tau_t for m in matches)
        if ok: hits += 1
    r1 = hits / max(1, len(gts))

    mean_tp = (sum(tiou_vals.values()) / len(tiou_vals)) if tiou_vals else 0.0
    mean_pergt = (sum(tiou_vals.values()) / max(1, len(gts)))
    return TemporalMetrics(r1_tau=r1, m_tiou_tponly=mean_tp, m_tiou_pergt=mean_pergt)

def compute_spatiotemporal_metrics(matches: List[MatchPair],
                                   preds: List[InstanceTimeline],
                                   gts: List[InstanceTimeline],
                                   tau_st: float) -> SpatioTemporalMetrics:
    viou_vals: Dict[Tuple[int,int], float] = {}
    for m in matches:
        viou_vals[(m.pred_idx, m.gt_idx)] = viou_star(preds[m.pred_idx], gts[m.gt_idx])

    hits = 0
    for j in range(len(gts)):
        ok = any(m.gt_idx == j and viou_vals[(m.pred_idx,m.gt_idx)] >= tau_st for m in matches)
        if ok: 
            hits += 1
    r1 = hits / max(1, len(gts))

    mean_tp = (sum(viou_vals.values()) / len(viou_vals)) if viou_vals else 0.0
    mean_pergt = (sum(viou_vals.values()) / max(1, len(gts)))
    
    return SpatioTemporalMetrics(r1_tau_st=r1, m_viou_tponly=mean_tp, m_viou_pergt=mean_pergt)
