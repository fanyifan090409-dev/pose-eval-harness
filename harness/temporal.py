#!/usr/bin/env python3
"""temporal.py — T5 M3 引擎：PoseTrack 轨迹段 + 抖动三指标 + OneEuro 前后对比（G1 §3）。

数据事实（2026-10-01 实证，改动前必读）：
- 推断集 = build_posetrack.py 产出的硬链接场，image_id = int(vid)*10000+frame 全局唯一
  （ultralytics save_json 取数值化 stem），GT 合并 JSON 同规则对齐
- GT 标注帧结构 = 步长 4 稀疏区 + 中间步长 1 密集窗口 → 帧间隔 >1 必须断段
  （只有步长 1 连续帧能产出 ≥10 帧轨迹段；GT 段实测 993 条、中位长 31）
- GT v∈{0,1}：v=0 即 (0,0) 占位（34.4% 关键点，无坐标）→ 有效性掩码必须 GT v>0
  且 pred 置信度 >0（preds 第三位是 conf）——骨架注释「v 全 1 不过滤」已被实证推翻
- 17 关键点直接按 PoseTrack categories 索引报告（0-4 头部点语义与 COCO 略异，
  与抖动指标无关，论文注明）

口径（G1 §3）：
- 匹配：单档 t=0.5 全关键点 OKS 贪心（与 M1/M2 同一 match_greedy 引擎，
  G1「匈牙利」为泛指）；dets 按 score 降序截 top-20
- 轨迹段：同 (vid, track_id) 按帧号连续（间隔 >1 断段），长度 ≥10 才统计
- 归一化：s_t = sqrt(GT bbox w·h)（person-scale）
- 三指标（每段每关键点，像素域）：
  d_t = ‖p_t − p_{t−1}‖₂ / s_t（帧间位移；均值 + P95）
  j_t = ‖p_t − ŝ_t‖₂ / s_t，ŝ = 因果 5 帧中值滤波（抖动；均值 + P95）
  跳变率 = P(d_t > τ)，τ ∈ {0.05, 0.10, 0.20}（默认 0.10，G1 §3.2）
- OneEuro（Casiez 默认参数）施加在归一化坐标 q_t = p_t / s_t 上，逐段逐关键点
  因果滤波，段内有效性断档处重置状态；平滑后三指标在像素域重算（p̃_t = q̃_t·s_t，
  公式与原始域完全一致）
- smoothed_px 对照口径（2026-10-01 实证新增）：OneEuro 直接施于像素坐标。
  实证：q 域反归一化会把 GT bbox 逐帧尺度噪声（|s_t/s_{t-1}-1| median 0.97%、
  P95 8.1%）重新注入像素域，抖动指标不降反升；像素域滤波则全面下降
- 平滑对比口径：轨迹段固定由原始匹配决定（不重匹配，隔离平滑本身的效应），
  只替换成段 det 的关键点坐标，未匹配 det 原样
- 聚合：均值类指标 = 各段值再取均值（G1「每条轨迹计算后取均值」）；P95/跳变率 =
  全段逐对值池化
- 平滑精度代价：对替换后的 preds 重跑 evaluate_m1 全量 OKS AP

输出：metrics.json 的 m3 段。

用法：python harness/temporal.py --run runs/<run_id>
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.oneeuro import one_euro_filter
from harness.stratified_map import MAX_DETS, compute_oks, evaluate_m1, load_gt_coco, match_greedy

K = 17
MATCH_T = 0.5        # 单档匹配阈值（与 M2 同口径）
MIN_SEG_LEN = 10     # G1 §3.1：只统计连续 ≥10 帧的轨迹段
TAUS = (0.05, 0.10, 0.20)   # 跳变率阈值：默认 0.10 + 敏感性
MEDIAN_WIN = 5       # 因果 5 帧中值滤波
ONEEURO = dict(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
MEAN_METRICS = ("disp_mean", "jitter_mean")


def causal_median(seq: np.ndarray, win: int = MEDIAN_WIN) -> np.ndarray:
    """因果中值滤波：第 t 帧取 seq[max(0,t-win+1):t+1] 逐坐标中位数。"""
    seq = np.asarray(seq, dtype=np.float64)
    out = np.empty_like(seq)
    for t in range(len(seq)):
        out[t] = np.median(seq[max(0, t - win + 1):t + 1], axis=0)
    return out


def disp_jitter(positions: np.ndarray, scales: np.ndarray):
    """一段有效 run 的 (d, j)：d 用当前帧 s_t 归一化，j 对因果中值残差归一化。"""
    d = np.linalg.norm(np.diff(positions, axis=0), axis=1) / scales[1:]
    j = np.linalg.norm(positions - causal_median(positions), axis=1) / scales
    return d, j


def build_segments(gt_by_img, preds_by_img, im_meta):
    """逐标注帧匹配 → (vid, track_id) 轨迹 → 帧间隔 >1 断段 → 保留 ≥10 帧段。

    返回 list of list：每段是 [(frame, image_id, gt_ann, det), ...] 按帧升序。
    """
    by_vid_track = defaultdict(list)
    for image_id, anns in gt_by_img.items():
        vid, frame, labeled = im_meta[image_id]
        if not labeled:
            continue
        dets = preds_by_img.get(image_id, [])
        if not dets:
            continue
        dets = sorted(dets, key=lambda d: -d["score"])[:MAX_DETS]
        # 复刻 M1：gt 按 _ignore 升序，匹配后映射回原始索引
        order = sorted(range(len(anns)), key=lambda i: anns[i]["_ignore"])
        anns_sorted = [anns[i] for i in order]
        ious, _ = compute_oks(anns_sorted, dets)
        dtm = match_greedy(anns_sorted, ious, MATCH_T)
        for dind, gind_sorted in dtm.items():
            ann = anns[order[gind_sorted]]
            if ann["_ignore"]:
                continue  # dtIg 语义：匹配到 ignore GT 不计
            by_vid_track[(vid, ann["track_id"])].append(
                (frame, image_id, ann, dets[dind]))

    segments = []
    for entries in by_vid_track.values():
        entries.sort(key=lambda e: e[0])
        run = [entries[0]]
        for prev, cur in zip(entries, entries[1:]):
            if cur[0] == prev[0] + 1:
                run.append(cur)
            else:
                if len(run) >= MIN_SEG_LEN:
                    segments.append(run)
                run = [cur]
        if len(run) >= MIN_SEG_LEN:
            segments.append(run)
    return segments


def kp_runs(seg, k):
    """一段内单个关键点的有效连续 run。

    有效性 = GT v>0 且 pred conf>0 且坐标有限（v=0 是 (0,0) 占位）。
    返回 [(positions (T,2) 像素, scales (T,), dets 引用列表), ...]，断档处切分。
    """
    runs, cur_pos, cur_sc, cur_dets = [], [], [], []
    for frame, image_id, ann, det in seg:
        gv = ann["keypoints"][3 * k + 2]
        pv = det["keypoints"][3 * k + 2]
        x, y = det["keypoints"][3 * k], det["keypoints"][3 * k + 1]
        if gv > 0 and pv > 0 and np.isfinite(x) and np.isfinite(y):
            cur_pos.append([x, y])
            cur_sc.append(float(np.sqrt(ann["bbox"][2] * ann["bbox"][3])))
            cur_dets.append(det)
        elif cur_pos:
            runs.append((np.array(cur_pos, dtype=np.float64),
                         np.array(cur_sc, dtype=np.float64), cur_dets))
            cur_pos, cur_sc, cur_dets = [], [], []
    if cur_pos:
        runs.append((np.array(cur_pos, dtype=np.float64),
                     np.array(cur_sc, dtype=np.float64), cur_dets))
    return runs


def collect_metrics(segments):
    """对段集合算每关键点指标。

    返回 (per_kp, pool_d)：per_kp[k][metric] = 段级值列表；pool_d[k] = 逐对 d 池。
    """
    per_kp = {k: {m: [] for m in MEAN_METRICS} for k in range(K)}
    pool_d = {k: [] for k in range(K)}
    for seg in segments:
        for k in range(K):
            d_parts, j_parts = [], []
            for pos, sc, _ in kp_runs(seg, k):
                if len(pos) < 2:
                    continue
                d, j = disp_jitter(pos, sc)
                d_parts.append(d)
                j_parts.append(j)
            if not d_parts:
                continue
            d_cat = np.concatenate(d_parts)
            j_cat = np.concatenate(j_parts)
            per_kp[k]["disp_mean"].append(float(d_cat.mean()))
            per_kp[k]["jitter_mean"].append(float(j_cat.mean()))
            pool_d[k].append(d_cat)
            pool_d[k].append(j_cat)  # 抖动值同池存（idx 奇偶区分）
    return per_kp, pool_d


def summarize(per_kp, pool_d, kp_names):
    """聚合：均值类 = 段级值再取均值；P95/跳变率 = 全段池化。"""
    out_per_kp = []
    for k in range(K):
        ds = [x for i, x in enumerate(pool_d[k]) if i % 2 == 0]
        js = [x for i, x in enumerate(pool_d[k]) if i % 2 == 1]
        d_all = np.concatenate(ds) if ds else np.array([])
        j_all = np.concatenate(js) if js else np.array([])
        entry = {"id": k, "name": kp_names[k]}
        for m in MEAN_METRICS:
            vals = per_kp[k][m]
            entry[m] = round(float(np.mean(vals)), 6) if vals else None
        entry["disp_p95"] = round(float(np.percentile(d_all, 95)), 6) if len(d_all) else None
        entry["jitter_p95"] = round(float(np.percentile(j_all, 95)), 6) if len(j_all) else None
        for tau in TAUS:
            entry[f"jump_{tau}"] = round(float((d_all > tau).mean()), 6) if len(d_all) else None
        entry["n_pairs"] = int(len(d_all))
        entry["n_segments"] = len(per_kp[k]["disp_mean"])
        out_per_kp.append(entry)
    overall = {}
    for m in MEAN_METRICS:
        vals = [e[m] for e in out_per_kp if e[m] is not None]
        overall[m] = round(float(np.mean(vals)), 6) if vals else None
    for m in ("disp_p95", "jitter_p95") + tuple(f"jump_{t}" for t in TAUS):
        vals = [e[m] for e in out_per_kp if e[m] is not None]
        overall[m] = round(float(np.mean(vals)), 6) if vals else None
    return out_per_kp, overall


def smooth_replacements(segments):
    """OneEuro 平滑（q 域）→ {id(det): 替换后 51 值 keypoints}。

    每段每关键点：有效 run 内 q_t = p_t / s_t 逐 run 因果滤波（断档重置状态），
    反归一化 p̃_t = q̃_t·s_t 写回副本；未涉及的关键点保持原值。
    """
    repl = {}
    for seg in segments:
        for k in range(K):
            for pos, sc, dets in kp_runs(seg, k):
                q = pos / sc[:, None]
                qs = np.column_stack([
                    one_euro_filter(q[:, 0], **ONEEURO),
                    one_euro_filter(q[:, 1], **ONEEURO),
                ])
                for i, det in enumerate(dets):
                    kp = repl.setdefault(id(det), list(det["keypoints"]))
                    kp[3 * k] = float(qs[i, 0] * sc[i])
                    kp[3 * k + 1] = float(qs[i, 1] * sc[i])
    return repl


def pixel_replacements(segments):
    """对照口径：OneEuro 直接施于像素坐标（同样的段/run/断档重置规则）。"""
    repl = {}
    for seg in segments:
        for k in range(K):
            for pos, sc, dets in kp_runs(seg, k):
                ps = np.column_stack([
                    one_euro_filter(pos[:, 0], **ONEEURO),
                    one_euro_filter(pos[:, 1], **ONEEURO),
                ])
                for i, det in enumerate(dets):
                    kp = repl.setdefault(id(det), list(det["keypoints"]))
                    kp[3 * k] = float(ps[i, 0])
                    kp[3 * k + 1] = float(ps[i, 1])
    return repl


def apply_repl(segments, repl):
    """用替换映射重建段 det（坐标被替换、score 不变）。"""
    return [
        [(frame, image_id, ann,
          {"keypoints": repl.get(id(det), det["keypoints"]), "score": det["score"]})
         for frame, image_id, ann, det in seg]
        for seg in segments
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<run_id> 目录名")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    meta = json.load(open(run_dir / "meta.json"))
    datasets_cfg = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))
    ds_cfg = datasets_cfg["datasets"][meta["dataset"]]
    gt_json = ds_cfg.get("gt_json")
    if not gt_json:
        print(f"数据集 {meta['dataset']} 无 gt_json，M3 不可用", file=sys.stderr)
        sys.exit(1)

    data = json.load(open(gt_json))
    im_meta = {im["id"]: (im["vid_id"], im["frame"], im["is_labeled"])
               for im in data["images"]}
    kp_names = data["categories"][0]["keypoints"]

    gt_by_img = load_gt_coco(gt_json)
    preds = json.load(open(run_dir / "preds.json"))
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    segments = build_segments(gt_by_img, preds_by_img, im_meta)
    seg_lens = [len(s) for s in segments]
    n_videos = len({im_meta[s[0][1]][0] for s in segments})
    print(f"轨迹段: {len(segments)} 条（{n_videos} 视频），"
          f"长度 min/median/max = "
          f"{min(seg_lens) if seg_lens else 0}/"
          f"{int(np.median(seg_lens)) if seg_lens else 0}/"
          f"{max(seg_lens) if seg_lens else 0}")

    # 原始域指标（段固定由原始匹配决定）
    per_kp_raw, pool_raw = collect_metrics(segments)
    raw_kp, raw_overall = summarize(per_kp_raw, pool_raw, kp_names)

    # 平滑：同一批段、只替换成段 det 的坐标（不重匹配，隔离平滑效应）
    full_raw = evaluate_m1(gt_by_img, preds_by_img)[0]
    smooth_out = {}
    for domain, repl_fn in (("smoothed", smooth_replacements),
                            ("smoothed_px", pixel_replacements)):
        repl = repl_fn(segments)
        sm_segs = apply_repl(segments, repl)
        sm_kp, sm_overall = summarize(*collect_metrics(sm_segs), kp_names)
        # 平滑精度代价：全量 OKS AP（原始 preds vs 坐标替换后 preds）
        preds_sm = [{**p, "keypoints": repl.get(id(p), p["keypoints"])} for p in preds]
        preds_sm_by_img = {}
        for p in preds_sm:
            preds_sm_by_img.setdefault(p["image_id"], []).append(p)
        full_sm = evaluate_m1(gt_by_img, preds_sm_by_img)[0]
        map_delta = {
            "ap_raw": round(float(full_raw.mean()), 4),
            "ap_smoothed": round(float(full_sm.mean()), 4),
            "delta": round(float(full_sm.mean() - full_raw.mean()), 4),
        }
        smooth_out[domain] = {
            "per_keypoint": sm_kp,
            "overall": sm_overall,
            "map_delta": map_delta,
        }
        print(f"{domain} AP {map_delta['ap_raw']} → {map_delta['ap_smoothed']} "
              f"(Δ {map_delta['delta']})")

    m3 = {
        "protocol": {
            "match_t": MATCH_T,
            "min_seg_len": MIN_SEG_LEN,
            "taus": list(TAUS),
            "median_win": MEDIAN_WIN,
            "oneeuro": ONEEURO,
            "valid_mask": "GT v>0 且 pred conf>0",
            "break_rule": "帧间隔>1 断段",
            "normalization": "s_t = sqrt(GT bbox w*h)",
            "smooth_scope": "段由原始匹配固定，只替换成段 det 坐标（不重匹配）",
            "smooth_domains": {
                "smoothed": "q 域（G1 契约）：q_t=p_t/s_t 滤波后反归一化",
                "smoothed_px": "对照：OneEuro 直接施于像素坐标（q 域反归一化重注入 GT bbox 尺度噪声）",
            },
        },
        "segments": {
            "n": len(segments),
            "n_videos": n_videos,
            "len_min": min(seg_lens) if seg_lens else 0,
            "len_median": int(np.median(seg_lens)) if seg_lens else 0,
            "len_max": max(seg_lens) if seg_lens else 0,
        },
        "raw": {"per_keypoint": raw_kp, "overall": raw_overall},
        **smooth_out,
    }

    metrics_path = run_dir / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.is_file() else {}
    metrics["run_id"] = args.run
    metrics["m3"] = m3
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(m3, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
