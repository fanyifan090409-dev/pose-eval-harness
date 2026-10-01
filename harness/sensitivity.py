#!/usr/bin/env python3
"""sensitivity.py — 论文 A 评审意见 P1 落地：三项敏感性分析（2026-10-01）。

A. M3 幸存者偏差：重跑匹配层统计被排除的段/帧比例与匹配失败率
B. M2 交集域：两模型有效图像交集上重算角度 RMSE + Δ + 配对 bootstrap CI
C. OneEuro 参数网格：min_cutoff × beta × 2 域重算三指标 + AP 代价抽样

输出：runs/sensitivity.json（{analysis: {...}}，三个子命令可独立跑，结果合并写盘）

用法：
  python harness/sensitivity.py --which A --model round5-3
  python harness/sensitivity.py --which B --dataset coco-val
  python harness/sensitivity.py --which C --model round5-3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.oneeuro import one_euro_filter
from harness.stratified_map import MAX_DETS, compute_oks, evaluate_m1, load_gt_coco, match_greedy
from harness.temporal import (
    K, MIN_SEG_LEN, TAUS, build_segments, collect_metrics, kp_runs, summarize,
)

MODELS = ("round5-3", "yolo11s-pose")
M2_DATASETS = ("coco-val", "crowdpose-val", "ochuman-test")
GRID_CUTOFF = (0.5, 1.0, 2.0, 4.0)
GRID_BETA = (0.001, 0.007, 0.05)
BOOT_N = 10000
SEED = 42
OUT = ROOT / "runs" / "sensitivity.json"


def load_run(run_id):
    run_dir = ROOT / "runs" / run_id
    meta = json.load(open(run_dir / "meta.json"))
    datasets_cfg = yaml.safe_load(open(ROOT / "configs" / "datasets.yaml"))
    gt_json = datasets_cfg["datasets"][meta["dataset"]]["gt_json"]
    data = json.load(open(gt_json))
    im_meta = {im["id"]: (im["vid_id"], im["frame"], im["is_labeled"])
               for im in data["images"]}
    kp_names = data["categories"][0]["keypoints"]
    gt_by_img = load_gt_coco(gt_json)
    preds = json.load(open(run_dir / "preds.json"))
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)
    return im_meta, kp_names, gt_by_img, preds, preds_by_img


def analysis_A(model):
    """M3 幸存者偏差：标注帧覆盖率 + 匹配失败率 + 段保留比例。"""
    run_id = f"{model}__posetrack-val"
    im_meta, _, gt_by_img, _, preds_by_img = load_run(run_id)

    n_labeled = sum(1 for v in im_meta.values() if v[2])
    frames_with_dets = 0
    n_gt_inst = 0
    n_pairs = 0
    n_unmatched = 0
    by_vid_track = {}
    for image_id, anns in gt_by_img.items():
        vid, frame, labeled = im_meta[image_id]
        if not labeled:
            continue
        dets = preds_by_img.get(image_id, [])
        if dets:
            frames_with_dets += 1
        else:
            continue
        dets = sorted(dets, key=lambda d: -d["score"])[:MAX_DETS]
        order = sorted(range(len(anns)), key=lambda i: anns[i]["_ignore"])
        anns_sorted = [anns[i] for i in order]
        for ann in anns:
            if not ann["_ignore"]:
                n_gt_inst += 1
        ious, _ = compute_oks(anns_sorted, dets)
        dtm = match_greedy(anns_sorted, ious, 0.5)
        matched_gt = set()
        for dind, gind_sorted in dtm.items():
            ann = anns_sorted[gind_sorted]
            if ann["_ignore"]:
                continue
            matched_gt.add(id(ann))
            n_pairs += 1
            by_vid_track.setdefault((vid, ann["track_id"]), []).append(
                (frame, image_id, ann, dets[dind]))
        n_unmatched += sum(1 for ann in anns if not ann["_ignore"]
                           and id(ann) not in matched_gt)

    all_segs = []
    for entries in by_vid_track.values():
        entries.sort(key=lambda e: e[0])
        run = [entries[0]]
        for prev, cur in zip(entries, entries[1:]):
            if cur[0] == prev[0] + 1:
                run.append(cur)
            else:
                all_segs.append(run)
                run = [cur]
        all_segs.append(run)
    kept_segs = [s for s in all_segs if len(s) >= MIN_SEG_LEN]

    def frames(segs):
        return len({(e[1], e[0]) for s in segs for e in s})

    kept_frames = frames(kept_segs)
    dropped_frames = frames(all_segs) - kept_frames
    return {
        "n_labeled_frames": n_labeled,
        "n_frames_with_dets": frames_with_dets,
        "n_frames_no_dets": n_labeled - frames_with_dets,
        "n_gt_inst": n_gt_inst,
        "n_matched_pairs": n_pairs,
        "n_unmatched": n_unmatched,
        "unmatched_rate": round(n_unmatched / n_gt_inst, 4) if n_gt_inst else None,
        "n_segs_all": len(all_segs),
        "n_segs_kept": len(kept_segs),
        "n_segs_dropped": len(all_segs) - len(kept_segs),
        "frames_all": frames(all_segs),
        "frames_kept": kept_frames,
        "frames_dropped": dropped_frames,
        "kept_frac_of_labeled": round(kept_frames / n_labeled, 4) if n_labeled else None,
    }


def analysis_B(dataset):
    """M2 交集域：两模型有效图像交集上重算 RMSE + Δ + 配对 CI。"""
    out = {}
    for joint in ("left_elbow", "right_elbow", "left_knee", "right_knee"):
        for vbin in ("v2", "v1"):
            units = {}
            for m in MODELS:
                d = json.load(open(ROOT / "runs" / f"{m}__{dataset}" / "m2_units.json"))
                units[m] = d[joint][vbin]
            errs_r = np.asarray(units["round5-3"]["errs"])
            imgs_r = np.asarray(units["round5-3"]["imgs"])
            errs_y = np.asarray(units["yolo11s-pose"]["errs"])
            imgs_y = np.asarray(units["yolo11s-pose"]["imgs"])
            inter = sorted(set(imgs_r.tolist()) & set(imgs_y.tolist()))
            if not inter:
                out[f"{joint}/{vbin}"] = {"note": "no shared images"}
                continue

            def grp(errs, imgs):
                g = {}
                for e, i in zip(errs, imgs):
                    if i in inter_set:
                        b = g.setdefault(int(i), [0, 0.0])
                        b[0] += 1
                        b[1] += float(e) ** 2
                return g

            inter_set = set(inter)
            gr = grp(errs_r, imgs_r)
            gy = grp(errs_y, imgs_y)
            ids = sorted(set(gr) | set(gy))
            n_img = len(ids)
            cnt_r = np.array([gr.get(i, [0, 0])[0] for i in ids], dtype=float)
            cnt_y = np.array([gy.get(i, [0, 0])[0] for i in ids], dtype=float)
            ss_r = np.array([gr.get(i, [0, 0.0])[1] for i in ids])
            ss_y = np.array([gy.get(i, [0, 0.0])[1] for i in ids])
            n_r = int(cnt_r.sum())
            n_y = int(cnt_y.sum())
            rmse_r = float(np.sqrt(ss_r.sum() / n_r))
            rmse_y = float(np.sqrt(ss_y.sum() / n_y))
            rng = np.random.default_rng(SEED)
            idx = np.arange(n_img)
            boot_deltas = np.empty(BOOT_N)
            for b in range(BOOT_N):
                ridx = rng.integers(0, n_img, n_img)
                nr = cnt_r[ridx].sum()
                ny = cnt_y[ridx].sum()
                rr = np.sqrt(ss_r[ridx].sum() / nr) if nr else 0.0
                ry = np.sqrt(ss_y[ridx].sum() / ny) if ny else 0.0
                boot_deltas[b] = ry - rr
            lo, hi = np.percentile(boot_deltas, [2.5, 97.5])
            out[f"{joint}/{vbin}"] = {
                "rmse_r5": round(rmse_r, 1),
                "rmse_yolo": round(rmse_y, 1),
                "delta": round(rmse_y - rmse_r, 1),
                "ci_lo": round(float(lo), 1),
                "ci_hi": round(float(hi), 1),
                "sig": not (lo <= 0 <= hi),
                "n_img_shared": n_img,
                "n_inst_r": n_r,
                "n_inst_y": n_y,
            }
    return out


def analysis_C(model):
    """OneEuro 参数网格：12 组合 × 2 域三指标 + 每域最弱/最强参数 AP 代价。"""
    run_id = f"{model}__posetrack-val"
    im_meta, kp_names, gt_by_img, preds, preds_by_img = load_run(run_id)
    segments = build_segments(gt_by_img, preds_by_img, im_meta)

    def metrics_after(repl):
        sm_segs = [
            [(f, i, a, {"keypoints": repl.get(id(d), d["keypoints"]),
                        "score": d["score"]})
             for f, i, a, d in seg]
            for seg in segments]
        _, overall = summarize(*collect_metrics(sm_segs), kp_names)
        return {k: round(float(v), 6) for k, v in overall.items()}

    def smooth_params(min_cutoff, beta, domain):
        repl = {}
        for seg in segments:
            for k in range(K):
                for pos, sc, dets in kp_runs(seg, k):
                    if domain == "px":
                        sm = np.column_stack([
                            one_euro_filter(pos[:, 0], min_cutoff=min_cutoff, beta=beta),
                            one_euro_filter(pos[:, 1], min_cutoff=min_cutoff, beta=beta)])
                        for i, det in enumerate(dets):
                            kp = repl.setdefault(id(det), list(det["keypoints"]))
                            kp[3 * k] = float(sm[i, 0])
                            kp[3 * k + 1] = float(sm[i, 1])
                    else:
                        q = pos / sc[:, None]
                        qs = np.column_stack([
                            one_euro_filter(q[:, 0], min_cutoff=min_cutoff, beta=beta),
                            one_euro_filter(q[:, 1], min_cutoff=min_cutoff, beta=beta)])
                        for i, det in enumerate(dets):
                            kp = repl.setdefault(id(det), list(det["keypoints"]))
                            kp[3 * k] = float(qs[i, 0] * sc[i])
                            kp[3 * k + 1] = float(qs[i, 1] * sc[i])
        return repl

    full_raw = float(evaluate_m1(gt_by_img, preds_by_img)[0].mean())
    grid = {}
    for mc in GRID_CUTOFF:
        for beta in GRID_BETA:
            for domain in ("px", "q"):
                grid[f"{domain}/mc={mc}/beta={beta}"] = metrics_after(
                    smooth_params(mc, beta, domain))

    ap_samples = {}
    for domain, params in (("px", (GRID_CUTOFF[0], GRID_BETA[0])),
                           ("px", (GRID_CUTOFF[-1], GRID_BETA[-1])),
                           ("q", (GRID_CUTOFF[0], GRID_BETA[0])),
                           ("q", (GRID_CUTOFF[-1], GRID_BETA[-1]))):
        repl = smooth_params(*params, domain)
        preds_sm = [{**p, "keypoints": repl.get(id(p), p["keypoints"])} for p in preds]
        psm_by_img = {}
        for p in preds_sm:
            psm_by_img.setdefault(p["image_id"], []).append(p)
        full_sm = float(evaluate_m1(gt_by_img, psm_by_img)[0].mean())
        ap_samples[f"{domain}/mc={params[0]}/beta={params[1]}"] = {
            "ap_raw": round(full_raw, 4),
            "ap_smoothed": round(full_sm, 4),
            "delta": round(full_sm - full_raw, 4),
        }
    return {"grid": grid, "ap_samples": ap_samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=("A", "B", "C"))
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--dataset", choices=M2_DATASETS)
    args = ap.parse_args()

    result = {}
    if OUT.is_file():
        result = json.load(open(OUT))

    if args.which == "A":
        for m in MODELS:
            result[f"A/{m}"] = analysis_A(m)
    elif args.which == "B":
        result[f"B/{args.dataset}"] = analysis_B(args.dataset)
    else:
        result[f"C/{args.model}"] = analysis_C(args.model)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items()
                      if k.startswith(f"{args.which}/")}, indent=1, default=str))


if __name__ == "__main__":
    main()
