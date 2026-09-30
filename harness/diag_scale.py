#!/usr/bin/env python3
"""diag_scale.py — M3 平滑诊断：GT bbox 尺度噪声 + q 域 vs 像素域 OneEuro 对照。

目的：解释「q 域平滑后像素域抖动不降反升」。假设机制 = q_t 平滑后乘回逐帧
s_t（GT bbox 标注噪声）把尺度噪声重新注入像素域；对照 = OneEuro 直接施于像素坐标。
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
from harness.stratified_map import load_gt_coco
from harness.temporal import ONEEURO, build_segments, collect_metrics, kp_runs, summarize

K = 17


def scale_noise(segments):
    """所有 kp run 内逐帧 |s_t/s_{t-1} - 1| 的 median/p95（尺度噪声水平）。"""
    ratios = []
    for seg in segments:
        for k in range(K):
            for pos, sc, _ in kp_runs(seg, k):
                if len(sc) < 2:
                    continue
                ratios.append(np.abs(sc[1:] / sc[:-1] - 1.0))
    r = np.concatenate(ratios) if ratios else np.array([])
    return float(np.median(r)), float(np.percentile(r, 95))


def pixel_replacements(segments):
    """OneEuro 直接施于像素坐标（与 q 域同样的段/run/断档重置口径）。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<run_id> 目录名")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    meta = json.load(open(run_dir / "meta.json"))
    ds_cfg = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))["datasets"][meta["dataset"]]
    data = json.load(open(ds_cfg["gt_json"]))
    im_meta = {im["id"]: (im["vid_id"], im["frame"], im["is_labeled"]) for im in data["images"]}
    kp_names = data["categories"][0]["keypoints"]
    gt_by_img = load_gt_coco(ds_cfg["gt_json"])
    preds = json.load(open(run_dir / "preds.json"))
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    segments = build_segments(gt_by_img, preds_by_img, im_meta)
    print(f"segments: {len(segments)}")

    med, p95 = scale_noise(segments)
    print(f"scale noise |s_t/s_{{t-1}}-1|: median={med:.4f} p95={p95:.4f}")

    _, raw_ov = summarize(*collect_metrics(segments), kp_names)

    repl = pixel_replacements(segments)
    sm_segs = [[(f, iid, ann,
                 {"keypoints": repl.get(id(d), d["keypoints"]), "score": d["score"]})
                for f, iid, ann, d in seg] for seg in segments]
    _, pix_ov = summarize(*collect_metrics(sm_segs), kp_names)

    m3 = json.load(open(run_dir / "metrics.json"))["m3"]
    q_ov = m3["smoothed"]["overall"]

    keys = ["disp_mean", "jitter_mean", "disp_p95", "jitter_p95",
            "jump_0.05", "jump_0.1", "jump_0.2"]
    print(f"{'metric':<12}{'raw':>10}{'q-sm':>10}{'px-sm':>10}")
    for m in keys:
        print(f"{m:<12}{raw_ov[m]:>10.6f}{q_ov[m]:>10.6f}{pix_ov[m]:>10.6f}")


if __name__ == "__main__":
    main()
