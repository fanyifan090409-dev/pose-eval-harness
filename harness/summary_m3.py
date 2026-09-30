#!/usr/bin/env python3
"""summary_m3.py — 打印各 run 的 M3 紧凑汇总（raw/smoothed/smoothed_px 三口径）。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for run in ["yolo11s-pose__posetrack-val", "round5-3__posetrack-val"]:
    m = json.load(open(ROOT / "runs" / run / "metrics.json"))["m3"]
    print("==", run)
    print(" segments:", m["segments"])
    for k in ["raw", "smoothed", "smoothed_px"]:
        ov = m[k]["overall"]
        md = m[k].get("map_delta", {})
        print(
            f" {k:<12}"
            f" disp_mean={ov['disp_mean']:.4f} jitter_mean={ov['jitter_mean']:.4f}"
            f" disp_p95={ov['disp_p95']:.4f}"
            f" jump.05={ov['jump_0.05']:.4f} jump.1={ov['jump_0.1']:.4f}"
            f" jump.2={ov['jump_0.2']:.4f}"
            f" | AP {md.get('ap_raw')} -> {md.get('ap_smoothed')} ({md.get('delta')})"
        )
