#!/usr/bin/env python3
"""P1 数据项（论文A 审稿修订）：A13 层差配对 bootstrap CI / A14 BH-FDR / A15 MAE·P95。

- A13：同一 run 内 v1−v2 层 RMSE 差的图像级配对 bootstrap（两层层共用同一
  重采样下标，配对口径），覆盖 6 个检测式 run × 4 关节 = 24 组合。
- A15：审稿 S7 承诺的 MAE/P95——按 run × 关节 × 层 计算 mean(|err|) 与 95 分位。
- A14：三个核心比较族的 BH-FDR——F1 M1 全 AP（3 比较）、F2 M2 逐格 RMSE（24）、
  F3 M3 抖动指标（7，轨迹段级）。bootstrap Δ 分布 → 双侧 p = 2·min(P(Δ≤0), P(Δ≥0))。

产物：runs/a13_layer_diff.json / runs/a14_fdr.json / runs/a15_mae_p95.json
用法：/home/ubuntu/Miniconda3/envs/py310/bin/python harness/p1_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.statistics import (  # noqa: E402
    pack_m2, rmse_boot, pack_m3, m3_overall, _sample_ap, load_units_npz,
    build_m1_units, load_run_data,
)

N_BOOT = 10000
SEED = 42
ALPHA = 0.05
RUNS = ["round5-3", "yolo11s-pose"]
DATASETS = ["coco-val", "crowdpose-val", "ochuman-test"]
JOINTS = ["left_elbow", "right_elbow", "left_knee", "right_knee"]
M3_METRICS = ("disp_mean", "jitter_mean", "disp_p95", "jitter_p95",
              "jump_0.05", "jump_0.1", "jump_0.2")
RUNS_DIR = ROOT / "runs"


def run_path(run, dataset):
    return RUNS_DIR / f"{run}__{dataset}"


def p_two_sided(deltas):
    """bootstrap Δ 分布 → 双侧 p（截断到 1）。"""
    return float(min(1.0, 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())))


def layer_stats(pack):
    """pack → (img_ids, 每图误差平方和 S2, 每图误差数 N)。"""
    errs, offsets, img_ids = pack
    s2 = np.add.reduceat(errs ** 2, offsets[:-1])
    n = np.diff(offsets).astype(np.int64)
    return img_ids, s2, n


def union_paired_delta(rng, u_ids, img_ids_1, s2_1, n_1, img_ids_2, s2_2, n_2):
    """并集图像域配对重采样：同一重采样下标作用于两层层共享的并集图像。

    返回 Δ = RMSE(v1) − RMSE(v2)（重采样版）；若任一层在重采样中无误差单元
    则返回 None（调用方重试）。RMSE² = Σ(cnt·S2) / Σ(cnt·N)，cnt 为并集计数。
    """
    idx = rng.integers(0, len(u_ids), size=len(u_ids))
    cnt = np.bincount(idx, minlength=len(u_ids))
    vals = []
    for img_ids, s2, n in ((img_ids_1, s2_1, n_1), (img_ids_2, s2_2, n_2)):
        pos = {i: k for k, i in enumerate(u_ids)}
        sel = np.array([pos[i] for i in img_ids], dtype=np.int64)
        c = cnt[sel].astype(np.float64)
        denom = float((n * c).sum())
        if denom <= 0:
            return None
        vals.append(np.sqrt(float((s2 * c).sum()) / denom))
    return vals[0] - vals[1]


# ---------- A13：v1−v2 层 RMSE 差，图像级配对 bootstrap ----------

def compute_a13():
    out = {}
    for run in RUNS:
        for ds in DATASETS:
            units = json.load(open(run_path(run, ds) / "m2_units.json"))
            for joint in JOINTS:
                v2 = units.get(joint, {}).get("v2") or {}
                v1 = units.get(joint, {}).get("v1") or {}
                if not v2.get("errs") or not v1.get("errs"):
                    continue
                img_ids_1, s2_1, n_1 = layer_stats(pack_m2(v1))
                img_ids_2, s2_2, n_2 = layer_stats(pack_m2(v2))
                u_ids = sorted(set(img_ids_1) | set(img_ids_2))
                rng = np.random.default_rng(SEED)
                deltas = np.empty(N_BOOT)
                for it in range(N_BOOT):
                    d = None
                    while d is None:
                        d = union_paired_delta(
                            rng, u_ids, img_ids_1, s2_1, n_1,
                            img_ids_2, s2_2, n_2)
                    deltas[it] = d
                r1 = float(np.sqrt(s2_1.sum() / n_1.sum()))
                r2 = float(np.sqrt(s2_2.sum() / n_2.sum()))
                lo, hi = np.percentile(deltas, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
                key = f"{run} | {ds} | {joint}"
                out[key] = {
                    "rmse_v1": round(r1, 4),
                    "rmse_v2": round(r2, 4),
                    "delta": round(r1 - r2, 4),
                    "ci_lo": round(float(lo), 4),
                    "ci_hi": round(float(hi), 4),
                    "sig": bool(lo > 0 or hi < 0),
                    "n_img_v1": int(len(img_ids_1)),
                    "n_img_v2": int(len(img_ids_2)),
                    "n_img_union": int(len(u_ids)),
                    "n_boot": N_BOOT,
                    "seed": SEED,
                }
    out["_protocol"] = {
        "design": "paired image-level bootstrap of RMSE(v1)-RMSE(v2) within run",
        "pairing": "single resample index over the union image set, "
                   "applied to both layers per replicate",
        "n_boot": N_BOOT, "seed": SEED, "alpha": ALPHA,
        "ci": "95% percentile CI",
    }
    return out


# ---------- A15：MAE / P95（审稿 S7） ----------

def compute_a15():
    out = {}
    for run in RUNS:
        for ds in DATASETS:
            units = json.load(open(run_path(run, ds) / "m2_units.json"))
            for joint in JOINTS:
                for vbin in ("v2", "v1"):
                    cell = units.get(joint, {}).get(vbin) or {}
                    errs = cell.get("errs")
                    if not errs:
                        continue
                    e = np.abs(np.asarray(errs, dtype=np.float64))
                    key = f"{run} | {ds} | {joint} | {vbin}"
                    out[key] = {
                        "mae": round(float(e.mean()), 4),
                        "p95": round(float(np.percentile(e, 95)), 4),
                        "rmse": round(float(np.sqrt(np.mean(e ** 2))), 4),
                        "n": int(len(e)),
                    }
    out["_protocol"] = {
        "metric": "MAE = mean(|err|); P95 = 95th percentile of |err|; "
                   "RMSE restated for cross-check against Table m2",
    }
    return out


# ---------- A14：BH-FDR 三族 ----------

def bootstrap_p_cell(units_a, units_b, n_units, cell_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for it in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        deltas[it] = cell_fn(units_a, idx) - cell_fn(units_b, idx)
    point = cell_fn(units_a, np.arange(n_units)) - cell_fn(units_b, np.arange(n_units))
    lo, hi = np.percentile(deltas, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {
        "point_delta": round(float(point), 6),
        "ci_lo": round(float(lo), 6),
        "ci_hi": round(float(hi), 6),
        "p": round(p_two_sided(deltas), 6),
        "sig": bool(lo > 0 or hi < 0),
    }


def bh_adjust(items):
    """items: [(label, p)] → [(label, p, p_adj, sig_adj)]，族内 BH。"""
    m = len(items)
    order = sorted(range(m), key=lambda i: items[i][1])
    out = [None] * m
    prev = 0.0
    for rank, i in enumerate(order):
        p = items[i][1]
        padj = min(1.0, p * m / (rank + 1))
        padj = max(padj, prev)  # BH 单调化
        prev = padj
        out[i] = padj
    return [
        (items[i][0], items[i][1], out[i], out[i] < ALPHA)
        for i in range(m)
    ]


def compute_a14():
    # F1：M1 全 AP（图像级配对，域 = 两模型 GT ∪ 预测图并集，口径同 CLI）
    f1 = []
    for ds in DATASETS:
        run_a = f"round5-3__{ds}"
        run_b = f"yolo11s-pose__{ds}"
        ga, pa = load_run_data(run_a)
        gb, pb = load_run_data(run_b)
        universe = sorted(set(ga) | set(gb) | set(pa) | set(pb))
        ua = build_m1_units(run_a, universe, (ga, pa))
        ub = build_m1_units(run_b, universe, (gb, pb))
        assert ua["img_ids"] == ub["img_ids"] == universe, f"{ds} 图像集不一致"
        n = len(universe)
        res = bootstrap_p_cell(
            ua, ub, n,
            lambda u, idx: _sample_ap(u["full"], u["n_gts"], idx).mean())
        f1.append((f"M1 {ds}", res))
        print(f"F1 {ds}: p={res['p']}", flush=True)

    # F2：M2 逐格 RMSE（图像级独立重采样，同 statistics.m2_bootstrap 口径）
    f2 = []
    for ds in DATASETS:
        ua = json.load(open(run_path("round5-3", ds) / "m2_units.json"))
        ub = json.load(open(run_path("yolo11s-pose", ds) / "m2_units.json"))
        for joint in JOINTS:
            for vbin in ("v2", "v1"):
                ea = ua.get(joint, {}).get(vbin) or {}
                eb = ub.get(joint, {}).get(vbin) or {}
                if not ea.get("errs") or not eb.get("errs"):
                    continue
                pa, pb = pack_m2(ea), pack_m2(eb)
                na, nb = len(pa[2]), len(pb[2])
                rng = np.random.default_rng(SEED)
                deltas = np.empty(N_BOOT)
                for it in range(N_BOOT):
                    ia = rng.integers(0, na, size=na)
                    ib = rng.integers(0, nb, size=nb)
                    deltas[it] = rmse_boot(pa, ia) - rmse_boot(pb, ib)
                point = rmse_boot(pa, np.arange(na)) - rmse_boot(pb, np.arange(nb))
                lo, hi = np.percentile(deltas, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
                label = f"M2 {ds} {joint} {vbin}"
                f2.append((label, {
                    "point_delta": round(float(point), 4),
                    "ci_lo": round(float(lo), 4),
                    "ci_hi": round(float(hi), 4),
                    "p": round(p_two_sided(deltas), 6),
                    "sig": bool(lo > 0 or hi < 0),
                }))
                print(f"{label}: p={f2[-1][1]['p']}", flush=True)

    # F3：M3 抖动指标（轨迹段级独立重采样）
    ds = "posetrack-val"
    sa = json.load(open(run_path("round5-3", ds) / "m3_units.json"))
    sb = json.load(open(run_path("yolo11s-pose", ds) / "m3_units.json"))
    pa, pb = pack_m3(sa), pack_m3(sb)
    na, nb = len(sa), len(sb)
    rng = np.random.default_rng(SEED)
    deltas = {m: np.empty(N_BOOT) for m in M3_METRICS}
    for it in range(N_BOOT):
        ia = rng.integers(0, na, size=na)
        ib = rng.integers(0, nb, size=nb)
        oa, ob = m3_overall(pa, ia), m3_overall(pb, ib)
        for m in M3_METRICS:
            deltas[m][it] = oa[m] - ob[m]
    f3 = []
    for m in M3_METRICS:
        d = deltas[m]
        point = m3_overall(pa, np.arange(na))[m] - m3_overall(pb, np.arange(nb))[m]
        lo, hi = np.percentile(d, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        label = f"M3 {ds} {m}"
        f3.append((label, {
            "point_delta": round(float(point), 6),
            "ci_lo": round(float(lo), 6),
            "ci_hi": round(float(hi), 6),
            "p": round(p_two_sided(d), 6),
            "sig": bool(lo > 0 or hi < 0),
        }))
        print(f"{label}: p={f3[-1][1]['p']}", flush=True)

    # 汇总前先把三族原始结果落盘（防汇总步崩溃丢 30 分钟计算）
    json.dump({"F1_M1_full_AP": [[l, it] for l, it in f1],
               "F2_M2_rmse_cells": [[l, it] for l, it in f2],
               "F3_M3_temporal": [[l, it] for l, it in f3]},
              open(RUNS_DIR / "a14_raw.json", "w"), indent=2)
    families = {}
    for name, items in (("F1_M1_full_AP", f1), ("F2_M2_rmse_cells", f2),
                        ("F3_M3_temporal", f3)):
        adjusted = bh_adjust([(lbl, it["p"]) for lbl, it in items])
        rows = {}
        for (lbl, it), (_, p, padj, sig_adj) in zip(items, adjusted):
            rows[lbl] = {**it, "p_adj": round(padj, 6), "sig_adj": bool(sig_adj)}
        n_sig = sum(1 for _, it in items if it["sig"])
        n_sig_adj = sum(1 for r in rows.values() if r["sig_adj"])
        families[name] = {
            "rows": rows,
            "n": len(items),
            "n_sig_raw": n_sig,
            "n_sig_bh": n_sig_adj,
        }
        print(f"{name}: {len(items)} 比较, 原始显著 {n_sig}, BH 后显著 {n_sig_adj}",
              flush=True)
    families["_protocol"] = {
        "comparisons": "round5-3 vs yolo11s-pose, detection-based protocol",
        "p": "two-sided bootstrap p = 2*min(P(d<=0), P(d>=0)), 10000 resamples, seed 42",
        "bh": "Benjamini-Hochberg within family, alpha=0.05",
        "scope": "core comparison families of the main protocol "
                 "(F1: full AP per dataset; F2: per-cell RMSE; F3: temporal metrics)",
    }
    return families


def main():
    # --a14-only：A13/A15 已落盘时只重算 A14（修复汇总 bug 后重跑用）
    if "--a14-only" not in sys.argv:
        a13 = compute_a13()
        json.dump(a13, open(RUNS_DIR / "a13_layer_diff.json", "w"), indent=2)
        print(f"A13 done: {len(a13) - 1} combinations", flush=True)

        a15 = compute_a15()
        json.dump(a15, open(RUNS_DIR / "a15_mae_p95.json", "w"), indent=2)
        print(f"A15 done: {len(a15) - 1} cells", flush=True)

    a14 = compute_a14()
    json.dump(a14, open(RUNS_DIR / "a14_fdr.json", "w"), indent=2)
    print("A14 done", flush=True)


if __name__ == "__main__":
    main()
