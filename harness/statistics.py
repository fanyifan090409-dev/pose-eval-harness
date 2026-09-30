#!/usr/bin/env python3
"""T6 统计层：bootstrap CI + Spearman/Kendall（G1 §4）。

- paired_bootstrap：通用单元级配对百分位 bootstrap（M2 实例 / M3 轨迹段）
- ap_bootstrap：M1 图像级——预计算逐图匹配记录（match_rows_by_img 唯一源），
  重采样图像重算 AP；两模型共享重采样下标（配对）
- AP 的 NumPy 路径与 stratified_map.ap_from_rows 数值全等（stable sort +
  101 点 searchsorted + 后缀最大平滑），CLI 里用静态全量 AP 自检断言
- spearman_matrix / kendall_consistency：RQ4 冗余性分析（cells 表，T7 接入）
- CLI：--dataset --run-a --run-b → runs/stats__<dataset>.json
  （m1/m2/m3 各自独立：对应引擎的 units 落盘存在才算，缺失跳过不报错）

单元落盘契约（各引擎产出，本层只读）：
- m1：不落盘，CLI 内 match_rows_by_img → pack_units → npz 缓存（runs/<run>/m1_units.npz）
- m2：runs/<run>/m2_units.json = {joint: {v2: {"errs": [...], "imgs": [...]}, v1: {...}}}
  （angles.py 写；imgs 为误差的图像归属）
- m3：runs/<run>/m3_units.json = [ {kp_id: {"d": [...], "j": [...]}}, ... ] 每段一个
  （temporal.py 写；d/j 为该段该关键点的逐对值，两者长度不同各自维护偏移）

配对口径：M1 两模型共享图像集 → 配对重采样；M2/M3 的单元集随匹配结果而
不同（误差实例/轨迹段不跨模型对齐）→ 各自独立重采样（非配对，protocol 段注明）。

用法：
    python harness/statistics.py --dataset coco-val \
        --run-a yolo11s-pose__coco-val --run-b round5-3__coco-val --n-boot 200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.stratified_map import K, LAYERS, load_gt_coco, match_rows_by_img

N_BOOT = 10000
SEED = 42
ALPHA = 0.05
M3_METRICS = ("disp_mean", "jitter_mean", "disp_p95", "jitter_p95",
              "jump_0.05", "jump_0.1", "jump_0.2")
M3_TAUS = (0.05, 0.1, 0.2)


# ---------- AP 的 NumPy 路径（与 ap_from_rows 全等） ----------

def ap_from_arrays(scores, tp, fp, n_gt):
    """NumPy 版 10 阈值 AP：stable 降序排序 → cumsum → 后缀最大平滑 → 101 点插值。"""
    if n_gt <= 0:
        return np.zeros(10)
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    fp = fp[order]
    aps = np.empty(10)
    rec_grid = np.linspace(0, 1, 101)
    for ti in range(10):
        tpc = np.cumsum(tp[:, ti])
        fpc = np.cumsum(fp[:, ti])
        rc = tpc / n_gt
        pr = tpc / (tpc + fpc + np.spacing(1.0))
        pr = np.maximum.accumulate(pr[::-1])[::-1]  # 后向平滑 = 后缀最大（与逐元素循环等价）
        inds = np.searchsorted(rc, rec_grid, side="left")
        q = np.where(inds < len(pr), pr[np.minimum(inds, len(pr) - 1)], 0.0)
        aps[ti] = q.mean()
    return aps


def pack_units(per_img, universe):
    """逐图记录 → 平铺数组 + 每图偏移。单元 = universe 内图像（含空图）。

    universe = GT ∪ 两模型预测的图像并集（升序），配对同域；
    preds.json 覆盖图集可能超过 GT 注释图（无注释图 dt-only 照计 FP，与
    official.json pycoco 口径一致），且两模型预测图集不同，故必须对齐并集。
    拼接序 = universe 升序；evaluate_m1 用 dict 插入序，仅跨图 score 精确
    平局时稳定排序结果不同（实测无平局，静态自检 1e-4 容差覆盖）。
    """
    def empty_img():
        return 0, {L: 0 for L in LAYERS}, {L: 0 for L in LAYERS}, [], {L: [] for L in LAYERS}

    entries = [per_img.get(i) or empty_img() for i in universe]
    n_gts = np.array([e[0] for e in entries], dtype=np.int64)

    def pack(rows_of):
        scores, tps, fps = [], [], []
        offsets = np.zeros(len(universe) + 1, dtype=np.int64)
        for i, e in enumerate(entries):
            rows = rows_of(e)
            offsets[i + 1] = offsets[i] + len(rows)
            for s, t, f in rows:
                scores.append(s)
                tps.append(t)
                fps.append(f)
        return (np.asarray(scores, dtype=np.float64),
                np.asarray(tps, dtype=bool),
                np.asarray(fps, dtype=bool),
                offsets)

    full = pack(lambda e: e[3])
    layers = {L: pack(lambda e, L=L: e[4][L]) for L in LAYERS}
    return {"img_ids": list(universe), "n_gts": n_gts, "full": full, "layers": layers}


def _sample_ap(pack, n_gts, idx):
    scores, tp, fp, offsets = pack
    lens = offsets[idx + 1] - offsets[idx]
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(10)
    seg_start = np.repeat(offsets[idx], lens)
    flat_idx = seg_start + (np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens))
    n_gt = int(n_gts[idx].sum())
    return ap_from_arrays(scores[flat_idx], tp[flat_idx], fp[flat_idx], n_gt)


def ap_bootstrap(units_a, units_b, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """M1 图像级配对 bootstrap。返回 {cell: {ci_lo, ci_hi, mean_delta, point_delta,
    ap_a, ap_b, sig, n_units, n_boot, seed}}；cell ∈ full/v2/v1/v0。"""
    n_units = len(units_a["img_ids"])
    assert units_a["img_ids"] == units_b["img_ids"], "两 run 图像集不一致，无法配对"
    rng = np.random.default_rng(seed)
    cells = {"full": ("full",)} | {f"v{L}": ("layers", L) for L in LAYERS}

    def cell_ap(units, cell, idx):
        src = cells[cell]
        if src[0] == "full":
            return _sample_ap(units["full"], units["n_gts"], idx)
        return _sample_ap(units["layers"][src[1]], units["n_gts"], idx)

    deltas = {c: np.empty(n_boot) for c in cells}
    for it in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        for c in cells:
            deltas[c][it] = cell_ap(units_a, c, idx).mean() - cell_ap(units_b, c, idx).mean()

    out = {}
    for c in cells:
        idx0 = np.arange(n_units)
        ap_a = float(cell_ap(units_a, c, idx0).mean())
        ap_b = float(cell_ap(units_b, c, idx0).mean())
        lo, hi = np.percentile(deltas[c], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[c] = {
            "ci_lo": round(float(lo), 6),
            "ci_hi": round(float(hi), 6),
            "mean_delta": round(float(deltas[c].mean()), 6),
            "point_delta": round(ap_a - ap_b, 6),
            "ap_a": round(ap_a, 6),
            "ap_b": round(ap_b, 6),
            "sig": bool(lo > 0 or hi < 0),
        }
    out["_protocol"] = {"n_units": n_units, "n_boot": n_boot, "seed": seed,
                        "alpha": alpha, "paired": True, "unit": "image"}
    return out


def save_units_npz(units, path):
    np.savez(
        path,
        img_ids=np.array(units["img_ids"]),
        n_gts=units["n_gts"],
        f_scores=units["full"][0], f_tp=units["full"][1], f_fp=units["full"][2],
        f_offsets=units["full"][3],
        **{f"l{L}_scores": units["layers"][L][0] for L in LAYERS},
        **{f"l{L}_tp": units["layers"][L][1] for L in LAYERS},
        **{f"l{L}_fp": units["layers"][L][2] for L in LAYERS},
        **{f"l{L}_offsets": units["layers"][L][3] for L in LAYERS},
    )


def load_units_npz(path):
    z = np.load(path, allow_pickle=False)
    units = {
        "img_ids": list(z["img_ids"]),
        "n_gts": z["n_gts"],
        "full": (z["f_scores"], z["f_tp"], z["f_fp"], z["f_offsets"]),
        "layers": {L: (z[f"l{L}_scores"], z[f"l{L}_tp"], z[f"l{L}_fp"], z[f"l{L}_offsets"])
                   for L in LAYERS},
    }
    return units


# ---------- M2 图像级 bootstrap（单元集随匹配结果而不同 → 非配对） ----------

def pack_m2(unit):
    """{errs, imgs} → (errs_flat, offsets, img_ids)：按图像分组，img_ids 升序。"""
    by_img = {}
    for e, i in zip(unit["errs"], unit["imgs"]):
        by_img.setdefault(i, []).append(e)
    img_ids = sorted(by_img)
    errs_flat = []
    offsets = np.zeros(len(img_ids) + 1, dtype=np.int64)
    for k, i in enumerate(img_ids):
        es = by_img[i]
        offsets[k + 1] = offsets[k] + len(es)
        errs_flat.extend(es)
    return np.asarray(errs_flat, dtype=np.float64), offsets, img_ids


def rmse_boot(pack, idx):
    """图像重采样后的 RMSE；所选图像无误差单元时返回 None。"""
    errs, offsets, _ = pack
    lens = offsets[idx + 1] - offsets[idx]
    tot = int(lens.sum())
    if tot == 0:
        return None
    seg_start = np.repeat(offsets[idx], lens)
    flat_idx = seg_start + (np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens))
    e = errs[flat_idx]
    return float(np.sqrt(np.mean(e**2)))


def m2_bootstrap(ua2, ub2, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """M2 图像级 bootstrap：两模型各自独立重采样图像（配对不可行，单元集不同）。

    返回 {joint: {vbin: {ci_lo, ci_hi, mean_delta, point_delta, rmse_a, rmse_b,
    sig, n_img_a, n_img_b, n_boot, seed}}}
    """
    rng = np.random.default_rng(seed)
    out = {}
    for joint in ua2:
        out[joint] = {}
        for vbin in ua2[joint]:
            ea = ua2[joint][vbin]
            eb = ub2.get(joint, {}).get(vbin) or {}
            if not ea.get("errs") or not eb.get("errs"):
                continue
            pa = pack_m2(ea)
            pb = pack_m2(eb)
            na, nb = len(pa[2]), len(pb[2])
            deltas = np.empty(n_boot)
            for it in range(n_boot):
                ia = rng.integers(0, na, size=na)
                ib = rng.integers(0, nb, size=nb)
                deltas[it] = rmse_boot(pa, ia) - rmse_boot(pb, ib)
            rmse_a = rmse_boot(pa, np.arange(na))
            rmse_b = rmse_boot(pb, np.arange(nb))
            lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[joint][vbin] = {
                "ci_lo": round(float(lo), 6),
                "ci_hi": round(float(hi), 6),
                "mean_delta": round(float(deltas.mean()), 6),
                "point_delta": round(rmse_a - rmse_b, 6),
                "rmse_a": round(rmse_a, 6),
                "rmse_b": round(rmse_b, 6),
                "sig": bool(lo > 0 or hi < 0),
                "n_img_a": na, "n_img_b": nb,
                "n_boot": int(n_boot), "seed": int(seed),
            }
    return out


# ---------- M3 单元打包与聚合（复刻 temporal.summarize 的 overall 层） ----------

def pack_m3(seg_units):
    """每段 {kp_id: {"d": [...], "j": [...]}} → {kp: (d_flat, j_flat, d_off, j_off)}。

    d/j 每 run 长度不同（d 比 j 少 1 个），必须各自维护偏移。
    """
    packs = {}
    for k in range(K):
        d_flat, j_flat = [], []
        d_off = np.zeros(len(seg_units) + 1, dtype=np.int64)
        j_off = np.zeros(len(seg_units) + 1, dtype=np.int64)
        for i, seg in enumerate(seg_units):
            v = seg.get(str(k), {})
            d = np.asarray(v.get("d", []), dtype=np.float64)
            j = np.asarray(v.get("j", []), dtype=np.float64)
            d_off[i + 1] = d_off[i] + len(d)
            j_off[i + 1] = j_off[i] + len(j)
            d_flat.append(d)
            j_flat.append(j)
        packs[k] = (np.concatenate(d_flat) if d_flat else np.zeros(0),
                    np.concatenate(j_flat) if j_flat else np.zeros(0),
                    d_off, j_off)
    return packs


def m3_overall(packs, idx):
    """复刻 summarize 聚合：每关键点段级均值/pooled 分位/跳变率 → 关键点间取均值。"""
    per_kp = {}
    for k in range(K):
        d_f, j_f, d_off, j_off = packs[k]
        lens = d_off[idx + 1] - d_off[idx]
        sel = lens > 0
        if not sel.any():
            continue
        idx_sel = idx[sel]
        lens_sel = lens[sel]
        tot = int(lens_sel.sum())
        seg_start = np.repeat(d_off[idx_sel], lens_sel)
        flat_idx = seg_start + (np.arange(tot) - np.repeat(np.cumsum(lens_sel) - lens_sel, lens_sel))
        d_pool = d_f[flat_idx]
        j_lens = j_off[idx_sel + 1] - j_off[idx_sel]
        j_tot = int(j_lens.sum())
        j_seg_start = np.repeat(j_off[idx_sel], j_lens)
        j_flat_idx = j_seg_start + (np.arange(j_tot) - np.repeat(np.cumsum(j_lens) - j_lens, j_lens))
        j_pool = j_f[j_flat_idx]
        bounds = np.concatenate(([0], np.cumsum(lens_sel)[:-1]))
        j_bounds = np.concatenate(([0], np.cumsum(j_lens)[:-1]))
        d_seg_means = np.add.reduceat(d_pool, bounds) / lens_sel
        j_seg_means = np.add.reduceat(j_pool, j_bounds) / j_lens
        per_kp[k] = {
            "disp_mean": float(d_seg_means.mean()),
            "jitter_mean": float(j_seg_means.mean()),
            "disp_p95": float(np.percentile(d_pool, 95)),
            "jitter_p95": float(np.percentile(j_pool, 95)),
            **{f"jump_{t}": float((d_pool > t).mean()) for t in M3_TAUS},
        }
    overall = {m: float(np.mean([per_kp[k][m] for k in per_kp])) for m in M3_METRICS}
    return overall


def m3_bootstrap(seg_units_a, seg_units_b, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """M3 轨迹段级 bootstrap：8 个 overall 指标的 Δ CI。

    两模型段集不同（匹配结果不同）→ 各自独立重采样（非配对）。
    """
    packs_a = pack_m3(seg_units_a)
    packs_b = pack_m3(seg_units_b)
    na, nb = len(seg_units_a), len(seg_units_b)
    rng = np.random.default_rng(seed)
    deltas = {m: np.empty(n_boot) for m in M3_METRICS}
    for it in range(n_boot):
        ia = rng.integers(0, na, size=na)
        ib = rng.integers(0, nb, size=nb)
        oa = m3_overall(packs_a, ia)
        ob = m3_overall(packs_b, ib)
        for m in M3_METRICS:
            deltas[m][it] = oa[m] - ob[m]
    out = {}
    for m in M3_METRICS:
        lo, hi = np.percentile(deltas[m], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = {
            "ci_lo": round(float(lo), 6),
            "ci_hi": round(float(hi), 6),
            "mean_delta": round(float(deltas[m].mean()), 6),
            "sig": bool(lo > 0 or hi < 0),
        }
    out["_protocol"] = {"n_units_a": na, "n_units_b": nb, "n_boot": n_boot, "seed": seed,
                        "alpha": alpha, "paired": False, "unit": "track segment"}
    return out


# ---------- RQ4 冗余性（T7 接入 cells 表） ----------

def spearman_matrix(metric_values):
    """metric_values: {name: 等长数组}（cells 对齐）→ (names, rho 矩阵, p 矩阵)。"""
    from scipy.stats import spearmanr
    names = list(metric_values)
    arrs = [np.asarray(metric_values[n], dtype=np.float64) for n in names]
    rho = np.ones((len(names), len(names)))
    p = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r, pv = spearmanr(arrs[i], arrs[j])
            rho[i, j] = rho[j, i] = r
            p[i, j] = p[j, i] = pv
    return names, rho, p


def kendall_consistency(diff_metric, diff_ap):
    """协议指标差与全量 OKS 差的符号一致性（cells 对齐，忽略同分对）。τ=(C-D)/(C+D)。"""
    d1 = np.asarray(diff_metric, dtype=np.float64)
    d2 = np.asarray(diff_ap, dtype=np.float64)
    c = d = 0
    n = len(d1)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = np.sign(d1[i] - d1[j])
            s2 = np.sign(d2[i] - d2[j])
            if s1 == 0 or s2 == 0:
                continue
            if s1 == s2:
                c += 1
            else:
                d += 1
    return {"concordant": c, "discordant": d,
            "tau": round((c - d) / (c + d), 6) if c + d else None}


# ---------- CLI ----------

def load_run_data(run_id):
    """读 run 的 meta/GT/preds → (gt_by_img, preds_by_img)。"""
    run_dir = ROOT / "runs" / run_id
    meta = json.load(open(run_dir / "meta.json"))
    gt_json = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))["datasets"][meta["dataset"]]["gt_json"]
    gt_by_img = load_gt_coco(gt_json)
    preds = json.load(open(run_dir / "preds.json"))
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)
    return gt_by_img, preds_by_img


def build_m1_units(run_id, universe, data=None):
    """match_rows_by_img → pack_units（universe 域）→ npz 缓存。

    data=(gt_by_img, preds_by_img) 由调用方传入可省一次重载；缓存与 universe
    不符（如旧打包序）自动重建。
    """
    run_dir = ROOT / "runs" / run_id
    cache = run_dir / "m1_units.npz"
    if cache.is_file():
        units = load_units_npz(cache)
        if units["img_ids"] == list(universe):
            return units
    if data is None:
        data = load_run_data(run_id)
    gt_by_img, preds_by_img = data
    units = pack_units(match_rows_by_img(gt_by_img, preds_by_img), universe)
    save_units_npz(units, cache)
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    run_a, run_b = ROOT / "runs" / args.run_a, ROOT / "runs" / args.run_b
    ma = json.load(open(run_a / "metrics.json"))
    mb = json.load(open(run_b / "metrics.json"))
    stats = {
        "dataset": args.dataset,
        "run_a": args.run_a,
        "run_b": args.run_b,
        "protocol": {"n_boot": args.n_boot, "seed": SEED, "alpha": ALPHA,
                     "ci": "95% 百分位 bootstrap",
                     "pairing": {"m1": "paired（共享图像域）",
                                 "m2": "independent（单元集随匹配结果不同）",
                                 "m3": "independent（轨迹段集随匹配结果不同）"}},
    }

    # M1：图像级 AP bootstrap + 静态自检（打包路径必须复现 metrics.json 的 m1）
    if "m1" in ma and "m1" in mb:
        ga, pa = load_run_data(args.run_a)
        gb, pb = load_run_data(args.run_b)
        # 配对域 = 两模型 GT ∪ 预测图并集（升序）；两 run 预测图集不同，未对齐无法配对
        universe = sorted(set(ga) | set(gb) | set(pa) | set(pb))
        ua = build_m1_units(args.run_a, universe, (ga, pa))
        ub = build_m1_units(args.run_b, universe, (gb, pb))
        for name, u, m in (("a", ua, ma["m1"]), ("b", ub, mb["m1"])):
            idx0 = np.arange(len(u["img_ids"]))
            ap_full = float(_sample_ap(u["full"], u["n_gts"], idx0).mean())
            err = abs(ap_full - m["full"]["ap"])
            # metrics.json 只存 4 位小数，容差取半末位舍入上界 1e-4
            assert err < 1e-4, f"M1 打包自检失败 {name}: {ap_full} vs {m['full']['ap']}"
            print(f"M1 静态自检 {name}: AP {ap_full:.4f} 与 metrics.json 一致")
        stats["m1"] = ap_bootstrap(ua, ub, n_boot=args.n_boot)
        print(f"M1 bootstrap 完成: {list(stats['m1'].keys())}")

    # M2：图像级 RMSE bootstrap（依赖 angles.py 的 m2_units.json，误差带图像归属）
    units_files = (run_a / "m2_units.json", run_b / "m2_units.json")
    if all(f.is_file() for f in units_files) and "m2" in ma and "m2" in mb:
        ua2 = json.load(open(units_files[0]))
        ub2 = json.load(open(units_files[1]))
        rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
        # 静态自检：全量 errs 的 RMSE 必须复现 metrics.json（4 位小数容差 1e-4）
        for joint in ua2:
            for vbin in ua2[joint]:
                if not ua2[joint][vbin].get("errs"):
                    continue
                ref = ma["m2"]["angles"][joint][vbin]["rmse"]
                assert abs(rmse(ua2[joint][vbin]["errs"]) - ref) < 1e-4, \
                    f"M2 自检失败 a/{joint}/{vbin}"
        print("M2 静态自检 a: RMSE 与 metrics.json 一致")
        stats["m2"] = m2_bootstrap(ua2, ub2, n_boot=args.n_boot)
        stats["m2"]["_protocol"] = {"paired": False, "unit": "image",
                                    "n_boot": args.n_boot, "seed": SEED, "alpha": ALPHA,
                                    "note": "两模型图像集不同（有效匹配对差），各自独立重采样"}
        print(f"M2 bootstrap 完成: {list(stats['m2'].keys())}")

    # M3：轨迹段级 bootstrap（依赖 temporal.py 的 m3_units.json）
    units_files3 = (run_a / "m3_units.json", run_b / "m3_units.json")
    if all(f.is_file() for f in units_files3) and "m3" in ma and "m3" in mb:
        ua3 = json.load(open(units_files3[0]))
        ub3 = json.load(open(units_files3[1]))
        # 静态自检：全量 idx 聚合必须复现 metrics.json 的 m3 raw overall（6 位小数容差 1e-6）
        oa0 = m3_overall(pack_m3(ua3), np.arange(len(ua3)))
        ob0 = m3_overall(pack_m3(ub3), np.arange(len(ub3)))
        for m in M3_METRICS:
            assert abs(oa0[m] - ma["m3"]["raw"]["overall"][m]) < 1e-6, f"M3 自检失败 a/{m}"
            assert abs(ob0[m] - mb["m3"]["raw"]["overall"][m]) < 1e-6, f"M3 自检失败 b/{m}"
        print("M3 静态自检 a/b: 聚合与 metrics.json 一致")
        stats["m3"] = m3_bootstrap(ua3, ub3, n_boot=args.n_boot)
        print(f"M3 bootstrap 完成: {[k for k in stats['m3'] if k != '_protocol']}")

    out_path = ROOT / "runs" / f"stats__{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"stats: {out_path}")


if __name__ == "__main__":
    main()
