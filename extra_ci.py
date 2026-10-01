#!/usr/bin/env python3
"""G7 审稿补算：Δ_occ CI / κ CI / M3 域对比 CI（全离线 NumPy，无 GPU）。

Part 1  Δ_occ：m1_units.npz 图像级配对 bootstrap（每模型 Δ_occ = AP_v1 − AP_full）
Part 2  κ：重跑 M2 匹配收集逐实例 φ（带图像归属）→ 图像级 bootstrap（非配对）
Part 3  M3：逐段预计算三域 d/j → 段级 bootstrap（域间配对、模型间非配对）

用法：python extra_ci.py --n-boot 10000
输出：runs/extra_ci.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
from harness.angles import (JOINTS, KNEE_HIP, TAUS, angle_deg, cohen_kappa,
                            kpts_valid_gt, kpts_valid_pred)
from harness.statistics import (_sample_ap, load_units_npz, m3_overall,
                                pack_m3, M3_METRICS)
from harness.stratified_map import MAX_DETS, compute_oks, load_gt_coco, match_greedy
from harness.temporal import (K, apply_repl, build_segments,
                              pixel_replacements, segment_units,
                              smooth_replacements)

N_BOOT = 10000
SEED = 42
ALPHA = 0.05
M1_DATASETS = ("coco-val", "crowdpose-val", "ochuman-test")
MODELS = ("round5-3", "yolo11s-pose")
DOMAINS = ("raw", "smoothed_px", "smoothed")


def pct(deltas):
    lo, hi = np.percentile(deltas, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
            "mean_delta": round(float(deltas.mean()), 6),
            "sig": bool(lo > 0 or hi < 0)}


def load_run_gt_preds(run_id):
    run_dir = ROOT / "runs" / run_id
    meta = json.load(open(run_dir / "meta.json"))
    cfg = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))
    gt_by_img = load_gt_coco(cfg["datasets"][meta["dataset"]]["gt_json"])
    preds = json.load(open(run_dir / "preds.json"))
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)
    return gt_by_img, preds_by_img


# ---------- Part 1: Δ_occ CI ----------

def delta_occ_bootstrap(units_a, units_b, n_boot, seed):
    """图像级配对 bootstrap：Δ_occ = mAP(v1) − mAP(full)，逐模型 + 模型差。"""
    assert units_a["img_ids"] == units_b["img_ids"], "图像域不一致，无法配对"
    n = len(units_a["img_ids"])
    rng = np.random.default_rng(seed)
    dist = {m: np.empty(n_boot) for m in ("a", "b")}
    for it in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for key, u in (("a", units_a), ("b", units_b)):
            ap_v1 = _sample_ap(u["layers"][1], u["n_gts"], idx).mean()
            ap_full = _sample_ap(u["full"], u["n_gts"], idx).mean()
            dist[key][it] = ap_v1 - ap_full
    idx0 = np.arange(n)
    out = {}
    for key, u in (("a", units_a), ("b", units_b)):
        ap_v1 = _sample_ap(u["layers"][1], u["n_gts"], idx0).mean()
        ap_full = _sample_ap(u["full"], u["n_gts"], idx0).mean()
        out[key] = {"pt": round(ap_v1 - ap_full, 6), **pct(dist[key])}
    diff = dist["a"] - dist["b"]
    out["diff"] = {"pt": round(out["a"]["pt"] - out["b"]["pt"], 6), **pct(diff)}
    out["_protocol"] = {"n_units": n, "n_boot": n_boot, "seed": seed,
                        "paired": True, "unit": "image",
                        "note": "Δ_occ = mAP(v=1) − mAP(full)，a=YOLO11s-pose，b=round5-3"}
    return out


# ---------- Part 2: κ CI ----------

def collect_phi(gt_by_img, preds_by_img):
    """复刻 evaluate_m2 的代偿段：逐实例 (img_id, φ_gt>τ, φ_pred>τ)。"""
    inst = {t: [] for t in TAUS}
    for img_id in set(gt_by_img) | set(preds_by_img):
        anns = gt_by_img.get(img_id, [])
        dets = preds_by_img.get(img_id, [])
        if not anns or not dets:
            continue
        dets = sorted(dets, key=lambda d: -d["score"])[:MAX_DETS]
        order = sorted(range(len(anns)), key=lambda i: anns[i]["_ignore"])
        anns_s = [anns[i] for i in order]
        ious, _ = compute_oks(anns_s, dets)
        dtm = match_greedy(anns_s, ious, 0.5)
        for dind, gind in dtm.items():
            if anns_s[gind]["_ignore"]:
                continue
            gt_kp = np.asarray(anns_s[gind]["keypoints"], dtype=float).reshape(-1, 3)
            pred_kp = np.asarray(dets[dind]["keypoints"], dtype=float).reshape(-1, 3)
            for knee in ("left_knee", "right_knee"):
                hip = KNEE_HIP[knee]
                v = JOINTS[knee]["vertex"]
                a_i = JOINTS[knee]["dist"]
                idxs = (hip, v, a_i)
                if not (kpts_valid_gt(gt_kp, idxs) and kpts_valid_pred(pred_kp, idxs)):
                    continue
                phi_gt = angle_deg(gt_kp[hip, :2], gt_kp[v, :2], gt_kp[a_i, :2])
                phi_pred = angle_deg(pred_kp[hip, :2], pred_kp[v, :2], pred_kp[a_i, :2])
                if phi_gt is None or phi_pred is None:
                    continue
                for tau in TAUS:
                    inst[tau].append((img_id, phi_gt > tau, phi_pred > tau))
    return inst


def group_by_img(inst):
    by_img = {}
    for img_id, gp, pp in inst:
        by_img.setdefault(img_id, []).append((gp, pp))
    img_ids = sorted(by_img)
    return by_img, img_ids


def agg_confusion(by_img, idx):
    tp = fp = fn = tn = 0
    for i in idx:
        for gp, pp in by_img[i]:
            if gp and pp:
                tp += 1
            elif gp and not pp:
                fn += 1
            elif not gp and pp:
                fp += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def kappa_bootstrap(inst_a, inst_b, n_boot, seed):
    """两模型独立图像级重采样：κ / 灵敏度 / 特异度 逐模型 CI + 模型差 CI。"""
    by_a, ids_a = group_by_img(inst_a)
    by_b, ids_b = group_by_img(inst_b)
    na, nb = len(ids_a), len(ids_b)
    rng = np.random.default_rng(seed)
    stats = ("kappa", "sens", "spec")
    dist = {m: {s: np.empty(n_boot) for s in stats} for m in ("a", "b")}
    ids_arr_a = np.asarray(ids_a, dtype=np.int64)
    ids_arr_b = np.asarray(ids_b, dtype=np.int64)
    for it in range(n_boot):
        ia = ids_arr_a[rng.integers(0, na, size=na)].tolist()
        ib = ids_arr_b[rng.integers(0, nb, size=nb)].tolist()
        for key, by, idx in (("a", by_a, ia), ("b", by_b, ib)):
            tp, fp, fn, tn = agg_confusion(by, idx)
            dist[key]["kappa"][it] = cohen_kappa(tp, fp, fn, tn)
            dist[key]["sens"][it] = tp / (tp + fn) if tp + fn else np.nan
            dist[key]["spec"][it] = tn / (tn + fp) if tn + fp else np.nan
    out = {}
    for key, by, ids in (("a", by_a, ids_a), ("b", by_b, ids_b)):
        tp, fp, fn, tn = agg_confusion(by, ids)
        out[key] = {
            "kappa": {"pt": round(cohen_kappa(tp, fp, fn, tn), 6),
                      **pct(dist[key]["kappa"])},
            "sens": {"pt": round(tp / (tp + fn), 6) if tp + fn else None,
                     **pct(dist[key]["sens"])},
            "spec": {"pt": round(tn / (tn + fp), 6) if tn + fp else None,
                     **pct(dist[key]["spec"])},
            "n_img": len(ids),
        }
    for s in stats:
        out["diff_" + s] = {"pt": round(out["a"][s]["pt"] - out["b"][s]["pt"], 6),
                            **pct(dist["a"][s] - dist["b"][s])}
    return out


# ---------- Part 3: M3 域对比 CI ----------

def domain_seg_units(segments):
    """逐段三域 d/j 单元（段由原始匹配固定；域只是坐标替换）。"""
    raw, px, q = [], [], []
    for seg in segments:
        raw.append(segment_units([seg])[0])
        px.append(segment_units(apply_repl([seg], pixel_replacements([seg])))[0])
        q.append(segment_units(apply_repl([seg], smooth_replacements([seg])))[0])
    return raw, px, q


def domain_bootstrap(units_raw, units_px, units_q, n_boot, seed):
    """段级 bootstrap：域间配对（同 idx），模型差由独立重采样分布元素级相减。"""
    packs = {"raw": pack_m3(units_raw), "smoothed_px": pack_m3(units_px),
             "smoothed": pack_m3(units_q)}
    n = len(units_raw)
    rng = np.random.default_rng(seed)
    per_dom = {d: {m: np.empty(n_boot) for m in M3_METRICS} for d in DOMAINS}
    for it in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for d in DOMAINS:
            o = m3_overall(packs[d], idx)
            for m in M3_METRICS:
                per_dom[d][m][it] = o[m]
    idx0 = np.arange(n)
    out = {}
    for pair, dom in (("px-raw", "smoothed_px"), ("q-raw", "smoothed")):
        out[pair] = {}
        for m in M3_METRICS:
            pt_dom = m3_overall(packs[dom], idx0)[m]
            pt_raw = m3_overall(packs["raw"], idx0)[m]
            out[pair][m] = {"pt": round(pt_dom - pt_raw, 6),
                            **pct(per_dom[dom][m] - per_dom["raw"][m])}
    out["per_domain"] = {}
    for d in DOMAINS:
        out["per_domain"][d] = {m: {"pt": round(m3_overall(packs[d], idx0)[m], 6)}
                                for m in M3_METRICS}
    out["_protocol"] = {"n_units": n, "n_boot": n_boot, "seed": seed,
                        "paired": True, "unit": "track segment",
                        "note": "域间同一重采样下标（配对）；模型间见 between_models"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    out = {"protocol": {"n_boot": args.n_boot, "seed": SEED, "alpha": ALPHA,
                        "ci": "95% 百分位 bootstrap"}}

    # Part 1：Δ_occ
    print("[1/3] Δ_occ bootstrap", flush=True)
    out["m1_delta_occ"] = {}
    for ds in M1_DATASETS:
        ua = load_units_npz(ROOT / "runs" / f"yolo11s-pose__{ds}" / "m1_units.npz")
        ub = load_units_npz(ROOT / "runs" / f"round5-3__{ds}" / "m1_units.npz")
        out["m1_delta_occ"][ds] = delta_occ_bootstrap(ua, ub, args.n_boot, SEED)
        print(f"  {ds}: Δ_occ diff pt={out['m1_delta_occ'][ds]['diff']['pt']}", flush=True)
    json.dump(out, open(ROOT / "runs" / "extra_ci.json", "w"), indent=1, ensure_ascii=False)

    # Part 2：κ
    print("[2/3] κ bootstrap", flush=True)
    out["m2_kappa"] = {}
    for ds in M1_DATASETS:
        out["m2_kappa"][ds] = {}
        for m in MODELS:
            gt, pr = load_run_gt_preds(f"{m}__{ds}")
            inst = collect_phi(gt, pr)
            if m == "round5-3":
                inst_b = inst
            else:
                inst_a = inst
        # 静态自检：全量混淆矩阵必须复现 metrics.json
        mm_b = json.load(open(ROOT / "runs" / f"round5-3__{ds}" / "metrics.json"))
        ref_b = mm_b["m2"]["compensation"]["taus"]
        for tau in TAUS:
            by, ids = group_by_img(inst_b[tau])
            tp, fp, fn, tn = agg_confusion(by, ids)
            t = f"{tau:g}"
            assert (tp, fp, fn, tn) == (ref_b[t]["tp"], ref_b[t]["fp"],
                                        ref_b[t]["fn"], ref_b[t]["tn"]), \
                f"κ 自检失败 {ds} r5-3 τ={tau}"
            out["m2_kappa"][ds][t] = kappa_bootstrap(inst_a[tau], inst_b[tau],
                                                     args.n_boot, SEED)
            print(f"  {ds} τ={tau}: κ diff pt={out['m2_kappa'][ds][t]['diff_kappa']['pt']}",
                  flush=True)
    json.dump(out, open(ROOT / "runs" / "extra_ci.json", "w"), indent=1, ensure_ascii=False)

    # Part 3：M3 域对比
    print("[3/3] M3 domain bootstrap", flush=True)
    out["m3_domains"] = {}
    per_model = {}
    for m in MODELS:
        run_dir = ROOT / "runs" / f"{m}__posetrack-val"
        meta = json.load(open(run_dir / "meta.json"))
        cfg = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))
        gt_json = cfg["datasets"][meta["dataset"]]["gt_json"]
        data = json.load(open(gt_json))
        im_meta = {im["id"]: (im["vid_id"], im["frame"], im["is_labeled"])
                   for im in data["images"]}
        gt_by_img = load_gt_coco(gt_json)
        preds = json.load(open(run_dir / "preds.json"))
        preds_by_img = {}
        for p in preds:
            preds_by_img.setdefault(p["image_id"], []).append(p)
        segments = build_segments(gt_by_img, preds_by_img, im_meta)
        print(f"  {m}: {len(segments)} 段，预计算三域单元", flush=True)
        raw_u, px_u, q_u = domain_seg_units(segments)
        res = domain_bootstrap(raw_u, px_u, q_u, args.n_boot, SEED)
        # 静态自检：全量聚合必须复现 metrics.json（1e-6）
        mm = json.load(open(run_dir / "metrics.json"))
        packs = {d: pack_m3(u) for d, u in
                 (("raw", raw_u), ("smoothed_px", px_u), ("smoothed", q_u))}
        for d in DOMAINS:
            o = m3_overall(packs[d], np.arange(len(raw_u)))
            ref = mm["m3"][d]["overall"]
            for metric in M3_METRICS:
                assert abs(o[metric] - ref[metric]) < 1e-6, f"M3 自检失败 {m}/{d}/{metric}"
        print(f"  {m}: 静态自检通过（raw/px/q 与 metrics.json 一致）", flush=True)
        per_model[m] = {"px-raw": res["px-raw"], "q-raw": res["q-raw"],
                        "per_domain": res["per_domain"],
                        "n_segments": len(segments)}
    out["m3_domains"]["within_model"] = per_model

    # 模型间对比（独立重采样分布元素级相减）需单独跑：此处用每模型逐域 bootstrap 分布
    print("[3/3] M3 between-model bootstrap", flush=True)
    between = {}
    for m in MODELS:
        run_dir = ROOT / "runs" / f"{m}__posetrack-val"
        mm = json.load(open(run_dir / "metrics.json"))
        meta = json.load(open(run_dir / "meta.json"))
        cfg = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))
        gt_json = cfg["datasets"][meta["dataset"]]["gt_json"]
        data = json.load(open(gt_json))
        im_meta = {im["id"]: (im["vid_id"], im["frame"], im["is_labeled"])
                   for im in data["images"]}
        gt_by_img = load_gt_coco(gt_json)
        preds = json.load(open(run_dir / "preds.json"))
        preds_by_img = {}
        for p in preds:
            preds_by_img.setdefault(p["image_id"], []).append(p)
        segments = build_segments(gt_by_img, preds_by_img, im_meta)
        raw_u, px_u, q_u = domain_seg_units(segments)
        packs = {"raw": pack_m3(raw_u), "smoothed_px": pack_m3(px_u),
                 "smoothed": pack_m3(q_u)}
        n = len(raw_u)
        rng = np.random.default_rng(SEED + (0 if m == "round5-3" else 1000))
        dist = {d: {mt: np.empty(args.n_boot) for mt in M3_METRICS} for d in DOMAINS}
        for it in range(args.n_boot):
            idx = rng.integers(0, n, size=n)
            for d in DOMAINS:
                o = m3_overall(packs[d], idx)
                for mt in M3_METRICS:
                    dist[d][mt][it] = o[mt]
        idx0 = np.arange(n)
        between[m] = {"dist": {d: {mt: dist[d][mt] for mt in M3_METRICS}
                               for d in DOMAINS},
                      "pt": {d: {mt: round(m3_overall(packs[d], idx0)[mt], 6)
                                 for mt in M3_METRICS} for d in DOMAINS},
                      "n_segments": n}
    out["m3_domains"]["between_models"] = {}
    for d in DOMAINS:
        out["m3_domains"]["between_models"][d] = {}
        for mt in M3_METRICS:
            da = between["yolo11s-pose"]["dist"][d][mt]
            db = between["round5-3"]["dist"][d][mt]
            pt = (between["yolo11s-pose"]["pt"][d][mt]
                  - between["round5-3"]["pt"][d][mt])
            out["m3_domains"]["between_models"][d][mt] = {
                "pt": round(pt, 6), **pct(da - db)}
    print(f"  between_models raw disp_mean: "
          f"{out['m3_domains']['between_models']['raw']['disp_mean']}", flush=True)

    with open(ROOT / "runs" / "extra_ci.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("done → runs/extra_ci.json", flush=True)


if __name__ == "__main__":
    main()
