#!/usr/bin/env python3
"""T4 M2 引擎：关节角 RMSE + 代偿检出一致性（G1 §2）。

契约要点：
- 关节（COCO17 索引，0-based）：左肘 7(5,9) / 右肘 8(6,10) / 左膝 13(11,15) / 右膝 14(12,16)
- 内角公式：a=p近−p顶, b=p远−p顶, θ=arccos(a·b/(|a||b|))；|a| 或 |b|<ε 剔除
- 有效性掩码（G1 §2.2）：GT 角度=三点 GT v>0；预测角度=同三点 GT v>0 且 pred 置信度>0
  （preds.json 的 keypoints 第三位是置信度，GT 第三位是 v，两者语义不同）
- 人员匹配：与 M1 相同口径（全关键点 OKS 贪心、gt 按 ignore 升序），单档 t=0.5
  （G1 §2 未写死阈值，2026-09-30 补注：M2 在 t=0.5 匹配对域上计算）
- 误差 e=θ_pred−θ_gt，RMSE/MAE/P95(|e|) + ±5° 桶直方图；按顶点关节 GT v 分层（v=2/v=1）
- 代偿检出（G1 §2.3）：膝内扣角 φ=髋→膝 与 髋→踝 的图像平面夹角（同侧髋，左 11/右 12）；
  三点 GT v>0 且 pred conf>0 才计；φ 对 τ 二值化（默认 15°，敏感性 10°/17°），
  报告灵敏度/特异度/Cohen's κ（κ 自实现），不报绝对检出率

用法：python harness/angles.py --run runs/<run_id>
输出：metrics.json 的 m2 段
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 直接 python harness/angles.py 运行时 harness 包不在默认路径
from harness.stratified_map import MAX_DETS, compute_oks, load_gt_coco, match_greedy

JOINTS = {
    "left_elbow": {"vertex": 7, "prox": 5, "dist": 9},
    "right_elbow": {"vertex": 8, "prox": 6, "dist": 10},
    "left_knee": {"vertex": 13, "prox": 11, "dist": 15},
    "right_knee": {"vertex": 14, "prox": 12, "dist": 16},
}
KNEE_HIP = {"left_knee": 11, "right_knee": 12}
EPS = 1e-6
MATCH_T = 0.5  # 与 M1 同匹配逻辑，单档阈值（G1 §2 补注 2026-09-30）
TAUS = (15.0, 10.0, 17.0)  # 默认 15°，敏感性 10°/17°（G1 §2.3）
HIST_EDGES = np.arange(-180, 181, 5)  # ±5° 桶，72 桶


def angle_deg(vertex, p1, p2):
    """以 vertex 为顶点，p1/p2 为射线端点算内角（度）。退化时返回 None。"""
    a = np.asarray(p1, dtype=float) - vertex
    b = np.asarray(p2, dtype=float) - vertex
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return None
    c = np.clip(float(a @ b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def cohen_kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def kpts_valid_gt(gt_kp, idxs):
    """GT 侧有效性：idx 全部 v>0 且坐标有限。"""
    k = gt_kp.reshape(-1, 3)
    for i in idxs:
        x, y, v = k[i]
        if not (v > 0 and np.isfinite(x) and np.isfinite(y)):
            return False
    return True


def kpts_valid_pred(pred_kp, idxs):
    """预测侧有效性：idx 全部置信度>0 且坐标有限。"""
    k = pred_kp.reshape(-1, 3)
    for i in idxs:
        x, y, c = k[i]
        if not (c > 0 and np.isfinite(x) and np.isfinite(y)):
            return False
    return True


def evaluate_m2(gt_by_img, preds_by_img):
    """在 t=0.5 匹配对域上计算角度误差 + 代偿检出一致性。

    返回 (angles, compensation)，angles 结构：
      {joint: {"v2": {"errs": [...], "abs_errs": [...]}, "v1": {...}}}
    compensation：{tau: {"tp":.., "fp":.., "fn":.., "tn":..}}（双膝合并）
    """
    angles = {j: {"v2": {"errs": [], "abs": []}, "v1": {"errs": [], "abs": []}} for j in JOINTS}
    comp = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in TAUS}
    n_pairs = 0

    for img_id in set(gt_by_img) | set(preds_by_img):
        anns = gt_by_img.get(img_id, [])
        dets = preds_by_img.get(img_id, [])
        if not anns or not dets:
            continue
        dets = sorted(dets, key=lambda d: -d["score"])[:MAX_DETS]
        order = sorted(range(len(anns)), key=lambda i: anns[i]["_ignore"])
        anns_s = [anns[i] for i in order]
        ious, _ = compute_oks(anns_s, dets)
        dtm = match_greedy(anns_s, ious, MATCH_T)
        for dind, gind in dtm.items():
            if anns_s[gind]["_ignore"]:
                continue
            n_pairs += 1
            gt_kp = np.asarray(anns_s[gind]["keypoints"], dtype=float).reshape(-1, 3)
            pred_kp = np.asarray(dets[dind]["keypoints"], dtype=float).reshape(-1, 3)

            for jname, j in JOINTS.items():
                idxs = (j["prox"], j["vertex"], j["dist"])
                if not (kpts_valid_gt(gt_kp, idxs) and kpts_valid_pred(pred_kp, idxs)):
                    continue
                vbin = "v2" if gt_kp[j["vertex"], 2] == 2 else "v1"
                t_gt = angle_deg(gt_kp[j["vertex"], :2], gt_kp[j["prox"], :2], gt_kp[j["dist"], :2])
                t_pred = angle_deg(pred_kp[j["vertex"], :2], pred_kp[j["prox"], :2], pred_kp[j["dist"], :2])
                if t_gt is None or t_pred is None:
                    continue
                e = t_pred - t_gt
                angles[jname][vbin]["errs"].append(e)
                angles[jname][vbin]["abs"].append(abs(e))

            # 代偿：双膝各自按同侧髋→膝/髋→踝算 φ，双膝合并进混淆矩阵
            for knee in ("left_knee", "right_knee"):
                hip = KNEE_HIP[knee]
                v = JOINTS[knee]["vertex"]
                a_i = JOINTS[knee]["dist"]  # 踝
                idxs = (hip, v, a_i)
                if not (kpts_valid_gt(gt_kp, idxs) and kpts_valid_pred(pred_kp, idxs)):
                    continue
                phi_gt = angle_deg(gt_kp[hip, :2], gt_kp[v, :2], gt_kp[a_i, :2])
                phi_pred = angle_deg(pred_kp[hip, :2], pred_kp[v, :2], pred_kp[a_i, :2])
                if phi_gt is None or phi_pred is None:
                    continue
                for tau in TAUS:
                    gp = phi_gt > tau
                    pp = phi_pred > tau
                    if gp and pp:
                        comp[tau]["tp"] += 1
                    elif gp and not pp:
                        comp[tau]["fn"] += 1
                    elif not gp and pp:
                        comp[tau]["fp"] += 1
                    else:
                        comp[tau]["tn"] += 1
    return angles, comp, n_pairs


def summarize(angles, comp, n_pairs):
    m2_angles = {}
    for jname in JOINTS:
        m2_angles[jname] = {}
        for vbin in ("v2", "v1"):
            errs = np.array(angles[jname][vbin]["errs"])
            abs_e = np.array(angles[jname][vbin]["abs"])
            n = len(errs)
            m2_angles[jname][vbin] = {
                "n": int(n),
                "rmse": round(float(np.sqrt(np.mean(errs**2))), 4) if n else None,
                "mae": round(float(np.mean(abs_e)), 4) if n else None,
                "p95": round(float(np.percentile(abs_e, 95)), 4) if n else None,
                # 直方图（±5° 桶，72 桶）供 T7 作图；报告层再画
                "hist": np.histogram(errs, bins=HIST_EDGES)[0].astype(int).tolist() if n else [],
            }
    m2_comp = {}
    for tau in TAUS:
        c = comp[tau]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        n = tp + fp + fn + tn
        m2_comp[f"{tau:g}"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
            "cohen_kappa": round(cohen_kappa(tp, fp, fn, tn), 4) if cohen_kappa(tp, fp, fn, tn) is not None else None,
        }
    return {
        "matching": {"t": MATCH_T, "n_pairs": n_pairs,
                     "note": "全关键点 OKS 贪心，口径同 M1；单档 t=0.5（G1 §2 补注）"},
        "angles": m2_angles,
        "compensation": {
            "definition": "膝内扣角 φ=髋→膝 与 髋→踝 图像平面夹角（同侧髋），双膝合并",
            "taus": m2_comp,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<run_id> 目录名")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    meta = json.load(open(run_dir / "meta.json"))
    datasets_cfg = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))
    gt_json = datasets_cfg["datasets"][meta["dataset"]].get("gt_json")
    if not gt_json:
        print(f"数据集 {meta['dataset']} 未配置 gt_json", file=sys.stderr)
        sys.exit(1)

    gt_by_img = load_gt_coco(gt_json)
    preds = json.load(open(run_dir / "preds.json"))
    preds_by_img = {}
    for d in preds:
        preds_by_img.setdefault(d["image_id"], []).append(d)

    angles, comp, n_pairs = evaluate_m2(gt_by_img, preds_by_img)
    m2 = summarize(angles, comp, n_pairs)

    metrics_path = run_dir / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.is_file() else {}
    metrics["run_id"] = args.run
    metrics["m2"] = m2
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(m2, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
