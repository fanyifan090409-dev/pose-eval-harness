#!/usr/bin/env python3
"""T3 M1 引擎：pycocotools 2.0.11 keypoints 协议复刻 + v 分层 AP（G1 §1）。

协议核心（2026-09-30 实证，与 pycocotools 2.0.11 原库全量对拍四位小数全等）：
- _prepare：num_keypoints=0 的 GT 标 ignore（不只是 iscrowd）
- computeOks：k1=0 的 GT 用双倍 bbox 距离兜底；k1>0 只对 vg>0 关键点取均值
- evaluateImg：gt 按 ignore 升序、dt 按 score 降序（mergesort 稳定序）截 top-20；
  匹配循环带 ignore 断点；匹配到 ignore GT → dtIg（tp/fp 都不计）
- accumulate：全图 det 合并按 score 全局稳定排序 → 逐阈值 cumsum → rc=tp/npig、
  pr=tp/(tp+fp) 后向平滑 → recThrs 101 点 searchsorted（越界保 0）→ 均值

分层口径（G1 §1.2，门禁 3：匹配逻辑不动）：
- 每对 (det, GT) 的 17 维逐关键点贡献 d_i=exp(-e_i) 全量保留
- 层 L 的 OKS_L = mean(d_i : vg_i ∈ L)；人员匹配仍按全关键点 OKS
- 层内 AP：匹配对且该 GT 有 L 层关键点 → OKS_L ≥ t 记 tp 否则 fp；
  匹配对但无 L 层关键点 → 本层不计（dtIg 语义）；未匹配 det → fp（与全量一致）
- Δ_occ = mAP(v=1 层) − mAP(全量)；v=0 层只报告不排名（G1 §1.1）

用法：python harness/stratified_map.py --run runs/<run_id>
输出：metrics.json 的 m1 段 + 门禁 1 对拍（vs official.json pycoco 段）打印
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
VARS = (SIGMAS * 2) ** 2
K = 17
MAX_DETS = 20  # pycocotools keypoints Params.maxDets = [20]
IOU_T = np.linspace(0.5, 0.95, 10)
LAYERS = (2, 1, 0)  # G1 §1.1：v=2 可见 / v=1 遮挡 / v=0 缺失（v=0 只报告不排名）


def compute_oks(gt, dts):
    """复刻 pycocotools 2.0.11 computeOks。返回 (ious, d_all)。

    ious: (D, G) 逐对 OKS；d_all: (D, G, 17) 逐关键点贡献 d_i=exp(-e_i)（分层用）。
    """
    ious = np.zeros((len(dts), len(gt)))
    d_all = np.zeros((len(dts), len(gt), K))
    for j, g in enumerate(gt):
        gk = np.array(g["keypoints"], dtype=np.float64)
        xg, yg, vg = gk[0::3], gk[1::3], gk[2::3]
        k1 = int(np.count_nonzero(vg > 0))
        bb = g["bbox"]
        x0, x1 = bb[0] - bb[2], bb[0] + bb[2] * 2
        y0, y1 = bb[1] - bb[3], bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            dk = np.array(dt["keypoints"], dtype=np.float64)
            xd, yd = dk[0::3], dk[1::3]
            if k1 > 0:
                dx, dy = xd - xg, yd - yg
            else:
                # 无可见关键点时测 det 到双倍 bbox 区域的最近距离（官方分支）
                z = np.zeros(K)
                dx = np.maximum(z, x0 - xd) + np.maximum(z, xd - x1)
                dy = np.maximum(z, y0 - yd) + np.maximum(z, yd - y1)
            e = (dx**2 + dy**2) / VARS / (g["area"] + np.spacing(1)) / 2
            d = np.exp(-e)
            d_all[i, j] = d
            if k1 > 0:
                d = d[vg > 0]  # 官方 OKS 只对 vg>0 关键点取均值
            ious[i, j] = d.mean()
    return ious, d_all


def match_greedy(gt, ious, t):
    """复刻 evaluateImg 匹配循环。gt 已按 _ignore 升序。返回 {dind: gind}。"""
    iscrowd = [g["iscrowd"] for g in gt]
    gtIg = [g["_ignore"] for g in gt]
    gtm, dtm = {}, {}
    for dind in range(len(ious)):
        iou = min(t, 1 - 1e-10)
        m = -1
        for gind in range(len(gt)):
            if gind in gtm and not iscrowd[gind]:
                continue
            # 已匹配到非 ignore GT 后遇到 ignore GT → 停止（官方断点）
            if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                break
            if ious[dind, gind] < iou:
                continue
            iou = ious[dind, gind]
            m = gind
        if m == -1:
            continue
        dtm[dind] = m
        if not iscrowd[m]:
            gtm[m] = dind
    return dtm


def ap_from_rows(rows, n_gt):
    """复刻 accumulate：全局稳定排序 → 逐阈值 cumsum → 101 点 precision 平均。"""
    rows.sort(key=lambda r: -r[0])
    tp = np.array([r[1] for r in rows], dtype=bool)
    fp = np.array([r[2] for r in rows], dtype=bool)
    aps = []
    for ti in range(10):
        tpc = np.cumsum(tp[:, ti])
        fpc = np.cumsum(fp[:, ti])
        rc = tpc / n_gt
        pr = tpc / (tpc + fpc + np.spacing(1))
        pr = pr.tolist()
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        inds = np.searchsorted(rc, np.linspace(0, 1, 101), side="left")
        q = np.zeros(101)
        for ri, pi in enumerate(inds):
            if pi < len(pr):
                q[ri] = pr[pi]
        aps.append(float(q.mean()))
    return np.array(aps)


def match_rows_by_img(gt_by_img, preds_by_img):
    """逐图像匹配（evaluate_m1 的匹配循环唯一源，T6 bootstrap 复用）。

    返回 {img_id: (n_gt, key_counts, person_counts, rows_full, rows_layer)}：
    - n_gt：该图非 ignore GT 数（recall 分母分量）
    - key_counts/person_counts：{L: 数}，层关键点数 / 含层关键点的 GT 人数
    - rows_full：[(score, tp(10,bool), fp(10,bool))]；rows_layer：{L: 同结构}
    门禁 3：分层与全量共用本函数，匹配逻辑不可能分叉。
    """
    per_img = {}
    for img_id in set(gt_by_img) | set(preds_by_img):
        anns = gt_by_img.get(img_id, [])
        n_gt = sum(1 for a in anns if not a["_ignore"])
        key_counts = {L: 0 for L in LAYERS}
        person_counts = {L: 0 for L in LAYERS}
        for a in anns:
            if a["_ignore"]:
                continue
            vg = np.array(a["keypoints"], dtype=np.float64)[2::3]
            for L in LAYERS:
                mask = vg == L
                key_counts[L] += int(mask.sum())
                person_counts[L] += int(mask.any())
        dets = preds_by_img.get(img_id, [])
        if not dets:
            per_img[img_id] = (n_gt, key_counts, person_counts, [], {L: [] for L in LAYERS})
            continue
        dets = sorted(dets, key=lambda d: -d["score"])[:MAX_DETS]
        # 复刻 evaluateImg：gt 按 ignore 升序（ignore 在后），ious 列同步重排
        order = sorted(range(len(anns)), key=lambda i: anns[i]["_ignore"])
        anns_sorted = [anns[i] for i in order]
        ious, d_all = compute_oks(anns_sorted, dets)

        tp_f = np.zeros((len(dets), 10), dtype=bool)
        fp_f = np.zeros((len(dets), 10), dtype=bool)
        tp_l = {L: np.zeros((len(dets), 10), dtype=bool) for L in LAYERS}
        fp_l = {L: np.zeros((len(dets), 10), dtype=bool) for L in LAYERS}
        vg_by_gt = [np.array(a["keypoints"], dtype=np.float64)[2::3] for a in anns_sorted]

        for ti, t in enumerate(IOU_T):
            dtm = match_greedy(anns_sorted, ious, t)
            for dind in range(len(dets)):
                if dind in dtm:
                    gind = dtm[dind]
                    if anns_sorted[gind]["_ignore"]:
                        continue  # dtIg：tp/fp 都不计（官方语义）
                    tp_f[dind, ti] = True
                    for L in LAYERS:
                        mask = vg_by_gt[gind] == L
                        if mask.any():
                            oks_l = float(d_all[dind, gind, mask].mean())
                            if oks_l >= t:
                                tp_l[L][dind, ti] = True
                            else:
                                fp_l[L][dind, ti] = True
                        # 该 GT 无 L 层关键点：本层不计（dtIg 语义）
                else:
                    fp_f[dind, ti] = True
                    for L in LAYERS:
                        fp_l[L][dind, ti] = True

        rows_full = [(d["score"], tp_f[di], fp_f[di]) for di, d in enumerate(dets)]
        rows_layer = {L: [(d["score"], tp_l[L][di], fp_l[L][di]) for di, d in enumerate(dets)]
                      for L in LAYERS}
        per_img[img_id] = (n_gt, key_counts, person_counts, rows_full, rows_layer)
    return per_img


def evaluate_m1(gt_by_img, preds_by_img):
    """全量 + v 分层。返回 (full_aps, layer_aps, gt_fracs, gt_person_fracs, n_gt)。"""
    per_img = match_rows_by_img(gt_by_img, preds_by_img)
    rows_full = []
    rows_layer = {L: [] for L in LAYERS}
    n_gt = 0
    key_counts = {L: 0 for L in LAYERS}
    person_counts = {L: 0 for L in LAYERS}
    total_keys = 0

    for img_id, (n_i, kc, pc, rf, rl) in per_img.items():
        n_gt += n_i
        rows_full.extend(rf)
        for L in LAYERS:
            rows_layer[L].extend(rl[L])
            key_counts[L] += kc[L]
            person_counts[L] += pc[L]
        total_keys += sum(kc.values())

    full_aps = ap_from_rows(rows_full, n_gt)
    layer_aps = {L: ap_from_rows(rows_layer[L], n_gt) for L in LAYERS}
    gt_fracs = {L: key_counts[L] / total_keys if total_keys else 0.0 for L in LAYERS}
    gt_person_fracs = {L: person_counts[L] / n_gt if n_gt else 0.0 for L in LAYERS}
    return full_aps, layer_aps, gt_fracs, gt_person_fracs, n_gt


def load_gt_coco(json_path):
    """读 COCO 格式 GT json，按 image_id 分组并打 _ignore 标记（官方 _prepare 语义）。"""
    data = json.load(open(json_path))
    gt_by_img = {}
    for a in data["annotations"]:
        # num_keypoints=0 也标 ignore（pycocotools 2.0.11 keypoints 专属规则）
        a["_ignore"] = 1 if (a["iscrowd"] or a["num_keypoints"] == 0) else 0
        gt_by_img.setdefault(a["image_id"], []).append(a)
    return gt_by_img


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
        print(f"数据集 {meta['dataset']} 未配置 gt_json，M1 暂不支持（T3.1 接入 crowdpose/ochuman）",
              file=sys.stderr)
        sys.exit(0)

    gt_by_img = load_gt_coco(gt_json)
    preds = json.load(open(run_dir / "preds.json"))
    preds_by_img = {}
    for d in preds:
        preds_by_img.setdefault(d["image_id"], []).append(d)

    full_aps, layer_aps, gt_fracs, gt_person_fracs, n_gt = evaluate_m1(gt_by_img, preds_by_img)

    official = json.load(open(run_dir / "official.json"))
    pycoco = official.get("pycoco")
    gate1_diff = None
    if pycoco:
        gate1_diff = round(abs(full_aps.mean() - pycoco["ap"]), 4)
        print(f"门禁 1 对拍：harness AP {full_aps.mean():.4f} vs 官方 pycoco {pycoco['ap']:.4f} "
              f"差 {gate1_diff:.4f} 点（≤0.5 达标）" if gate1_diff <= 0.5 else
              f"差 {gate1_diff:.4f} 点（超 0.5 门禁！）")

    m1 = {
        "protocol": "pycocotools-2.0.11",
        "full": {
            "ap": round(float(full_aps.mean()), 4),
            "ap50": round(float(full_aps[0]), 4),
            "ap_per_iou": [round(float(x), 4) for x in full_aps],
        },
        "layers": {
            f"v{L}": {
                "ap": round(float(layer_aps[L].mean()), 4),
                "ap50": round(float(layer_aps[L][0]), 4),
                "gt_frac": round(gt_fracs[L], 4),
                # 有该层关键点的 GT 人数占比 = 该层 recall 天花板
                # （CrowdPose v2 层因 head/neck 映射裁剪仅 0.59，AP 系统性低估需论文注明）
                "gt_person_frac": round(gt_person_fracs[L], 4),
                "low_sample": bool(gt_fracs[L] < 0.01),  # G1：占比 <1% 样本不足仅参考
            }
            for L in LAYERS
        },
        "delta_occ": None,
        "n_gt": n_gt,
        "gate1_diff": gate1_diff,
    }
    # Δ_occ = mAP(v=1 层) − mAP(全量)（G1 §1.3）；无 v=1 标注的数据集为 null
    if gt_fracs[1] > 0:
        m1["delta_occ"] = round(float(layer_aps[1].mean() - full_aps.mean()), 4)

    metrics_path = run_dir / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.is_file() else {}
    metrics["run_id"] = args.run
    metrics["m1"] = m1
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(m1, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
