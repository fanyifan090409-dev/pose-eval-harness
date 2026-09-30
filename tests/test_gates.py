#!/usr/bin/env python3
"""T8 门禁：G1 §6 三门禁的可执行验收（pytest 可收集，亦可直接 python 运行）。

门禁 1：harness 全量 mAP vs official.json pycoco 段 ≤0.5 点（coco-val 两模型真实重算）
门禁 2：合成自检——GT 角度代 GT 预测 RMSE=0、代偿混淆矩阵完美一致；
        常数轨迹抖动=0、跳变率=0；常数序列经 OneEuro 仍为常数
门禁 3：分层不改匹配——合成数据全 v=2 时 v2 层 AP 与全量 AP 逐阈值全等，
        v1/v0 层 AP=0；架构上 full 与分层共用 match_rows_by_img（stratified_map.py）

用法：python tests/test_gates.py 或 pytest tests/test_gates.py
退出码：全绿 0，任一失败 1。
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.angles import evaluate_m2
from harness.oneeuro import one_euro_filter
from harness.stratified_map import evaluate_m1, load_gt_coco
from harness.temporal import disp_jitter

GATE1_TOL = 0.5  # G1 §6：官方 vs harness 差 ≤0.5 点


# ---------- 合成数据构造 ----------

def _synth_ann(img_id, pts, v=2):
    """17 个 [x,y] 点 → COCO 格式 ann：keypoints 平铺为 [x,y,v]×17（引擎契约）。"""
    kps = [c for p in pts for c in (p[0], p[1], v)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return {"image_id": img_id, "category_id": 1, "keypoints": kps,
            "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
            "iscrowd": 0, "num_keypoints": 17, "_ignore": 0}


def _skeleton():
    """站姿骨架坐标（角度定义完整即可，不求解剖精确）。"""
    return [[300, 80], [290, 75], [310, 75], [280, 80], [320, 80],   # 0-4 头
            [280, 150], [320, 150], [250, 220], [350, 220],          # 5-8 肩/肘
            [220, 290], [380, 290], [270, 300], [330, 300],          # 9-12 腕/髋
            [260, 380], [340, 380], [250, 460], [350, 460]]          # 13-16 膝/踝


# ---------- 门禁 2：M2 角度合成自检 ----------

def test_gate2_m2_gt_as_pred():
    """GT 关键点代 GT 预测（conf=1）→ 四关节 RMSE=0、代偿混淆矩阵完美一致（κ=1）。"""
    gt_by_img, preds_by_img = {}, {}
    for img_id in (1, 2):
        kps = _skeleton()
        gt_by_img[img_id] = [_synth_ann(img_id, kps, v=2)]
        # 预测侧第三位语义是置信度，坐标与 GT 完全一致（平铺 51 长）
        pred_kps = [c for p in kps for c in (p[0], p[1], 1.0)]
        preds_by_img[img_id] = [{"image_id": img_id, "category_id": 1,
                                 "keypoints": pred_kps, "score": 0.99}]
    angles, comp, n_pairs = evaluate_m2(gt_by_img, preds_by_img)
    assert n_pairs == 2, f"匹配对应为 2，实得 {n_pairs}"
    for jname in ("left_elbow", "right_elbow", "left_knee", "right_knee"):
        for vb in ("v2", "v1"):
            n = len(angles[jname][vb]["errs"])
            if n:
                errs = np.array(angles[jname][vb]["errs"])
                assert np.max(np.abs(errs)) < 1e-9, f"{jname} {vb} 误差非零"
    for tau in (15.0, 10.0, 17.0):
        c = comp[tau]
        assert c["fp"] == 0 and c["fn"] == 0, f"τ={tau} 混淆矩阵不完美：{c}"
        assert c["tp"] + c["tn"] == 4, f"τ={tau} 应 4 个实例（2 人 × 双膝）"


# ---------- 门禁 2：M3 时序合成自检 ----------

def test_gate2_m3_constant_traj():
    """常数轨迹 → d=0、j=0（全部 τ 跳变率=0）。"""
    pos = np.tile([100.0, 200.0], (10, 1))
    sc = np.full(10, 120.0)
    d, j = disp_jitter(pos, sc)
    assert np.all(d == 0.0) and np.all(j == 0.0)
    for tau in (0.05, 0.10, 0.20):
        assert (d > tau).mean() == 0.0


def test_gate2_m3_oneeuro_constant():
    """常数序列经 OneEuro 滤波仍为常数（Casiez 默认参数）。"""
    x = np.full(20, 3.7)
    out = one_euro_filter(x, min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
    assert np.allclose(out, 3.7, atol=1e-9)


# ---------- 门禁 3：分层不改匹配（合成语义验证） ----------

def test_gate3_layer_equals_full():
    """全 v=2 合成数据 → v2 层 AP 与全量 AP 逐阈值全等；v1/v0 层 AP=0。"""
    gt_by_img, preds_by_img = {}, {}
    for img_id in (1, 2, 3):
        kps = _skeleton()
        gt_by_img[img_id] = [_synth_ann(img_id, kps, v=2)]
        match = [c for p in kps for c in (p[0], p[1], 1.0)]  # 精确复制 → OKS=1 必匹配
        far = [c for x, y in kps for c in (x + 500, y + 500, 0.9)]  # 平移 → 未匹配 FP
        preds_by_img[img_id] = [
            {"image_id": img_id, "category_id": 1, "keypoints": match, "score": 0.95},
            {"image_id": img_id, "category_id": 1, "keypoints": far, "score": 0.90},
        ]
    full_aps, layer_aps, *_ = evaluate_m1(gt_by_img, preds_by_img)
    assert np.allclose(full_aps, layer_aps[2]), \
        f"v2 层 AP 与全量不一致：{full_aps} vs {layer_aps[2]}"
    assert layer_aps[1].mean() == 0.0 and layer_aps[0].mean() == 0.0


# ---------- 门禁 1：真实数据重算对拍（coco-val 两模型） ----------

def test_gate1_coco_recompute():
    """harness 重算全量 mAP vs official.json pycoco 段 ≤0.5 点（G1 §6）。"""
    datasets_cfg = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))
    gt_json = datasets_cfg["datasets"]["coco-val"]["gt_json"]
    gt_by_img = load_gt_coco(gt_json)
    for model in ("round5-3", "yolo11s-pose"):
        run_dir = ROOT / "runs" / f"{model}__coco-val"
        preds = json.load(open(run_dir / "preds.json"))
        preds_by_img = {}
        for p in preds:
            preds_by_img.setdefault(p["image_id"], []).append(p)
        full_aps, *_ = evaluate_m1(gt_by_img, preds_by_img)
        pycoco = json.load(open(run_dir / "official.json"))["pycoco"]["ap"]
        diff = abs(float(full_aps.mean()) - pycoco)
        assert diff <= GATE1_TOL, \
            f"{model}: harness {full_aps.mean():.4f} vs pycoco {pycoco:.4f} 差 {diff:.4f}"


def main():
    tests = [test_gate2_m2_gt_as_pred, test_gate2_m3_constant_traj,
             test_gate2_m3_oneeuro_constant, test_gate3_layer_equals_full,
             test_gate1_coco_recompute]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'三门禁全绿' if failed == 0 else f'{failed} 项失败'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
