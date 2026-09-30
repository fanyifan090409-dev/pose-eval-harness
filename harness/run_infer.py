#!/usr/bin/env python3
"""run_infer.py — T2 推断层：单次 val(save_json=True) 双算

每个 (模型, 数据集) 一次 model.val() 产出（runs/<model>__<dataset>/）：
- preds.json：predictions.json 副本（COCO 格式，keypoints 51 值平铺、原图坐标）
- official.json：两个口径段
  - posemetrics：ultralytics val 内部 PoseMetrics 指标（is_coco=False 引擎，
    与历史 bench 同口径）
  - pycoco：harness 侧 pycocotools 2.0.11 对同一份 preds.json 的官方发布数口径
    （仅数据集配置了 gt_json 时产出；门禁 1 对拍源，M1 纯 NumPy 引擎差 ≤0.5 点）
- meta.json：运行元信息（权重/数据 yaml/gt json/ultralytics 版本 + commit/时间）

口径说明（关键业务规则，2026-09-30 实证）：
- ultralytics 判 is_coco 只看 data.yaml 的 val 是否以 coco/val2017.txt 结尾；
  本 harness 全部用 is_coco=False 布局 → val 指标走内部 PoseMetrics
- 官方发布数口径（pycocotools keypoints）与 PoseMetrics 差 20+ 点的根源：
  pycocotools 把 num_keypoints=0 的 GT 标 ignore（不参与 npig、匹配上也不算 tp/fp）、
  OKS 对这类 GT 用双倍 bbox 距离兜底、maxDets=20
- 所以 official.json 必须双段记录，论文表格注明口径

用法：
    python harness/run_infer.py --model yolo11s-pose --dataset coco-val --device 0
    python harness/run_infer.py --model round5-3 --dataset crowdpose-val
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def git_commit(pkg_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(pkg_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["round5-3", "yolo11s-pose"])
    ap.add_argument("--dataset", required=True,
                    choices=["coco-val", "crowdpose-val", "ochuman-test"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    models_cfg = load_yaml(ROOT / "configs/models.yaml")
    datasets_cfg = load_yaml(ROOT / "configs/datasets.yaml")
    weights = models_cfg["models"][args.model]["weights"]
    ds_cfg = datasets_cfg["datasets"][args.dataset]
    data_yaml = ds_cfg["data_yaml"]
    gt_json = ds_cfg.get("gt_json")

    from ultralytics import YOLO
    import ultralytics

    run_id = f"{args.model}__{args.dataset}"
    save_dir = RUNS / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        save_json=True,
        project=str(RUNS),
        name=run_id,
        exist_ok=True,
    )

    results = metrics.results_dict
    # 只保留论文关心的指标键，其余（速度等）不进官方记录
    keep = [k for k in results if k.startswith("metrics/") or k == "fitness"]
    official = {"posemetrics": {k: results[k] for k in keep}}

    pred_src = save_dir / "predictions.json"
    if not pred_src.is_file():
        print(f"ERROR: {pred_src} 未生成，save_json 未生效", file=sys.stderr)
        sys.exit(1)
    shutil.move(str(pred_src), str(save_dir / "preds.json"))

    if gt_json:
        # pycoco 段：官方发布数口径，对同一份 preds 补算（门禁 1 对拍源）
        sys.path.insert(0, str(ROOT))
        from harness.pycoco import evaluate_pycoco

        official["pycoco"] = evaluate_pycoco(gt_json, str(save_dir / "preds.json"))

    pkg_root = Path(ultralytics.__file__).resolve().parents[1]
    meta = {
        "run_id": run_id,
        "model": args.model,
        "dataset": args.dataset,
        "weights": weights,
        "data_yaml": str(data_yaml),
        "gt_json": gt_json,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_path": str(pkg_root),
        "ultralytics_commit": git_commit(pkg_root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(save_dir / "official.json", "w") as f:
        json.dump(official, f, indent=2, ensure_ascii=False)
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"official.json: {json.dumps(official, ensure_ascii=False)}")
    print(f"run dir: {save_dir}")


if __name__ == "__main__":
    main()
