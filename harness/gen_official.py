#!/usr/bin/env python3
"""ViTPose 推断产物补 official.json（仅 pycoco 段，无 posemetrics）。

用法: python harness/gen_official.py --model vitpose-b [--dataset coco-val]
默认对该模型全部非 skip_pycoco 数据集生成。
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.pycoco import evaluate_pycoco

DS = yaml.safe_load(open(ROOT / "configs" / "datasets.yaml"))["datasets"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dataset", default=None)
ap.add_argument("--protocol", default="gt_bbox_oracle_topdown_udp")
args = ap.parse_args()

datasets = [args.dataset] if args.dataset else list(DS.keys())
for ds in datasets:
    cfg = DS[ds]
    run_dir = ROOT / "runs" / f"{args.model}__{ds}"
    if not (run_dir / "preds.json").is_file():
        print(f"!! 缺 {run_dir}/preds.json，跳过")
        sys.exit(1)
    if cfg.get("skip_pycoco"):
        official = {"protocol": args.protocol}
        with open(run_dir / "official.json", "w") as f:
            json.dump(official, f, indent=2, ensure_ascii=False)
        print(f"{args.model}__{ds} official.json 桩（skip_pycoco，无 pycoco 段）")
        continue
    pycoco = evaluate_pycoco(cfg["gt_json"], str(run_dir / "preds.json"))
    official = {"pycoco": pycoco, "protocol": args.protocol}
    with open(run_dir / "official.json", "w") as f:
        json.dump(official, f, indent=2, ensure_ascii=False)
    print(f"{args.model}__{ds} pycoco AP={pycoco['ap']:.4f} AP50={pycoco['ap50']:.4f}")
