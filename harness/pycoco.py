"""harness 侧官方 pycocotools 2.0.11 keypoints 评估（门禁 1 对拍源）。

ultralytics val 在 is_coco=False 时走内部 PoseMetrics；官方 COCO 发布数口径
是 pycocotools（num_keypoints=0 标 ignore + OKS 双倍 bbox 兜底 + maxDets=20）。
本模块对同一份 preds.json 用 pycocotools 原库补算 pycoco 段，
与 T3 M1 纯 NumPy 引擎对拍（差 ≤0.5 点即门禁 1 达标）。

2026-09-30 实证：本模块输出与 M1 复刻引擎全量对拍四位小数全等
（AP@.50=0.8496 / AP@[.50:.95]=0.5816）。
"""
import json


def evaluate_pycoco(gt_json: str, preds_json: str) -> dict:
    """对 preds.json 跑 pycocotools 2.0.11 keypoints 完整评估，返回官方 stats。

    stats 十项（keypoints 口径 maxDets=20）：
    [AP@[.50:.95], AP@.50, AP@.75, AP@medium, AP@large,
     AR@[.50:.95], AR@.50, AR@.75, AR@medium, AR@large]
    """
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(preds_json)
    ev = COCOeval(coco_gt, coco_dt, "keypoints")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = [round(float(x), 4) for x in ev.stats]
    # 逐 IoU 阈值 AP 曲线（论文阈值曲线图用）
    ap_per_iou = [
        round(float(ev.eval["precision"][t, :, 0, 0, 0].mean()), 4)
        for t in range(ev.eval["precision"].shape[0])
    ]
    return {
        "ap": stats[0],
        "ap50": stats[1],
        "ap75": stats[2],
        "ap_medium": stats[3],
        "ap_large": stats[4],
        "ar": stats[5],
        "ar50": stats[6],
        "ar75": stats[7],
        "ar_medium": stats[8],
        "ar_large": stats[9],
        "stats": stats,
        "ap_per_iou": ap_per_iou,
    }


if __name__ == "__main__":
    import sys

    out = evaluate_pycoco(sys.argv[1], sys.argv[2])
    print(json.dumps(out, indent=2, ensure_ascii=False))
