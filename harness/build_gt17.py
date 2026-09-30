#!/usr/bin/env python3
"""build_gt17.py — T3.1 数据集接入：官方 GT json → M1 可评的 17 点 COCO GT json。

两个转换，规则与训练期转换脚本（scripts/convert_crowdpose_val.py、
scripts/screen_ochuman.py）严格同源，保证「GT 语义 = 模型训练所见语义」：

crowdpose-val：mmpose_crowdpose_val.json（14 点）
  - 映射与训练一致：CrowdPose 0-11 → COCO 5-16；head_top(12)/neck(13) 丢弃
    （2026-09-30 实证：丢弃两点带走 14570 个 v=1 + 2484 个 v=0，非 v 破坏）
  - COCO 0-4 面部 = v=0 假点（x=y=0）；越界 v>0 → v=0（官方 json 仅 8 点触发）
  - 补 area = bbox w*h（源 json 缺失，pycocotools computeOks 需要）
  - num_keypoints 按转换后 17 点重算（pycocotools _prepare 的 ignore 规则用）

ochuman-test：ochuman_coco_format_test_range_0.00_1.00.json（17 点直通）
  - image id 重映射 = int(文件名 stem)：ultralytics save_json 的 preds image_id
    取 int(文件 stem)，而官方 json 的 id 是 1..2231 顺序号，两者不对应
  - 剔除 yolo-test/images 缺失的 1 张图（001287.jpg）及其标注

用法：python harness/build_gt17.py --dataset crowdpose-val|ochuman-test
"""
import argparse
import json
import os

import numpy as np

# CrowdPose 14 点（0-11 为肩→踝，12=head_top，13=neck）→ COCO 17 点
MAP14_17 = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13, 9: 14, 10: 15, 11: 16}

SRC = {
    "crowdpose-val": "/home/ubuntu/isef/datasets/crowdpose/crowdpose_annotations/mmpose_crowdpose_val.json",
    "ochuman-test": "/home/ubuntu/isef/datasets/ochuman/ochuman_coco_format_test_range_0.00_1.00.json",
}
DST = {
    "crowdpose-val": "/home/ubuntu/isef/datasets/crowdpose/crowdpose_annotations/crowdpose_val_gt17.json",
    "ochuman-test": "/home/ubuntu/isef/datasets/ochuman/ochuman_test_gt.json",
}
IMG_DIR = {
    "ochuman-test": "/home/ubuntu/isef/datasets/ochuman/yolo-test/images",
}


def convert_crowdpose(d):
    out = {k: v for k, v in d.items() if k != "annotations"}
    out["images"] = d["images"]
    imgs = {i["id"]: i for i in d["images"]}
    anns = []
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        w, h = im["width"], im["height"]
        kp = np.array(a["keypoints"], dtype=float).reshape(14, 3)
        k17 = np.zeros((17, 3))
        for src, dst in MAP14_17.items():
            px, py, v = kp[src]
            if v > 0 and (px < 0 or px > w or py < 0 or py > h):
                v = 0  # 与训练转换同规则：越界点不可评分
            k17[dst] = (px, py, v)
        x, y, bw, bh = a["bbox"]
        anns.append({
            "id": a["id"],
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "bbox": a["bbox"],
            "area": float(bw * bh),
            "iscrowd": a["iscrowd"],
            "keypoints": k17.reshape(-1).tolist(),
            "num_keypoints": int((k17[:, 2] > 0).sum()),
        })
    out["annotations"] = anns
    return out


def convert_ochuman(d):
    files = set(os.listdir(IMG_DIR["ochuman-test"]))
    old_to_new = {}
    images = []
    for i in d["images"]:
        if i["file_name"] in files:
            new_id = int(os.path.splitext(i["file_name"])[0])
            old_to_new[i["id"]] = new_id
            images.append({**i, "id": new_id})
    anns = []
    img_by_new = {i["id"]: i for i in images}
    for a in d["annotations"]:
        if a["image_id"] not in old_to_new:
            continue  # 图像缺失的标注一并剔除（001287.jpg）
        im = img_by_new[old_to_new[a["image_id"]]]
        w, h = im["width"], im["height"]
        kp = np.array(a["keypoints"], dtype=float).reshape(17, 3)
        for i in range(17):
            px, py, v = kp[i]
            if v > 0 and (np.isnan(px) or np.isnan(py) or px < 0 or py < 0 or px >= w or py >= h):
                kp[i, 2] = 0  # 与训练转换同规则（实测 test json 无触发点）
        anns.append({**a, "image_id": old_to_new[a["image_id"]], "keypoints": kp.reshape(-1).tolist()})
    out = {k: v for k, v in d.items() if k != "annotations"}
    out["images"] = images
    out["annotations"] = anns
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["crowdpose-val", "ochuman-test"])
    args = ap.parse_args()
    d = json.load(open(SRC[args.dataset]))
    out = convert_crowdpose(d) if args.dataset == "crowdpose-val" else convert_ochuman(d)
    with open(DST[args.dataset], "w") as f:
        json.dump(out, f)
    print(f"{args.dataset}: {len(out['images'])} 图 / {len(out['annotations'])} 实例 → {DST[args.dataset]}")


if __name__ == "__main__":
    main()
