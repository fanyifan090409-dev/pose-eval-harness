#!/usr/bin/env python3
"""coco_to_yolo.py — person_keypoints_val2017.json → YOLO pose 标签（官方口径）

与 build_pose_dataset.py 的 build_coco 不同：本脚本零筛选、零抽样、无上限——
官方 coco-pose 转换口径（ultralytics json2yolo）是全量转换：
- 不过滤 num_keypoints（含 num_keypoints=0 的纯 bbox 标注，COCO 官方评测也计入）
- 不设每图 person 上限
- bbox 归一化、v 值原样保留（COCO v∈{0,1,2} 与 YOLO 语义一致，G1 已核验）
- 5000 张 val2017 每张都写标签文件（无标注写空文件），val 集恒为 5000 张

用法：python harness/coco_to_yolo.py
输出：~/isef/datasets/coco2017/labels/val2017/*.txt
"""
import json
import os

COCO_KPT = 17
ANN_PATH = "/home/ubuntu/isef/datasets/coco2017/annotations/person_keypoints_val2017.json"
OUT_DIR = "/home/ubuntu/isef/datasets/coco2017/labels/val2017"


def main():
    data = json.load(open(ANN_PATH))
    imgmap = {i["id"]: i for i in data["images"]}
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    os.makedirs(OUT_DIR, exist_ok=True)
    n_person = 0
    for iid, info in imgmap.items():
        W, H = info["width"], info["height"]
        lines = []
        for a in anns_by_img.get(iid, []):
            x, y, w, h = a["bbox"]
            k = a["keypoints"]
            kstr = " ".join(f"{k[3 * i] / W:.6f} {k[3 * i + 1] / H:.6f} {k[3 * i + 2]}" for i in range(COCO_KPT))
            lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f} {kstr}")
        n_person += len(lines)
        name = info["file_name"].rsplit(".", 1)[0] + ".txt"
        with open(os.path.join(OUT_DIR, name), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"images: {len(imgmap)}, persons: {n_person}")


if __name__ == "__main__":
    main()
