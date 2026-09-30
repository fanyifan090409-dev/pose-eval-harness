#!/usr/bin/env python3
"""build_posetrack.py — PoseTrack18 val 170 视频 → 合并 GT JSON + YOLO 布局推断集

动机（ultralytics save_json 撞车问题）：
- pred_to_json 的 image_id = int(stem) if stem.isnumeric() else stem；
- PoseTrack 每视频帧名都是 000000.jpg…000151.jpg → stem 逐视频重复，无法对齐 GT。
- 解法：硬链接场 images/posetrack_val_flat/{vid}{frame:04d}.jpg（vid 6 位、frame ≤151
  必 <10000）→ stem 全局唯一且数值化 → image_id = int(vid)*10000+frame。
- GT 合并 JSON 用同一 id 规则，temporal.py 按 image_id 对齐 preds 与 GT。

PoseTrack val 标注布局（2026-10-01 实证，改动前必读）：
- 每视频独立 COCO JSON：images 含全帧（58-151 帧、帧号连续），is_labeled 稀疏标注；
- 标注帧结构 = 步长 4 稀疏区 + 中间步长 1 密集窗口（如 000342：0,4,…,32 然后
  34-64 逐帧然后 68,72,…,96）→ 帧间隔 >1 断段规则的真实依据，只有步长 1 密集
  窗口能产出 ≥10 连续帧轨迹段；
- GT v∈{0,1}：v=0 即 (0,0) 占位（34.4% 关键点，无坐标），v=1 必有坐标 →
  YOLO 标签转换 v=1→2（可见）、v=0→0；M3 有效性掩码必须 v>0；
- 图片尺寸逐视频统一（7 种：1920x1080/1280x720/960x720/854x480/654x480/640x480/
  1280x684），逐视频抽一张 jpg 读 W/H 用于标签归一化；
- track_id 是视频内轨迹身份（GT 轨迹段的来源）；
- 缺图帧数 0，20161 帧全在盘。

输出：
- annotations/posetrack_val_merged.json（GT；id=int(vid)*10000+frame；保留 track_id/
  vid_id/frame/is_labeled/nframes；补 width/height/area/num_keypoints/iscrowd）
- images/posetrack_val_flat/*.jpg（硬链接，零额外磁盘）
- labels/posetrack_val_flat/*.txt（YOLO pose 标签，未标注帧写空文件）

用法：python harness/build_posetrack.py（幂等：硬链接已存在则跳过）
"""
import json
import os
import re
from pathlib import Path

from PIL import Image

ROOT = Path("/home/ubuntu/isef/datasets/posetrack18/posetrack_data")
ANN_DIR = ROOT / "annotations" / "val"
IMG_DIR = Path("/home/ubuntu/isef/datasets/posetrack18") / "images" / "val"
FLAT_IMG = ROOT / "images" / "posetrack_val_flat"
FLAT_LBL = ROOT / "labels" / "posetrack_val_flat"
MERGED = ROOT / "annotations" / "posetrack_val_merged.json"

K = 17
FRAME_RE = re.compile(r"(\d+)\.jpg$")


def frame_of(file_name: str) -> int:
    return int(FRAME_RE.search(file_name).group(1))


def main():
    FLAT_IMG.mkdir(parents=True, exist_ok=True)
    FLAT_LBL.mkdir(parents=True, exist_ok=True)

    merged_imgs, merged_anns = [], []
    categories = None
    n_link = 0
    n_labeled = 0
    ann_id = 0
    for json_path in sorted(ANN_DIR.glob("*_mpii_test.json")):
        d = json.load(open(json_path))
        if categories is None:
            categories = d["categories"]
        vid = json_path.stem.replace("_mpii_test", "")  # 6 位字符串，如 "000342"
        vdir = IMG_DIR / f"{vid}_mpii_test"
        # 逐视频统一尺寸：读第一张 jpg（标签归一化用）
        W, H = Image.open(vdir / d["images"][0]["file_name"].split("/")[-1]).size

        # 原 image_id → (新 id, frame, 是否标注)
        id_map = {}
        for im in d["images"]:
            frame = frame_of(im["file_name"])
            new_id = int(vid) * 10000 + frame
            labeled = bool(im.get("is_labeled"))
            id_map[im["id"]] = (new_id, frame, labeled)
            src = vdir / f"{frame:06d}.jpg"
            dst = FLAT_IMG / f"{vid}{frame:04d}.jpg"
            if not dst.exists():
                os.link(src, dst)
                n_link += 1
            merged_imgs.append({
                "id": new_id,
                "vid_id": vid,
                "frame": frame,
                "file_name": str(dst),
                "width": W,
                "height": H,
                "is_labeled": labeled,
                "nframes": im.get("nframes"),
            })
            if labeled:
                n_labeled += 1

        # 标注按新 id 归组 → 写 YOLO 标签（v=1→2 可见；v=0 保持 0）
        lines_by_id = {}
        for a in d["annotations"]:
            # 无 bbox 的 1030 个实例全部是 17 点全 v=0 空壳（2026-10-01 实证），
            # 无可匹配信息（computeOks k1=0 分支也要 bbox），跳过
            if "bbox" not in a:
                continue
            new_id, frame, labeled = id_map[a["image_id"]]
            k = a["keypoints"]
            kstr = " ".join(
                f"{k[3 * i] / W:.6f} {k[3 * i + 1] / H:.6f} {2 if k[3 * i + 2] > 0 else 0}"
                for i in range(K)
            )
            x, y, w, h = a["bbox"]
            lines_by_id.setdefault(new_id, []).append(
                f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f} {kstr}"
            )
            merged_anns.append({
                "id": ann_id,
                "image_id": new_id,
                "category_id": a["category_id"],
                "track_id": a["track_id"],
                "bbox": a["bbox"],
                "area": float(a["bbox"][2] * a["bbox"][3]),
                "keypoints": k,
                "num_keypoints": int(sum(1 for i in range(K) if k[3 * i + 2] > 0)),
                "iscrowd": 0,
            })
            ann_id += 1

        # 每帧都写标签文件（含内容为空者）——ultralytics 要求逐图有 label 文件，
        # 空文件在缓存扫描里计 empty 正常处理
        # 文件名必须与图片 stem 逐字符一致（含前导零）：img2label_path 只换扩展名
        for im in d["images"]:
            new_id, _, _ = id_map[im["id"]]
            lines = lines_by_id.get(new_id, [])
            with open(FLAT_LBL / f"{vid}{frame_of(im['file_name']):04d}.txt", "w") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")

    merged = {
        "images": merged_imgs,
        "annotations": merged_anns,
        "categories": categories,
    }
    with open(MERGED, "w") as f:
        json.dump(merged, f)
    print(f"硬链接: {n_link} | 帧: {len(merged_imgs)} | 标注帧: {n_labeled} | 实例: {len(merged_anns)}")
    print(f"合并 GT: {MERGED}")


if __name__ == "__main__":
    main()
