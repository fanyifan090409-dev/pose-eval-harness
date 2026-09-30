# 复现清单（REPRODUCIBILITY）

本文是 eval-harness 的可复现性契约：按本文步骤可在干净机器上重建全部指标。
G2 完成时间 2026-10-01，commit 链 3f08e64 → 10bce88（T1-T8 每任务一 commit）。

## 1. 环境

- Ubuntu 24.04，conda env `py310`（python 3.10；关键依赖：numpy 2.2.6、pycocotools 2.0.11、scipy 1.15.3、matplotlib、pyyaml、ultralytics 8.3.x）
- 推断需 NVIDIA GPU（本实验 2× RTX 5090，ultralytics val 约数分钟/数据集）；M1/M2/M3/统计/报告层纯 CPU
- 代码位置：`/home/ubuntu/isef/eval-harness`（独立 git repo，不依赖 ultralytics-dev）

## 2. 模型权重

| 模型 | 路径 | 来源 |
|---|---|---|
| round5-3（数据轮最优） | `/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt` | 本实验训练（谱系见 lab-notebook） |
| yolo11s-pose（官方基线） | `/home/ubuntu/isef/weights/yolo11s-pose.pt` | ultralytics 官方发布权重（yolo11s-pose.pt，GitHub release） |

两模型推断配置（run_infer.py 内固定）：imgsz=640、batch=16、conf 默认、single_cls 按数据集数据类别数（COCO 系 7 类检测头为产品口径，论文注明）。

## 3. 数据

全部公开数据集，GT json 由 build 脚本从官方标注转换（转换规则见各脚本 docstring 与 G1 协议 §1.1）：

| 数据集 | 图片 | GT json（configs/datasets.yaml 指向，全部在 datasets 树内） | 生成脚本 |
|---|---|---|---|
| COCO val | `/home/ubuntu/isef/datasets/coco2017/val2017/` | `/home/ubuntu/isef/datasets/coco2017/annotations/person_keypoints_val2017.json`（官方原样） | — |
| CrowdPose val | `/home/ubuntu/isef/datasets/crowdpose/images/` | `/home/ubuntu/isef/datasets/crowdpose/crowdpose_annotations/crowdpose_val_gt17.json` | harness/build_gt17.py |
| OCHuman test | `/home/ubuntu/isef/datasets/ochuman/images/` | `/home/ubuntu/isef/datasets/ochuman/ochuman_test_gt.json` | harness/build_gt17.py |
| PoseTrack val | `/home/ubuntu/isef/datasets/posetrack18/`（硬链接场） | `/home/ubuntu/isef/datasets/posetrack18/posetrack_data/annotations/posetrack_val_merged.json` | harness/build_posetrack.py |

PoseTrack 硬链接场规则（重建时必读 build_posetrack.py docstring）：文件名 stem `{vid:06d}{frame:04d}` 数值化 → image_id = vid*10000+frame 全局唯一（绕开 ultralytics save_json 逐视频 stem 撞车）；170 视频 20161 图，标注帧 8923。

## 4. 复现步骤（以 coco-val 为例，其余数据集同理）

```bash
cd /home/ubuntu/isef/eval-harness
PY=/home/ubuntu/Miniconda3/envs/py310/bin/python

# ① 推断（单次推断双算：official.json 双口径 + preds.json）——已有 runs/ 产物可跳过
$PY harness/run_infer.py --weights /home/ubuntu/isef/runs/pose-round5-3/weights/best.pt \
  --dataset coco-val --device 0
# ② M1 遮挡分层（门禁 1 对拍打印在 stdout）
$PY harness/stratified_map.py --run runs/round5-3__coco-val
# ③ M2 角度误差 + 单元落盘
$PY harness/angles.py --run runs/round5-3__coco-val
# ④ M3 时序（PoseTrack）——两个 posetrack run 都跑
$PY harness/temporal.py --run runs/round5-3__posetrack-val
# ⑤ 统计层（M1/M2/M3 bootstrap，10k；需两模型 run 都已有）
$PY harness/statistics.py --dataset coco-val --run-a runs/yolo11s-pose__coco-val \
  --run-b runs/round5-3__coco-val
# ⑥ 报告层（读全部 runs + stats → md/图/summary.json）
$PY harness/report.py
# ⑦ 三门禁验收（门禁 1 会重算 coco-val 两模型全量 mAP 对拍官方）
$PY tests/test_gates.py
```

门禁定义（G1 §6）：①全量 mAP 与 official.json pycoco 段差 ≤0.5 点；②合成自检（GT 角度代预测 RMSE=0、常数轨迹抖动=0）；③分层不改匹配（合成全 v=2 数据 v2 层 AP ≡ 全量 AP）。

## 5. 关键口径（阅读代码前必读，全部有实证出处）

- 官方口径 = pycocotools 2.0.11 keypoints 协议（num_keypoints=0 的 GT 标 ignore、k1=0 双倍 bbox 兜底、maxDets=20），不是 ultralytics 内部 PoseMetrics——两者差 20+ 点。official.json 双段存档（posemetrics + pycoco）。
- preds.json 覆盖图集 = val 全图集，超出 GT 注释图（coco-val：4830/4956 pred 图 vs 2693 GT 注释图）；无注释图 dt-only 照计 FP——与 pycoco 口径一致，故门禁 1 对拍仍为 0.0000。
- M1 配对 universe = sorted(GT ∪ preds_a ∪ preds_b)（两模型预测图集不同）；M2/M3 单元集随匹配结果不同 → 独立重采样（非配对）。
- 层 AP 的 recall 分母 = 全部 GT 人员；每层报告 gt_person_frac（有该层关键点的 GT 人数占比）作召回天花板证据。CrowdPose v2 层受 14→17 映射裁剪（head_top/neck 未映射）gt_person_frac 仅 0.59，AP 系统性低估——论文须注明。
- M3 有效性掩码 = GT v>0 且 pred conf>0；PoseTrack v=0 即 (0,0) 占位（34.4% 关键点）；帧间隔>1 断段；段长 ≥10 才统计。
- OneEuro 双口径：smoothed（q 域，G1 契约）与 smoothed_px（像素域，对照）——q 域反归一化重注入 GT bbox 尺度噪声，像素域滤波零 AP 代价（论文主口径写作期定）。
- 门禁/测试合成数据 keypoints 契约为平铺 [x,y,v]×17（COCO json 原生格式），非嵌套列表。

## 6. 产物索引

- 每 run 目录：official.json（posemetrics+pycoco）、preds.json、metrics.json（m1/m2/m3 段）、m1_units.npz / m2_units.json / m3_units.json（统计单元）
- runs/stats__<ds>.json：bootstrap CI（run_a=yolo11s-pose、run_b=round5-3，Δ 方向 = a−b）
- runs/reports/：report_<ds>.md + fig_m1/fig_m2 各 3 张 + fig_m3_posetrack.png + summary.json（schema_version=1，契约见 harness/report.py docstring）

## 7. 归档

训练机 → 外接硬盘（Mac /Volumes/Expansion/姿态数据集/）：原始数据在 原始压缩包/ 与各数据集中文目录；本 repo 与 runs 产物在 实验记录/eval-harness/；训练产物各轮 results/weights 在 训练产物/。归档日志 ~/isef-data/backup-archive.log。
