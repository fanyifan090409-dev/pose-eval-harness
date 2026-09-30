# eval-harness — 论文 A 评估协议 harness

论文 A（评估协议篇）的评测执行层。上游契约：`docs/plans/论文A-G1评估协议.md`（Mac 侧），本仓库是 G2 的实现。

## 用法（一条命令跑全指标）

```bash
conda activate py310
cd ~/isef/eval-harness
python harness/run_infer.py --model round5-3 --dataset crowdpose-val   # 推断 + 官方指标
python harness/stratified_map.py --run runs/<run_id>                    # M1 分层 mAP
python harness/angles.py --run runs/<run_id>                            # M2 角度 + 代偿
python harness/temporal.py --run runs/<run_id>                          # M3 时序（PoseTrack）
python harness/statistics.py --run runs/<run_id>                        # bootstrap + 相关性
python harness/report.py --run runs/<run_id>                            # JSON + md + 图
```

## 架构

单次推断双算：`model.val(save_json=True)` 一次运行同时产出
①官方指标（`official.json`，双口径段）②预测结果（`preds.json`）。
M1/M2/M3 引擎纯离线读「预测 + GT」计算，不碰 GPU（可单测、可合成桩自检）。

## 运行产物（runs/，gitignore）

```
runs/<model>_<dataset>/
├── official.json   # 双口径：posemetrics 段（ultralytics 内部引擎）
│                   #        + pycoco 段（pycocotools 2.0.11 官方发布数口径，门禁 1 对拍源）
├── preds.json      # save_json 预测（M1/M2 引擎输入）
├── metrics.json    # 全部指标（G1 §5 JSON 契约，G6 作图用）
└── report.md       # 报告表格 + 图
```

## 口径（2026-09-30 实证）

官方发布数口径（pycocotools keypoints）与 ultralytics 内部 PoseMetrics 差 20+ 点，
根源：pycocotools 把 num_keypoints=0 的 GT 标 ignore（不参与 npig、匹配上也不算 tp/fp）、
OKS 对这类 GT 用双倍 bbox 距离兜底、maxDets=20。论文所有表格必须注明口径。

## 门禁（G1 §6，tests/test_gates.py）

1. 两模型全量 mAP50-95(P)：official.json 的 pycoco 段（pycocotools 2.0.11）vs M1 纯 NumPy 引擎 ≤0.5 点
2. 自检：GT 角度代预测 → RMSE=0；常数轨迹 → 抖动=0、跳变率=0
3. 分层实现不改变匹配逻辑（同模型全量 mAP 不因开启分层而变）
