# eval-harness 目录约定

## 定位

论文 A G2 实现。只做「读预测 + 读 GT → 算指标」，不改 ultralytics 源码、不动训练数据。

## 与 G1 契约的对应

| 本目录模块 | G1 章节 | 回答 |
|---|---|---|
| run_infer.py | §1.2 官方口径 | 官方全量 mAP（门禁 1） |
| stratified_map.py | §1 分层口径 + Δ_occ | RQ1 |
| angles.py | §2 关节角 + 代偿检出一致性 | RQ2 |
| temporal.py + oneeuro.py | §3 三指标 + OneEuro 前后 | RQ3 |
| statistics.py | §4 bootstrap/Spearman/Kendall | RQ4 |
| report.py | §5 报告产物 JSON schema | G6 输入 |

## 编码约定

- Python 3.10（py310 环境），零新增依赖（numpy/scipy/pandas/matplotlib/pycocotools/ultralytics 已有）
- 指标引擎（stratified_map/angles/temporal/statistics）必须是纯 NumPy/SciPy 函数，输入只认 JSON/数组 → 可单测
- 关键业务规则写中文批注：v 分层口径、有效性掩码、person-scale 归一化、匹配逻辑不动只分层统计
- 每个任务的 commit 用 Conventional Commits（feat/fix/test/docs）

## 红线（G1 继承）

- 零解剖先验内容；round6 模型不进基准；只用公开数据集；校园视频不进 A
- PoseTrack val 三个事实：帧间隔 >1 必须断段；GT v 第三位是占位符全 1 不做过滤；keypoints 51=17×3 以各 JSON categories 为准
