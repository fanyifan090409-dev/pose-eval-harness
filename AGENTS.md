# eval-harness 目录约定

## 定位

论文 A G2 实现。只做「读预测 + 读 GT → 算指标」，不改 ultralytics 源码、不动训练数据。

## 与 G1 契约的对应

| 本目录模块 | G1 章节 | 回答 |
|---|---|---|
| run_infer.py + pycoco.py | §1.2 官方口径 | 官方全量 mAP 双口径段（门禁 1） |
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

## 口径事实（2026-09-30 实证，改动前必读）

- 官方发布数口径 = pycocotools 2.0.11 keypoints：`_prepare` 把 **num_keypoints=0 的 GT 标 ignore**
  （不只是 iscrowd）；computeOks 对 k1=0 的 GT 用双倍 bbox 距离兜底；maxDets=[20]
- ultralytics 内部 PoseMetrics 无上述处理 → 同一模型两口径差 20+ 点
- 复刻引擎（M1 核心）与 pycocotools 原库全量对拍四位小数全等（0.8496/0.5816）

## 数据集接入事实（2026-09-30 实证）

- ultralytics save_json 的 preds image_id = int(文件 stem)、category_id 取数据集类别 id（实测为 1）→ GT json 的 image id 必须与文件 stem 对齐
- CrowdPose 官方 val json 是 14 点、无 area 字段；build_gt17.py 转换（与训练同源）：
  MAP 0-11→COCO 5-16，head_top/neck 丢弃（14570 v=1 + 2484 v=0 随之丢弃，非 v 破坏），
  面部 5 点 v=0 假点，越界 v>0 降 v=0（仅 8 点触发），area=bbox 宽高，num_keypoints 重算。
  转换后 v 分布 0/1/2 = 80898/37020/27041，与训练标签严格一致
- OCHuman test json 本身 17 点直通；image id 重映射 = int(文件名 stem)；缺失图 001287.jpg
  及其 1 个标注剔除（2230 图 / 3818 实例）
- pycoco.py 的 catIds 取自 GT（person=1）；preds category_id=1 实测一致，无需改写

## 红线（G1 继承）

- 零解剖先验内容；round6 模型不进基准；只用公开数据集；校园视频不进 A
- PoseTrack val 三个事实：帧间隔 >1 必须断段；GT v 第三位是占位符全 1 不做过滤；keypoints 51=17×3 以各 JSON categories 为准
