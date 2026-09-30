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

## 红线（G1 继承）

- 零解剖先验内容；round6 模型不进基准；只用公开数据集；校园视频不进 A
- PoseTrack val 数据事实（2026-10-01 实证，改动前必读）：
  - 推断集 = 硬链接场 images/posetrack_val_flat/，stem `{vid:06d}{frame:04d}` → int(stem) =
    vid*10000+frame 全局唯一 = GT image_id（ultralytics save_json 数值化 stem 规则对齐）
  - GT 标注帧结构 = 步长 4 稀疏区 + 中间步长 1 密集窗口 → 帧间隔 >1 必须断段
    （GT 段 ≥10 帧共 993 条、中位长 31）
  - GT v∈{0,1}：**v=0 即 (0,0) 占位（34.4% 关键点）**，有效性掩码 = GT v>0 且 pred conf>0
    （旧说法「v 全 1 不做过滤」已被实证推翻）；1030 个 ann 无 bbox（全 v=0 空壳）已跳过
  - 标签文件必须与图片 stem 逐字符一致（img2label_path 只换扩展名）；v=1→2、v=0→0
- M3 平滑双口径（2026-10-01 实证）：`smoothed` = q 域（G1 契约，q_t=p_t/s_t 滤波后反归一化）；
  `smoothed_px` = 对照（OneEuro 直接施于像素坐标）。q 域反归一化会重注入 GT bbox 尺度噪声
  （|s_t/s_{t-1}-1| median 0.97%/P95 8.1%）→ 抖动指标不降反升；像素域则全面下降。论文主口径待定
