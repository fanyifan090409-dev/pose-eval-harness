"""T5 M3 引擎：PoseTrack 轨迹 + 抖动三指标 + OneEuro。

G1 §3 契约 + PoseTrack val 三个事实：
- 逐帧独立推理（无追踪），预测与 GT 按全关键点 OKS 匈牙利匹配；
- PoseTrack val 只标 30 连续帧且每 4 帧抽标 → frame_id 间隔 >1 必须断段；
  只统计连续 ≥10 帧的轨迹段；
- GT v 第三位是占位符全 1（非遮挡）→ 不做 v 过滤；
- keypoints 51=17×3，关节映射以各 JSON 的 categories 为准；PoseTrack 15 关节
  → COCO17 共有关节映射（M3 只报可映射关节，论文声明）；
- person-scale 归一化：s_t = sqrt(w·h)（GT 人框）；
- 指标：帧间位移 d_t、抖动 j_t（因果 5 帧中值滤波残差）、跳变率（d_t>τ，τ∈{0.05,0.10,0.20}）；
- OneEuro 前后重算（参数见 oneeuro.py）+ 平滑后全量 OKS mAP 变化。

输出：metrics.json 的 m3 段。
"""


def main():
    raise NotImplementedError("T5 实现")


if __name__ == "__main__":
    main()
