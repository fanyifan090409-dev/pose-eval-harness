"""T7 报告层：JSON 落盘 + md 表格 + 三图。

G1 §5 契约：
- metrics.json：全指标原始值（schema 见 G2 规划 §2，G6 作图契约）；
- report.md：M1 表（全量 + 三层分层 + 层占比 + Δ_occ）、M2 表（四关节 RMSE/MAE/P95
  按 v 分层 + 代偿一致性 τ 敏感性）、M3 表（三指标原始 vs OneEuro + 平滑后 mAP）；
- 三图：分层 mAP 柱状（两模型并排）、角度误差分布、抖动时序曲线示例。

matplotlib 出图（py310 已有）。
"""


def main():
    raise NotImplementedError("T7 实现")


if __name__ == "__main__":
    main()
