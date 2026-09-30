"""T3 M1 引擎：OKS 匹配复刻 + v 分层 AP 累积。

G1 §1 契约：
- 先精读 ultralytics metrics.py（8.4.128）确认 OKS sigmas / v=0 处理 / 匹配顺序，
  逐行对照复刻官方全量 AP；
- 分层实现不改变匹配逻辑（门禁 3）：人员匹配仍按全关键点 OKS，
  只在统计时把逐关键点贡献 d_i 按 GT v 归层累计；
- 每层报告 mAP50-95(P) + 该层 GT 关键点占比 gt_frac；
- Δ_occ = mAP(v=1 层) − mAP(全量)。

输入：preds.json + GT 标注（各数据集 YOLO/JSON）
输出：metrics.json 的 m1 段（schema 见 G2 规划 §2）
纯 NumPy/SciPy（Hungarian 用 scipy.optimize.linear_sum_assignment）。
"""


def main():
    raise NotImplementedError("T3 实现")


if __name__ == "__main__":
    main()
