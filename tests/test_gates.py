"""T8 门禁 + 单测（G1 §6）。

门禁 2 合成自检（tests/fixtures 提供小合成桩）：
- GT 角度代 GT 预测 → RMSE=0
- 常数轨迹 → 抖动=0、跳变率=0

门禁 1/3 对拍入口（需真实运行产物，跑在 runs/ 上）：
- 官方全量 mAP 与 harness 自算全量差 ≤0.5 点（COCO val）
- 开启分层不改变全量 mAP（同模型同预测）

全绿才宣布 G2 完成。
"""


def main():
    raise NotImplementedError("T8 实现")


if __name__ == "__main__":
    main()
