"""T6 统计层：bootstrap CI + Spearman/Kendall。

G1 §4 契约：
- 模型比较：图像级（M1/M2）或轨迹级（M3）bootstrap 重采样 10,000 次，
  报告 Δ 的 95% 百分位 CI；CI 不含 0 判显著；
- 比较清单在实验前固定（2 模型 × 3 数据集 × 层/关节/指标），逐项报告 CI，
  清单全部如实列出（不挑结果）；
- RQ4 冗余性：三指标在模型×数据集×分层单元上的 Spearman 相关矩阵
  + 协议排名 vs 全量 OKS 排名的 Kendall τ。

纯 NumPy 实现；bootstrap 用固定种子保证可复现。
"""


def main():
    raise NotImplementedError("T6 实现")


if __name__ == "__main__":
    main()
