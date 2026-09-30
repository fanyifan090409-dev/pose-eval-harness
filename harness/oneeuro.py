"""因果 OneEuro 滤波（Casiez 默认参数，已核验作者官网 gery.casiez.net/1euro）。

G1 §3.3：离线对 p_t 施加因果 OneEuro（归一化坐标空间），参数固定：
    min_cutoff = 1.0, beta = 0.007, d_cutoff = 1.0
纯 NumPy 实现，只接受数组序列，逐帧因果（不回看未来）。
"""


def one_euro_filter(values, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
    """对一维序列做因果 OneEuro 平滑，返回同长数组。"""
    raise NotImplementedError("T5 实现")


if __name__ == "__main__":
    raise NotImplementedError("T5 实现")
