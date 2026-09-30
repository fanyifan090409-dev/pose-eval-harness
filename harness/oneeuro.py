#!/usr/bin/env python3
"""因果 OneEuro 滤波（Casiez 默认参数，已核验作者官网 gery.casiez.net/1euro）。

G1 §3.3：离线对 p_t 施加因果 OneEuro（归一化坐标空间），参数固定：
    min_cutoff = 1.0, beta = 0.007, d_cutoff = 1.0
纯 NumPy 实现，只接受数组序列，逐帧因果（不回看未来）。

实现依据：作者官网 filters.js 的 LowPassFilter 主循环——dx 对「上一帧滤波值」差分、
导数低通后 cutoff = min_cutoff + beta·|edx|、主低通 x_hat = x_prev + α·(x − x_prev)；
α = 1/(1 + τ/te)，τ = 1/(2π·cutoff)（te=1 逐帧）。
"""
import math

import numpy as np


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    """α = 1 / (1 + τ/t_e)，τ = 1/(2π·cutoff)，等价 r=2π·cutoff·t_e, α=r/(1+r)。"""
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def one_euro_filter(values, min_cutoff=1.0, beta=0.007, d_cutoff=1.0, t_e=1.0):
    """对一维序列做因果 OneEuro 平滑，返回同长数组。

    首帧原样返回（状态初始化 x_prev=x_0, dx_prev=0，与 filters.js 一致）。
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values.copy()
    alpha_d = _smoothing_factor(t_e, d_cutoff)
    out = np.empty_like(values)
    x_prev = values[0]
    dx_prev = 0.0
    out[0] = x_prev
    for i in range(1, len(values)):
        dx = values[i] - x_prev
        edx = alpha_d * dx + (1 - alpha_d) * dx_prev
        cutoff = min_cutoff + beta * abs(edx)
        alpha = _smoothing_factor(t_e, cutoff)
        x_hat = x_prev + alpha * (values[i] - x_prev)
        out[i] = x_hat
        x_prev = x_hat
        dx_prev = edx
    return out


if __name__ == "__main__":
    # 自检：常数序列不变；阶跃帧输出 ∈(0,1) 且单调收敛（α<1 的低通，不回看）；
    # 因果性：只改最后一帧输入，前面的输出不变
    const = one_euro_filter(np.ones(10))
    assert np.allclose(const, 1.0), "常数序列应原样"
    step = one_euro_filter(np.concatenate([np.zeros(5), np.ones(5)]))
    assert step[4] == 0.0, "阶跃前帧应保持 0"
    assert 0 < step[5] < 1.0, "阶跃帧输出应在 (0,1)（低通 α<1）"
    assert np.all(np.diff(step[5:]) > 0), "阶跃后应单调上升收敛"
    assert step[-1] < 1.0, "有限帧内不应完全到达 1"
    seq = np.arange(10, dtype=np.float64)
    out_a = one_euro_filter(seq)
    out_b = one_euro_filter(np.append(seq[:-1], 999.0))
    assert np.allclose(out_a[:-1], out_b[:-1]), "因果性被破坏（回看了未来）"
    print("oneeuro 自检通过")
