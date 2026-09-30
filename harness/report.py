#!/usr/bin/env python3
"""T7 报告层：读 runs → md 表格 + 三图 + summary.json（G6 作图契约）。

数据源（全部已有产物，零重算）：
- runs/<model>__<ds>/metrics.json：M1/M2/M3 原始值（各引擎落盘）
- runs/stats__<ds>.json：bootstrap CI（T6 落盘，run_a=yolo11s-pose、run_b=round5-3）
- runs/round5-3__posetrack-val/m3_units.json：逐段逐关键点 d/j（M3 示例曲线）

输出（runs/reports/）：
- report_<ds>.md：M1 表（全量+三层+Δ_occ+CI）、M2 表（四关节 × v2/v1 + 代偿一致性）、
  M3 表（八指标 raw vs OneEuro 双域 + 平滑 AP 代价）
- fig_m1_<ds>.png：分层 mAP 柱状图（两模型并排 + Δ CI 标注）
- fig_m2_<ds>.png：角度误差分布图（四关节 2×2，v=2 层两模型叠加）
- fig_m3_posetrack.png：抖动时序曲线示例（左膝最长 run，d_t/j_t 双面板）
- summary.json：全指标原始值（schema 见 SUMMARY_SCHEMA，G6 作图契约）

summary.json 契约（schema_version=1）：
{
  "schema_version": 1,
  "generated": "<ISO UTC>",
  "models": ["round5-3", "yolo11s-pose"],
  "run_pairs": {ds: ["yolo11s-pose__<ds>", "round5-3__<ds>"]},   # stats 口径 run_a/run_b
  "m1": {ds: {model: <metrics.json m1 段原样>, "bootstrap": {layer: <stats m1 段>}}},
  "m2": {ds: {model: <metrics.json m2 段原样>, "bootstrap": {joint: {vbin: <stats m2 段>}}}},
  "m3": {"posetrack-val": {model: <metrics.json m3 段原样>, "bootstrap": {metric: <stats m3 段>}}}
}
bootstrap 的 delta 方向一律 = run_a − run_b = yolo11s-pose − round5-3。

用法：python harness/report.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "reports"

MODELS = ["round5-3", "yolo11s-pose"]
LABEL = {"round5-3": "round5-3 (ours)", "yolo11s-pose": "yolo11s-pose (official)"}
DATASETS = ["coco-val", "crowdpose-val", "ochuman-test"]  # M1/M2 三数据集
POSETRACK = "posetrack-val"                                # M3 专属
M3_METRICS = ("disp_mean", "jitter_mean", "disp_p95", "jitter_p95",
              "jump_0.05", "jump_0.1", "jump_0.2")
JOINTS = ("left_elbow", "right_elbow", "left_knee", "right_knee")
HIST_EDGES = np.arange(-180, 181, 5)  # 与 angles.py HIST_EDGES 同口径（72 桶）


def load_json(p):
    with open(p) as f:
        return json.load(f)


def f4(x, signed=True):
    """4 位小数格式化；signed=True 时正数加 + 号。None 透传。"""
    if x is None:
        return "—"
    s = f"{x:.4f}"
    return f"+{s}" if signed and x > 0 else s


def fmt_ci(lo, hi):
    return f"[{f4(lo)}, {f4(hi)}]"


def load_all():
    """读 4 个 stats + 8 个 metrics + m3_units，返回组装好的数据结构。"""
    data = {"metrics": {}, "stats": {}, "m3_units": None}
    for ds in DATASETS:
        data["stats"][ds] = load_json(RUNS / f"stats__{ds}.json")
    data["stats"][POSETRACK] = load_json(RUNS / f"stats__{POSETRACK}.json")
    for ds in DATASETS + [POSETRACK]:
        for m in MODELS:
            data["metrics"][(ds, m)] = load_json(RUNS / f"{m}__{ds}/metrics.json")
    p = RUNS / f"round5-3__{POSETRACK}/m3_units.json"
    data["m3_units"] = load_json(p) if p.is_file() else None
    return data


# ---------- md 表格 ----------

def write_m1_table(f, ds, data):
    m_r = data["metrics"][(ds, "round5-3")]["m1"]
    m_y = data["metrics"][(ds, "yolo11s-pose")]["m1"]
    st = data["stats"][ds]["m1"]
    f.write(f"## M1 遮挡分层 mAP（pycocotools 2.0.11 口径）\n\n")
    f.write(f"门禁 1 对拍：round5-3 差 {m_r['gate1_diff']:.4f} / yolo11s-pose 差 {m_y['gate1_diff']:.4f} 点；"
            f"n_gt = {m_r['n_gt']}\n\n")
    f.write("| 层 | round5-3 | yolo11s-pose | Δ (y−r) | 95% CI | 显著 | 关键点占比 | GT 人占比 |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for layer in ("full", "v2", "v1", "v0"):
        ap_r = m_r["full"]["ap"] if layer == "full" else m_r["layers"][layer]["ap"]
        ap_y = m_y["full"]["ap"] if layer == "full" else m_y["layers"][layer]["ap"]
        b = st[layer]
        sig = "✓" if b["sig"] else "—"
        name = "全量" if layer == "full" else f"{layer} 层"
        if layer == "full":
            frac_cell, person_cell = "—", "—"
        else:
            fr = m_r["layers"][layer]
            frac_cell = f"{fr['gt_frac']:.4f}"
            person_cell = f"{fr['gt_person_frac']:.4f}"
            if fr.get("low_sample"):
                frac_cell += "（样本不足）"
        note = "" if layer != "v0" else "（不排名）"
        f.write(f"| {name}{note} | {ap_r:.4f} | {ap_y:.4f} | "
                f"{f4(b['point_delta'])} | {fmt_ci(b['ci_lo'], b['ci_hi'])} | {sig} | "
                f"{frac_cell} | {person_cell} |\n")
    f.write(f"\nΔ_occ = mAP(v=1 层) − mAP(全量)：round5-3 = {f4(m_r['delta_occ'])}，"
            f"yolo11s-pose = {f4(m_y['delta_occ'])}。\n\n")


def write_m2_table(f, ds, data):
    m_r = data["metrics"][(ds, "round5-3")]["m2"]
    m_y = data["metrics"][(ds, "yolo11s-pose")]["m2"]
    st = data["stats"][ds]["m2"]
    f.write(f"## M2 角度误差（匹配对域 t=0.5；round5-3 n_pairs={m_r['matching']['n_pairs']}，"
            f"yolo11s-pose n_pairs={m_y['matching']['n_pairs']}）\n\n")
    f.write("| 关节 | 层 | RMSE r/y | MAE r/y | P95 r/y | n r/y | ΔRMSE | 95% CI | 显著 |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for j in JOINTS:
        for vb in ("v2", "v1"):
            ar = m_r["angles"][j][vb]
            ay = m_y["angles"][j][vb]
            b = st[j][vb]
            f.write(f"| {j} | {vb} | {ar['rmse']:.2f} / {ay['rmse']:.2f} | "
                    f"{ar['mae']:.2f} / {ay['mae']:.2f} | {ar['p95']:.2f} / {ay['p95']:.2f} | "
                    f"{ar['n']} / {ay['n']} | {f4(b['point_delta'], signed=False)} | "
                    f"{fmt_ci(b['ci_lo'], b['ci_hi'])} | {'✓' if b['sig'] else '—'} |\n")
    f.write("\n### 代偿检出一致性（膝内扣 φ 二值化 vs GT，灵敏度/特异度/κ）\n\n")
    f.write("| τ | 模型 | 灵敏度 | 特异度 | Cohen's κ |\n|---|---|---|---|---|\n")
    for tau in ("15", "10", "17"):
        for m in MODELS:
            c = data["metrics"][(ds, m)]["m2"]["compensation"]["taus"][tau]
            f.write(f"| {tau}° | {LABEL[m]} | {c['sensitivity']:.4f} | "
                    f"{c['specificity']:.4f} | {c['cohen_kappa']:.4f} |\n")
    f.write("\n")


def write_m3_table(f, data):
    m_r = data["metrics"][(POSETRACK, "round5-3")]["m3"]
    m_y = data["metrics"][(POSETRACK, "yolo11s-pose")]["m3"]
    st = data["stats"][POSETRACK]["m3"]
    f.write(f"## M3 时序抖动（轨迹段 round5-3 {m_r['segments']['n']} 条 / "
            f"yolo11s-pose {m_y['segments']['n']} 条）\n\n")
    f.write("### 原始域（未平滑）\n\n")
    f.write("| 指标 | round5-3 | yolo11s-pose | Δ (y−r) | 95% CI | 显著 |\n|---|---|---|---|---|---|\n")
    for m in M3_METRICS:
        vr = m_r["raw"]["overall"][m]
        vy = m_y["raw"]["overall"][m]
        b = st[m]
        f.write(f"| {m} | {vr:.4f} | {vy:.4f} | {f4(b['mean_delta'], signed=False)} | "
                f"{fmt_ci(b['ci_lo'], b['ci_hi'])} | {'✓' if b['sig'] else '—'} |\n")
    f.write("\n### OneEuro 前后（q 域 = G1 契约；px 域 = 对照口径）\n\n")
    f.write("| 指标 | r raw | r q | r px | y raw | y q | y px |\n|---|---|---|---|---|---|---|\n")
    for m in M3_METRICS:
        f.write(f"| {m} | {m_r['raw']['overall'][m]:.4f} | "
                f"{m_r['smoothed']['overall'][m]:.4f} | {m_r['smoothed_px']['overall'][m]:.4f} | "
                f"{m_y['raw']['overall'][m]:.4f} | {m_y['smoothed']['overall'][m]:.4f} | "
                f"{m_y['smoothed_px']['overall'][m]:.4f} |\n")
    f.write("\n### 平滑精度代价（全量 OKS mAP50-95）\n\n")
    f.write("| 模型 | raw | q 域 | q Δ | px 域 | px Δ |\n|---|---|---|---|---|---|\n")
    for m in MODELS:
        d = data["metrics"][(POSETRACK, m)]["m3"]
        f.write(f"| {LABEL[m]} | {d['smoothed']['map_delta']['ap_raw']:.4f} | "
                f"{d['smoothed']['map_delta']['ap_smoothed']:.4f} | "
                f"{f4(d['smoothed']['map_delta']['delta'])} | "
                f"{d['smoothed_px']['map_delta']['ap_smoothed']:.4f} | "
                f"{f4(d['smoothed_px']['map_delta']['delta'])} |\n")
    f.write("\n")


# ---------- 图 ----------

def fig_m1(ds, data):
    """分层 mAP 柱状图：x = 全量/v2/v1/v0，两模型并排，Δ CI 标注在组下方。"""
    m_r = data["metrics"][(ds, "round5-3")]["m1"]
    m_y = data["metrics"][(ds, "yolo11s-pose")]["m1"]
    st = data["stats"][ds]["m1"]
    groups = ["full", "v2", "v1", "v0"]
    gnames = ["full", "v=2", "v=1", "v=0"]
    ap_r = [m_r["full"]["ap"] if g == "full" else m_r["layers"][g]["ap"] for g in groups]
    ap_y = [m_y["full"]["ap"] if g == "full" else m_y["layers"][g]["ap"] for g in groups]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, ap_r, w, label=LABEL["round5-3"], color="#4472c4")
    ax.bar(x + w / 2, ap_y, w, label=LABEL["yolo11s-pose"], color="#ed7d31")
    for xi, v in zip(x - w / 2, ap_r):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, ap_y):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    # Δ 标注放多行 tick label，避免 tight_layout 裁掉 axes 外文本
    labels = []
    for g in groups:
        b = st[g]
        sig = "*" if b["sig"] else ""
        labels.append(f"{gnames[groups.index(g)]}\nΔ={f4(b['point_delta'])}{sig}"
                      f"\n{fmt_ci(b['ci_lo'], b['ci_hi'])}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mAP50-95 (person)")
    ax.set_title(f"{ds}: occlusion-stratified mAP (v=0 not ranked)")
    ax.set_ylim(0, max(max(ap_r), max(ap_y)) * 1.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"fig_m1_{ds}.png", dpi=150)
    plt.close(fig)


def fig_m2(ds, data):
    """角度误差分布：2×2 四关节，v=2 层两模型 hist 叠加（72 桶 -180..180 步 5）。"""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    centers = (HIST_EDGES[:-1] + HIST_EDGES[1:]) / 2
    for ax, j in zip(axes.ravel(), JOINTS):
        for m, color in zip(MODELS, ("#4472c4", "#ed7d31")):
            h = np.array(data["metrics"][(ds, m)]["m2"]["angles"][j]["v2"]["hist"],
                         dtype=float)
            rmse = data["metrics"][(ds, m)]["m2"]["angles"][j]["v2"]["rmse"]
            ax.plot(centers, h, color=color, lw=1.6,
                    label=f"{LABEL[m]} (RMSE {rmse:.1f}°)")
        ax.set_xlim(-90, 90)
        ax.set_title(j)
        ax.set_xlabel("angle error (deg)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(f"{ds}: joint angle error distribution (v=2 layer)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_m2_{ds}.png", dpi=150)
    plt.close(fig)


def fig_m3(data):
    """抖动时序示例：round5-3 左膝（id 13）最长有效 run 的 d_t / j_t 双面板。"""
    units = data["m3_units"]
    best_seg = None
    best_d = None
    for seg in units:
        d = seg.get("13", {}).get("d")
        if d and (best_d is None or len(d) > len(best_d)):
            best_seg, best_d = seg, d
    d = np.array(best_d)
    j = np.array(best_seg["13"]["j"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(np.arange(len(d)), d, color="#4472c4", lw=1.2)
    axes[0].axhline(0.10, color="red", ls="--", lw=1, label="τ = 0.10")
    axes[0].set_ylabel("d_t (frame disp / s_t)")
    axes[0].set_title("round5-3, longest left-knee track run: per-frame displacement")
    axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(len(j)), j, color="#ed7d31", lw=1.2)
    axes[1].set_ylabel("j_t (residual / s_t)")
    axes[1].set_xlabel("frame t (within segment)")
    axes[1].set_title("jitter around causal 5-frame median")
    fig.tight_layout()
    fig.savefig(OUT / "fig_m3_posetrack.png", dpi=150)
    plt.close(fig)


# ---------- summary.json ----------

def build_summary(data):
    s = {"schema_version": 1,
         "generated": datetime.now(timezone.utc).isoformat(),
         "models": MODELS,
         "run_pairs": {ds: [f"yolo11s-pose__{ds}", f"round5-3__{ds}"]
                       for ds in DATASETS + [POSETRACK]},
         "m1": {}, "m2": {}, "m3": {}}
    for ds in DATASETS:
        s["m1"][ds] = {m: data["metrics"][(ds, m)]["m1"] for m in MODELS}
        s["m1"][ds]["bootstrap"] = {k: v for k, v in data["stats"][ds]["m1"].items()
                                    if k != "_protocol"}
        s["m2"][ds] = {m: data["metrics"][(ds, m)]["m2"] for m in MODELS}
        s["m2"][ds]["bootstrap"] = {k: v for k, v in data["stats"][ds]["m2"].items()
                                    if k != "_protocol"}
    s["m3"][POSETRACK] = {m: data["metrics"][(POSETRACK, m)]["m3"] for m in MODELS}
    s["m3"][POSETRACK]["bootstrap"] = {k: v for k, v in
                                       data["stats"][POSETRACK]["m3"].items()
                                       if k != "_protocol"}
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()

    for ds in DATASETS:
        with open(OUT / f"report_{ds}.md", "w") as f:
            f.write(f"# 协议报告：{ds}\n\n")
            f.write(f"基准模型：{LABEL['round5-3']}、{LABEL['yolo11s-pose']}；"
                    f"生成时间 {datetime.now(timezone.utc).isoformat()}；"
                    f"bootstrap Δ 方向 = yolo11s-pose − round5-3\n\n")
            write_m1_table(f, ds, data)
            write_m2_table(f, ds, data)
            f.write(f"![M1](fig_m1_{ds}.png)\n\n![M2](fig_m2_{ds}.png)\n\n")
        fig_m1(ds, data)
        fig_m2(ds, data)
        print(f"report_{ds}.md + 2 图")

    with open(OUT / f"report_{POSETRACK}.md", "w") as f:
        f.write(f"# 协议报告：{POSETRACK}\n\n")
        f.write(f"基准模型：{LABEL['round5-3']}、{LABEL['yolo11s-pose']}；"
                f"生成时间 {datetime.now(timezone.utc).isoformat()}；"
                f"bootstrap Δ 方向 = yolo11s-pose − round5-3\n\n")
        write_m3_table(f, data)
        f.write("![M3](fig_m3_posetrack.png)\n\n")
    fig_m3(data)
    print(f"report_{POSETRACK}.md + 1 图")

    summary = build_summary(data)
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"summary.json（schema_version={summary['schema_version']}）")


if __name__ == "__main__":
    main()
