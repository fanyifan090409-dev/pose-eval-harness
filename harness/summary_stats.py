#!/usr/bin/env python3
"""T6 结果紧凑汇总：逐数据集打印 bootstrap 显著性结论。"""
import glob
import json

for f in sorted(glob.glob("runs/stats__*.json")):
    s = json.load(open(f))
    print("==", s["dataset"], "(n_boot", s["protocol"]["n_boot"], ")")
    if "m1" in s:
        m1 = s["m1"]
        print(" M1 units:", m1["_protocol"]["n_units"])
        for c in ("full", "v2", "v1", "v0"):
            v = m1[c]
            print("  %-5s d=%+.4f CI=[%+.4f,%+.4f] sig=%s"
                  % (c, v["point_delta"], v["ci_lo"], v["ci_hi"], v["sig"]))
    if "m2" in s:
        for j in s["m2"]:
            if j == "_protocol":
                continue
            for vb in s["m2"][j]:
                v = s["m2"][j][vb]
                print("  m2 %s/%s: d=%+.3f CI=[%+.3f,%+.3f] sig=%s n=%s/%s"
                      % (j, vb, v["point_delta"], v["ci_lo"], v["ci_hi"],
                         v["sig"], v["n_img_a"], v["n_img_b"]))
    if "m3" in s:
        print(" M3 units:", s["m3"]["_protocol"]["n_units_a"], "/",
              s["m3"]["_protocol"]["n_units_b"])
        for m in ("disp_mean", "jitter_mean", "disp_p95", "jitter_p95",
                  "jump_0.05", "jump_0.1", "jump_0.2"):
            v = s["m3"][m]
            print("  %-12s CI=[%+.4f,%+.4f] sig=%s"
                  % (m, v["ci_lo"], v["ci_hi"], v["sig"]))
