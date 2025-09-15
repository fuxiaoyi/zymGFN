#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_tm_file(path, tm_col=2, sep="\t"):
    """读取单个 TM 文件，返回该文件的 TM 数组（float）。
    行格式示例：
    3.1.1.1_31_2_iteration3\t7atl\t2.097E-01\t5.322E-02\t4.134E-02\t48
    """
    vals = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) <= tm_col:
                # 兼容使用空白分隔的情况
                parts = line.split()
                if len(parts) <= tm_col:
                    continue
            try:
                # 形如 2.097E-01 的科学计数法，float 可直接解析
                vals.append(float(parts[tm_col]))
            except Exception:
                # 跳过无法解析的行
                continue
    return np.array(vals, dtype=float)

def natural_key(fname):
    """按 iteration 编号排序：3.1.1.1_TM_iteration3 -> 3"""
    base = os.path.basename(fname)
    m = re.search(r"iteration(\d+)", base)
    return int(m.group(1)) if m else 10**9

def main():
    ap = argparse.ArgumentParser(description="Boxplot of TM distributions across iterations")
    ap.add_argument("--pattern", type=str, default="TMscores/3.1.1.1_TM_iteration*",
                    help="glob 模式（默认匹配 3.1.1.1_TM_iteration*）")
    ap.add_argument("--tm_col", type=int, default=2,
                    help="TM 所在列的索引（从 0 开始，默认 2，即第三列）")
    ap.add_argument("--out_png", type=str, default="TMscores/tm_boxplot.png",
                    help="输出图片路径")
    ap.add_argument("--summary_csv", type=str, default="TMscores/tm_summary.csv",
                    help="保存统计摘要的 CSV")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern), key=natural_key)
    if not files:
        raise SystemExit(f"No files found for pattern: {args.pattern}")

    series_list, labels, rows = [], [], []

    for fp in files:
        arr = parse_tm_file(fp, tm_col=args.tm_col)
        if arr.size == 0:
            # 没有有效数据就跳过
            continue
        series_list.append(arr)
        m = re.search(r"iteration(\d+)", os.path.basename(fp))
        label = f"iter{m.group(1)}" if m else os.path.basename(fp)
        labels.append(label)

        rows.append({
            "file": os.path.basename(fp),
            "label": label,
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })

    if not series_list:
        raise SystemExit("No valid TM values parsed. Check file format and tm_col index.")

    # 保存摘要
    summary = pd.DataFrame(rows)
    summary.to_csv(args.summary_csv, index=False)
    print(f"[OK] saved summary -> {args.summary_csv}")
    print(summary)

    # 画箱线图
    plt.figure(figsize=(10, 5))
    plt.boxplot(series_list, labels=labels, showmeans=True)
    plt.xlabel("Iteration")
    plt.ylabel("TM")
    plt.title("TM distribution across iterations")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"[OK] saved figure -> {args.out_png}")

if __name__ == "__main__":
    main()
