#!/usr/bin/env python3
import os, re, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_any_csv(path):
    """自动识别分隔符；清理列名空格。"""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(cols, target="ml_score"):
    """大小写不敏感匹配列名；允许 'mlscore' 这类变体。"""
    low = [c.strip().lower() for c in cols]
    if target in low:
        return cols[low.index(target)]
    # 兼容变体
    for i, c in enumerate(low):
        if target.replace("_","") in c.replace("_",""):
            return cols[i]
    raise KeyError(f"'{target}' not found. Available columns: {cols}")

def numeric_key(fname, pat=r"outfile(\d+)"):
    m = re.search(pat, os.path.basename(fname))
    return int(m.group(1)) if m else 10**9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="generated_sequences/outfile*.csv",
                    help="要读取的文件模式（glob）")
    ap.add_argument("--column", default="ML_Score",
                    help="要做箱线图的列名（大小写不敏感）")
    ap.add_argument("--out", default="generated_sequences/ml_score_boxplot.png",
                    help="输出图片路径（png）")
    ap.add_argument("--summary_csv", default="generated_sequences/ml_score_summary.csv",
                    help="保存摘要统计的 CSV 路径")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern), key=lambda s: numeric_key(s, r"outfile(\d+)"))
    if not files:
        raise SystemExit(f"No files found for pattern: {args.pattern}")

    series_list, labels, rows = [], [], []

    for f in files:
        df = read_any_csv(f)
        col = find_col(df.columns.tolist(), target=args.column.lower())
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        series_list.append(s.values)

        m = re.search(r"outfile(\d+)", os.path.basename(f))
        label = f"out{m.group(1)}" if m else os.path.basename(f)
        labels.append(label)

        rows.append({
            "file": os.path.basename(f), "label": label, "n": int(s.shape[0]),
            "mean": float(np.mean(s)) if len(s)>0 else np.nan,
            "std": float(np.std(s, ddof=1)) if len(s)>1 else np.nan,
            "median": float(np.median(s)) if len(s)>0 else np.nan,
            "q25": float(np.percentile(s,25)) if len(s)>0 else np.nan,
            "q75": float(np.percentile(s,75)) if len(s)>0 else np.nan,
            "min": float(np.min(s)) if len(s)>0 else np.nan,
            "max": float(np.max(s)) if len(s)>0 else np.nan,
        })

    # 保存摘要
    summary = pd.DataFrame(rows)
    summary.to_csv(args.summary_csv, index=False)
    print(f"[OK] Saved summary to {args.summary_csv}")
    print(summary)

    # 画箱线图
    plt.figure(figsize=(9,5))
    plt.boxplot(series_list, labels=labels, showmeans=True)
    plt.xlabel("File (outfile batch)")
    plt.ylabel(args.column)
    plt.title(f"Distribution of {args.column} across outfile{{i}}.csv")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[OK] Saved figure to {args.out}")

if __name__ == "__main__":
    main()
