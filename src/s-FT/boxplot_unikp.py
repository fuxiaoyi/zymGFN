#!/usr/bin/env python3
import os, re, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_any_csv(path):
    """Auto-detect delimiter and trim column names."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df

def find_pred_linear_col(cols):
    low = [c.strip().lower() for c in cols]
    if "pred_linear" in low:
        return cols[low.index("pred_linear")]
    # tolerant partial match
    for i, c in enumerate(low):
        if "pred_linear" in c:
            return cols[i]
    raise KeyError(f"'pred_linear' not found. Available: {cols}")

def numeric_key(s):
    m = re.search(r"results_kcat(\d+)", os.path.basename(s))
    return int(m.group(1)) if m else 10**9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, default="generated_sequences/results_kcat*.csv",
                    help="glob pattern to find files")
    ap.add_argument("--out", type=str, default="generated_sequences/pred_linear_boxplot.png",
                    help="output figure path (png)")
    ap.add_argument("--summary_csv", type=str, default="pred_linear_summary.csv",
                    help="where to save the summary table")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern), key=numeric_key)
    if not files:
        raise SystemExit(f"No files found for pattern: {args.pattern}")

    series_list, labels, summaries = [], [], []
    for f in files:
        df = read_any_csv(f)
        col = find_pred_linear_col(df.columns.tolist())
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        series_list.append(s.values)
        m = re.search(r"results_kcat(\d+)", os.path.basename(f))
        labels.append(f"kcat{m.group(1)}" if m else os.path.basename(f))
        summaries.append({
            "file": os.path.basename(f),
            "label": labels[-1],
            "n": int(s.shape[0]),
            "mean": float(np.mean(s)) if len(s)>0 else np.nan,
            "std": float(np.std(s, ddof=1)) if len(s)>1 else np.nan,
            "median": float(np.median(s)) if len(s)>0 else np.nan,
            "q25": float(np.percentile(s,25)) if len(s)>0 else np.nan,
            "q75": float(np.percentile(s,75)) if len(s)>0 else np.nan,
            "min": float(np.min(s)) if len(s)>0 else np.nan,
            "max": float(np.max(s)) if len(s)>0 else np.nan,
        })

    # save summary table
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.summary_csv, index=False)
    print(f"Saved summary to {args.summary_csv}")
    print(summary_df)

    # boxplot
    plt.figure(figsize=(8,5))
    plt.boxplot(series_list, labels=labels, showmeans=True)
    plt.xlabel("File (kcat batch)")
    plt.ylabel("pred_linear")
    plt.title("Distribution of pred_linear across kcat files")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")

if __name__ == "__main__":
    main()
