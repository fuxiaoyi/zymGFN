import os
import csv
import argparse
import pandas as pd

def _clean_name(s: str) -> str:
    """统一名称（用于对齐 FASTA/CSV/TSV）"""
    return str(s).strip().split("\t")[0].lstrip(">").strip().strip('"')

def read_fasta_as_dict(fasta_path: str):
    """读取 FASTA，返回 {name: sequence}"""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    sequences = {}
    cur_name = None
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                cur_name = _clean_name(line)
            else:
                if cur_name is not None:
                    sequences[cur_name] = line.strip()
    return sequences

def read_tm_file(tm_path: str):
    """
    读取 Foldseek TM 输出（query, target, alntmscore, qtmscore, ttmscore, alnlen）
    返回列表：[ (name, TM, TM_norm_que, algn) , ... ]
    """
    if not os.path.exists(tm_path):
        raise FileNotFoundError(f"TM file not found: {tm_path}")
    rows = []
    with open(tm_path, "r") as f:
        lines = f.readlines()
    if lines and lines[0].lower().startswith("query"):
        lines = lines[1:]
    for entry in lines:
        parts = entry.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        name = _clean_name(parts[0])
        try:
            TM = float(parts[2])           # alntmscore
            TM_norm_que = float(parts[4])  # 这里用 ttmscore；若需 qtmscore 改 parts[3]
            algn = int(parts[5])
        except Exception:
            continue
        rows.append((name, TM, TM_norm_que, algn))
    return rows

def _read_csv_auto(path: str) -> pd.DataFrame:
    """自动分隔符读取 CSV/TSV"""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, sep="\t")

def load_tox_map(tox_csv: str):
    """
    读取 ToxinPred2 输出，返回 {name: ml_score}
    兼容列名：id/name, ml_score/score
    """
    m = {}
    if tox_csv is None or not os.path.exists(tox_csv):
        return m
    df = _read_csv_auto(tox_csv)
    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("id") or cols.get("name") or list(df.columns)[0]
    sc = cols.get("ml_score") or cols.get("score")
    if sc is None:
        # 没有分数列就直接返回空
        return m
    for _, r in df.iterrows():
        name = _clean_name(r[idc])
        try:
            ml = float(r[sc])
        except Exception:
            ml = 0.0
        m[name] = ml
    return m

def load_kcat_map(kcat_csv: str, col_use: str = "pred_linear"):
    """
    读取 UniKP kcat 文件，返回 {name: pred_linear}
    兼容列名：id/name，pred_linear 或 pred_log10
    """
    m = {}
    if kcat_csv is None or not os.path.exists(kcat_csv):
        return m
    df = _read_csv_auto(kcat_csv)
    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("id") or cols.get("name") or list(df.columns)[0]
    kc = cols.get(col_use.lower())
    if kc is None:
        # 兜底
        kc = cols.get("pred_linear") or cols.get("pred_log10")
        if kc is None:
            return m
    for _, r in df.iterrows():
        name = _clean_name(r[idc])
        v = r[kc]
        if pd.notna(v):
            try:
                m[name] = float(v)
            except Exception:
                pass
    return m

def append_to_csv(
    name, sequence, TM, TM_norm_que, algn, iteration_num,
    output_file, ml_score=None, kcat=None
):
    file_exists = os.path.exists(output_file) and os.stat(output_file).st_size > 0
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "name",
            "sequence",
            "TM",
            "TM_norm_que",
            "algn",
            "iteration_num",
            "ml_score",       # 新增
            "kcat",    # 新增（kcat）
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "name": name,
            "sequence": sequence,
            "TM": TM,
            "TM_norm_que": TM_norm_que,
            "algn": algn,
            "iteration_num": iteration_num,
            "ml_score": ml_score if ml_score is not None else "",
            "kcat": kcat if kcat is not None else "",
        })

def generate_dataset(iteration_num, ec_label, model_dir, tox_csv=None, kcat_csv=None):
    output_file = "logs.csv"

    # 1) 读 FASTA
    fasta_path = f"seq_gen_{ec_label}_iteration{iteration_num}.fasta"
    sequences_rep = read_fasta_as_dict(fasta_path)

    # 2) 读 TM
    tm_path = f"alpha_{ec_label}_TM_iteration{iteration_num}"
    tm_rows = read_tm_file(tm_path)

    # 3) 读 毒性 + kcat（可选）
    tox_map = load_tox_map(tox_csv) if tox_csv else {}
    kcat_map = load_kcat_map(kcat_csv, col_use="pred_linear") if kcat_csv else {}

    # 4) 合并写出
    for name, TM, TM_norm_que, algn in tm_rows:
        if name not in sequences_rep:
            # 若名字不对齐，尝试再清洗一次
            cname = _clean_name(name)
            if cname in sequences_rep:
                name = cname
            else:
                continue
        sequence = sequences_rep[name]
        ml = tox_map.get(name, None)
        kc = kcat_map.get(name, None)

        append_to_csv(
            name=name,
            sequence=sequence,
            TM=TM,
            TM_norm_que=TM_norm_que,
            algn=algn,
            iteration_num=iteration_num,
            output_file=output_file,
            ml_score=ml,
            kcat=kc
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)  # 保留与原参一致
    # 可选：显式传入毒性与 kcat 的文件；若不传，按默认路径推断
    parser.add_argument("--tox_csv", type=str, default=None,
                        help="ToxinPred2输出 CSV/TSV（含 ml_score）")
    parser.add_argument("--kcat_csv", type=str, default=None,
                        help="UniKP kcat CSV/TSV（含 pred_linear 或 pred_log10）")
    args = parser.parse_args()

    # 若未显式传入，则按默认路径推断
    if not args.tox_csv:
        # 你可以按项目目录结构修改下面默认路径
        guess = os.path.join("toxicity", f"{args.label}_tox_iteration{args.iteration_num}.csv")
        args.tox_csv = guess if os.path.exists(guess) else None
    if not args.kcat_csv:
        guess = os.path.join("kcat", f"{args.label}_kcat_iteration{args.iteration_num}.csv")
        args.kcat_csv = guess if os.path.exists(guess) else None

    print(f"iteration number {args.iteration_num}")
    print(f"tox_csv:  {args.tox_csv}")
    print(f"kcat_csv: {args.kcat_csv}")

    generate_dataset(
        iteration_num=args.iteration_num,
        ec_label=args.label,
        model_dir=args.model_dir,
        tox_csv=args.tox_csv,
        kcat_csv=args.kcat_csv
    )
