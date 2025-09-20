# dataset_gen_toxUnikp.py  —— hydra 版（最小改）
import os
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path, get_original_cwd

# -------------------
# 工具函数（原样保留/微调）
# -------------------
def _clean_name(s: str) -> str:
    return str(s).strip().split("\t")[0].lstrip(">").strip().strip('"')

def read_fasta_as_dict(fasta_path: str) -> Dict[str, str]:
    p = Path(fasta_path)
    if not p.exists():
        raise FileNotFoundError(f"FASTA not found: {p}")
    sequences: Dict[str, str] = {}
    cur_name: Optional[str] = None
    with p.open("r") as f:
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

def read_tm_file(tm_path: str) -> List[Tuple[str, float, float, int]]:
    p = Path(tm_path)
    if not p.exists():
        raise FileNotFoundError(f"TM file not found: {p}")
    rows: List[Tuple[str, float, float, int]] = []
    lines = p.read_text().splitlines()
    if lines and lines[0].lower().startswith("query"):
        lines = lines[1:]
    for entry in lines:
        parts = entry.split("\t")
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
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, sep="\t")

def load_tox_map(tox_csv: Optional[str]) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if not tox_csv:
        return m
    p = Path(tox_csv)
    if not p.exists():
        return m
    df = _read_csv_auto(str(p))
    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("id") or cols.get("name") or list(df.columns)[0]
    sc = cols.get("ml_score") or cols.get("score")
    if sc is None:
        return m
    for _, r in df.iterrows():
        name = _clean_name(r[idc])
        try:
            ml = float(r[sc])
        except Exception:
            ml = 0.0
        m[name] = ml
    return m

def load_kcat_map(kcat_csv: Optional[str], col_use: str = "pred_linear") -> Dict[str, float]:
    m: Dict[str, float] = {}
    if not kcat_csv:
        return m
    p = Path(kcat_csv)
    if not p.exists():
        return m
    df = _read_csv_auto(str(p))
    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("id") or cols.get("name") or list(df.columns)[0]
    kc = cols.get(col_use.lower()) or cols.get("pred_linear") or cols.get("pred_log10")
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
    name, sequence, TM, TM_norm_que, algn, iteration_num, output_file,
    ml_score=None, kcat=None
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="") as csvfile:
        fieldnames = [
            "name","sequence","TM","TM_norm_que","algn",
            "iteration_num","ml_score","kcat"
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

# -------------------
# 主逻辑：改为读 cfg.paths
# -------------------
def generate_dataset(
    cfg: DictConfig,
    iteration_num: int,
    ec_label: str,
    model_dir: str,
    *,
    output_logs: str,
    sequences_dir: str,
    tm_dir: str,
    tox_csv: Optional[str],
    kcat_csv: Optional[str],
):
    # 1) 解析路径：
    # - 输出、FASTA、TM 都默认相对 **Hydra 的 run 目录**（Path.cwd()）
    # run_cwd = Path.cwd()
    run_cwd = Path(cfg.run_dir)

    output_file = run_cwd / output_logs
    fasta_path  = Path(sequences_dir) / f"seq_gen_{ec_label}_iteration{iteration_num}.fasta"
    tm_path     = Path(tm_dir) / f"alpha_{ec_label}_TM_iteration{iteration_num}"

    # 2) 读取
    sequences_rep = read_fasta_as_dict(str(fasta_path))
    tm_rows       = read_tm_file(str(tm_path))
    tox_map       = load_tox_map(tox_csv)
    kcat_map      = load_kcat_map(kcat_csv, col_use="pred_linear")

    # 3) 合并写出
    for name, TM, TM_norm_que, algn in tm_rows:
        if name not in sequences_rep:
            cname = _clean_name(name)
            if cname in sequences_rep:
                name = cname
            else:
                continue
        sequence = sequences_rep[name]
        ml = tox_map.get(name)
        kc = kcat_map.get(name)
        append_to_csv(
            name=name,
            sequence=sequence,
            TM=TM,
            TM_norm_que=TM_norm_que,
            algn=algn,
            iteration_num=iteration_num,
            output_file=str(output_file),
            ml_score=ml,
            kcat=kc,
        )

# -------------------
# Hydra 入口
# -------------------
@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    """
    期望的配置结构（可在命令行覆盖）：
    iteration_num: 1
    label: "3.1.1.1"
    model_dir: "/path/to/model"

    paths:
      output_logs: "logs.csv"        # 输出文件（相对 run 目录）
      sequences: "."                 # FASTA 所在目录（相对 run 目录）
      tm_scores: "."                 # TM 文件所在目录（相对 run 目录）
      toxicity: null                 # 也可传绝对/相对路径到具体文件
      kcat: null
    """
    # 给 paths 设默认值（如果没提供）
    defaults = {
        "paths": {
            "output_logs": "logs.csv",
            "sequences": ".",
            "tm_scores": ".",
            "toxicity": None,
            "kcat": None,
        }
    }
    cfg = OmegaConf.merge(defaults, cfg)

    # 如果用户传的是“目录”，我们按默认文件名拼起来；如果传的是具体文件，则直接用
    tox_csv = cfg.paths.toxicity
    if tox_csv and os.path.isdir(tox_csv):
        tox_csv = str(Path(tox_csv) / f"{cfg.label}_tox_iteration{cfg.iteration_num}.csv")

    kcat_csv = cfg.paths.kcat
    if kcat_csv and os.path.isdir(kcat_csv):
        kcat_csv = str(Path(kcat_csv) / f"{cfg.label}_kcat_iteration{cfg.iteration_num}.csv")

    print("=== Effective Config ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    generate_dataset(
        cfg=cfg,
        iteration_num=cfg.iteration_num,
        ec_label=cfg.label,
        model_dir=cfg.model_dir,
        output_logs=cfg.paths.output_logs,
        sequences_dir=cfg.paths.sequences,
        tm_dir=cfg.paths.tm_scores,
        tox_csv=tox_csv,
        kcat_csv=kcat_csv,
    )

if __name__ == "__main__":
    main()

'''
python ${folder_path}dataset_gen_toxUnikp.py \
  iteration_num=${i} \
  label=${label} \
  model_dir="/home/.../ZymCTRL_local" \
  paths.output_logs="logs.csv" \
  paths.sequences="." \
  paths.tm_scores="." \
  paths.toxicity="${folder_path}outfile${i}.csv" \
  paths.kcat="${folder_path}results_kcat${i}.csv" \
  hydra.run.dir="${folder_path}output_iteration${i}"

'''