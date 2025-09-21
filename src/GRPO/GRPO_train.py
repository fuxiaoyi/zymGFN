# train_hydra.py —— 结构化配置（Hydra + dataclass）版本
from __future__ import annotations

from trainer.utils import *
from trainer.pLM_weightedDPO import weighted_DPO
from trainer.pLM_GRPO import pLM_GRPOTrainer

import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
from hydra.utils import to_absolute_path
from hydra.core.config_store import ConfigStore

from dataclasses import dataclass
from datasets import Dataset
from trl import GRPOConfig
from transformers import AutoTokenizer
from accelerate.utils import set_seed

import torch, numpy as np, random, pandas as pd, math, os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


# =======================
# 1) 结构化配置（Schema）
# =======================
@dataclass
class HPConfig:
    beta: float = 0.1
    seed: int = 42
    learning_rate: float = 2e-6
    batch_size: int = 8
    num_epochs: int = 1
    split_percent: float = 0.2
    adam_betas: Tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-8
    adam_decay: float = 0.1
    len_center: float = 260.0
    len_sigma: float = 0.5
    use_tox_penalty: bool = True
    tox_alpha: float = 1.0
    tox_factor: float = 0.1
    use_kcat_factor: bool = True
    kcat_beta: float = 2.0


@dataclass
class PathsConfig:
    tox_csv: Optional[str] = None
    kcat_csv: Optional[str] = None


@dataclass
class TrainConfig:
    iteration_num: int = MISSING
    label: str = MISSING
    model_dir: str = MISSING
    max_iteration_num: int = 30
    hp: HPConfig = field(default_factory=HPConfig)        # ✅ 用工厂创建新实例
    paths: PathsConfig = field(default_factory=PathsConfig)


# 将 schema 注册给 Hydra（无需外部 YAML 也能工作）
cs = ConfigStore.instance()
cs.store(name="train_schema", node=TrainConfig)


# =======================
# 2) 常用工具函数
# =======================
def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def _length_reward(seq_len: int, center: float = 260.0, sigma: float = 0.5) -> float:
    x = (seq_len / center) - 1.0
    return math.exp(-((x * x) / (sigma * sigma)))


def _minmax_norm(v, vmin, vmax):
    if vmax <= vmin:
        return 1.0
    return max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))


def reward_len(completions, **kwargs):
    # 你的占位 reward（与原代码一致）
    return 0


def format_sequence(sequence, label):
    return f"<sep><start>{sequence}<end><|endoftext|>"


def _clean_name(s: str) -> str:
    return str(s).strip().split("\t")[0].lstrip(">").strip().strip('"')


def _read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, sep="\t")


def read_fasta_as_dict(fasta_path: str) -> Dict[str, str]:
    p = Path(fasta_path)
    if not p.exists():
        raise FileNotFoundError(f"FASTA not found: {p}")
    seqs: Dict[str, str] = {}
    cur: Optional[str] = None
    for line in p.read_text().splitlines():
        if not line:
            continue
        if line.startswith(">"):
            cur = _clean_name(line)
        else:
            if cur is not None:
                seqs[cur] = line.strip()
    return seqs


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
            TM = float(parts[2])
            TM_norm_que = float(parts[4])   # ttmscore
            algn = int(parts[5])
        except Exception:
            continue
        rows.append((name, TM, TM_norm_que, algn))
    return rows


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
            m[name] = float(r[sc])
        except Exception:
            m[name] = 0.0
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
        v = r[kc]
        if pd.notna(v):
            try:
                m[_clean_name(r[idc])] = float(v)
            except Exception:
                pass
    return m


# =======================
# 3) 从上一轮构造训练集
# =======================
def build_dataset_from_prev_run(
    *,
    prev_run_dir: Path,
    iteration_num: int,
    label: str,
    hp: HPConfig,
    tox_csv: Optional[str],
    kcat_csv: Optional[str],
    tox_quantile: float = 0.70,
    use_negrew: bool = False,
) -> Dataset:
    logs_path = prev_run_dir / "logs.csv"
    if not logs_path.exists():
        raise FileNotFoundError(f"[build_dataset] logs.csv not found: {logs_path}")

    df = pd.read_csv(str(logs_path))
    df = df[df["iteration_num"] == (iteration_num - 1)]

    need_cols = ["TM_norm_que", "sequence", "algn"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"[ERROR] logs.csv missing required column: {c}")

    df["TM_norm_que"] = pd.to_numeric(df["TM_norm_que"], errors="coerce")
    df["algn"] = pd.to_numeric(df["algn"], errors="coerce")

    has_tox = hp.use_tox_penalty and ("ml_score" in df.columns)
    has_kcat_norm = hp.use_kcat_factor and ("kcat_norm" in df.columns)
    has_kcat_raw = hp.use_kcat_factor and (("kcat" in df.columns) or ("kcat_raw" in df.columns))

    # kcat 归一化
    if hp.use_kcat_factor and not has_kcat_norm and has_kcat_raw:
        kcol = "kcat" if "kcat" in df.columns else "kcat_raw"
        df[kcol] = pd.to_numeric(df[kcol], errors="coerce")
        valid = df[kcol].dropna()
        if len(valid) > 0:
            kmin, kmax = valid.min(), valid.max()
            df["kcat_norm"] = df[kcol].apply(lambda v: _minmax_norm(v, kmin, kmax) if pd.notna(v) else np.nan)
            has_kcat_norm = True
    tox_thr = df["ml_score"].quantile(tox_quantile) if has_tox else 1.0

    rows = []
    for _, r in df.iterrows():
        seq = str(r.get("sequence", ""))
        TMn = r.get("TM_norm_que")
        alg = r.get("algn")
        if not seq or pd.isna(TMn) or pd.isna(alg):
            continue

        # 毒性
        if has_tox:
            ml = r.get("ml_score", np.nan)
            if pd.isna(ml):
                continue
            ml = max(0.0, min(1.0, float(ml)))
            tox_factor = hp.tox_factor if ml >= tox_thr else 1.0
        else:
            tox_factor = 1.0

        # kcat
        kcat_factor = 1.0
        if hp.use_kcat_factor:
            if not has_kcat_norm:
                continue
            kn = r.get("kcat_norm", np.nan)
            if pd.isna(kn):
                continue
            kn = max(0.0, min(1.0, float(kn)))
            kcat_factor = (kn ** float(hp.kcat_beta))

        length_rew = _length_reward(len(seq), hp.len_center, hp.len_sigma)
        base_weight = (float(TMn) + (float(alg) / 100.0)) * length_rew

        if not use_negrew:
            weight = base_weight * tox_factor * kcat_factor
        else:
            a1, a2 = 3, 0.5
            base = base_weight * kcat_factor
            if has_tox and ml >= tox_thr:
                g = max((ml - tox_thr) / (1 - tox_thr), 0)
                weight = -max(a1 * g - a2 * base, 0)
            else:
                weight = base

        rows.append({
            "prompt": label,
            "completion": format_sequence(seq, label),
            "reward": float(weight),
            "weight": float(weight),
            "TM_norm_que": float(TMn),
            "algn": float(alg),
            "length_rew": float(length_rew),
            "tox_factor": float(tox_factor),
            "kcat_factor": float(kcat_factor),
        })

    if not rows:
        print("[WARN] No rows kept for dataset; check ml_score/kcat_norm in prev logs.")
    return Dataset.from_list(rows)


# =======================
# 4) Hydra 入口
# =======================
@hydra.main(version_base=None, config_path=None, config_name="train_schema")
def main(cfg: TrainConfig):
    """
    从 CLI 或 YAML 传入：
      iteration_num: 0  # 注意：从 0 开始
      label: "3.1.1.1"
      model_dir: "/root/zymGFN/zymCTRL/"
      max_iteration_num: 30
      hp: {... 超参 ...}
      paths:
        tox_csv: null
        kcat_csv: null

    ⚠️ 在外层 bash/SLURM 中，请保证同一迭代脚本统一：
        hydra.run.dir="${folder_path}output_iteration${i}"
    """
    # ------------- 基础参数 -------------
    it = int(cfg.iteration_num)
    lab = str(cfg.label)
    mdl = to_absolute_path(str(cfg.model_dir))
    max_it = int(cfg.max_iteration_num)
    HP = cfg.hp

    # ------------- 目录关系 -------------
    # run_dir = Path.cwd()  # 本轮输出目录（hydra.run.dir）
    run_dir = Path(cfg.run_dir)
    prev_run_dir = Path(cfg.prev_dir)
    # prev_run_dir = run_dir.with_name(f"output_iteration{it-1}") if it > 0 else None

    print(f"[Hydra] run_dir = {run_dir}")
    print(f"[Hydra] prev_run_dir = {prev_run_dir}")

    # ------------- 随机种子 -------------
    seed_everything(HP.seed)

    # ------------- 构造数据集（来自上一轮 logs.csv） -------------
    if it > 0:
        dataset = build_dataset_from_prev_run(
            prev_run_dir=prev_run_dir,
            iteration_num=it,
            label=lab,
            hp=HP,
            tox_csv=cfg.paths.tox_csv,
            kcat_csv=cfg.paths.kcat_csv,
        )
    else:
        # 第 0 轮没有上一轮日志；按你的流程，通常 0 轮是初始化/预训练
        raise RuntimeError("iteration_num==0 时请先准备初始训练数据或修改逻辑以支持 cold-start。")

    split = dataset.train_test_split(test_size=HP.split_percent, seed=HP.seed, shuffle=True)
    train_dataset, eval_dataset = split["train"], split["test"]

    # ------------- tokenizer -------------
    tokenizer = AutoTokenizer.from_pretrained(
        mdl, add_eos_token=False, add_bos_token=False, use_fast=True
    )
    tokenizer.eos_token_id = 1
    tokenizer.pad_token_id = 0

    # ------------- 模型/检查点 -------------
    if it > 1:
        model_path = str(prev_run_dir)     # 上一轮目录里保存的模型
        checkpoint = checkpoint_load(str(prev_run_dir))
    else:
        model_path = mdl
        checkpoint = None

    # ------------- 学习率 & 优化器 -------------
    lr_list = np.linspace(HP.learning_rate, 0.0, num=max_it)
    lr_this = float(lr_list[it-1] if it > 0 else lr_list[0])
    if hasattr(lr_this, "item"):
        lr_this = float(lr_this.item())

    optimizer, model, scheduler = load_optimizer_scheduler(model_path, checkpoint, lr_this, OmegaConf.to_container(HP, resolve=True))

    # ------------- 训练配置 -------------
    training_args = GRPOConfig(
        output_dir=str(run_dir),     # 产物写到本轮 run 目录
        logging_steps=100,
        beta=HP.beta,
        num_train_epochs=HP.num_epochs,
        learning_rate=lr_this,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
        save_steps=5,
        num_generations=8,
        bf16=True,
        gradient_checkpointing=True,
        torch_compile=False,
    )

    print("model ", model)
    trainer = pLM_GRPOTrainer(
        model=model,
        ref_model=mdl,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
    )

    # ------------- 把优化器移到正确设备 -------------
    from trainer.utils import _optimizer_to_device
    try:
        train_device = trainer.accelerator.device
    except Exception:
        train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _optimizer_to_device(optimizer, train_device)

    print("fixed LR (optimizer) before training:", trainer.optimizer.param_groups[0]["lr"])
    trainer.train()
    trainer.save_model()
    print("fixed LR (optimizer) after training:", trainer.optimizer.param_groups[0]["lr"])
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

'''
python train_hydra.py \
  iteration_num=7 \
  label=3.1.1.1 \
  model_dir=/root/zymGFN/zymCTRL/ \
  max_iteration_num=30 \
  hp.learning_rate=3e-6 hp.num_epochs=2 \
  paths.tox_csv=/abs/path/outfile7.csv \
  paths.kcat_csv=/abs/path/results_kcat7.csv \
  hydra.run.dir="/path/to/output_iteration7"
'''