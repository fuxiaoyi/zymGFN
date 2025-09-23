from src.utils import *
from src.pLM_weigtedDPO import weighted_DPO

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel)
from trl.trainer.utils import pad
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import argparse
import torch
import numpy as np
import random
import pandas as pd
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)
args = parser.parse_args()

CONFIG = {
    "beta": 0.1,           # GRPO的beta
    "seed": 42,
    "learning_rate": 2e-6,
    "batch_size": 8,
    "num_epochs": 1,
    "split_percent": 0.2,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "adam_decay": 0.1,

    # --- 新增：reward 相关开关/系数 ---
    "len_center": 260.0,
    "len_sigma": 0.5,
    "use_tox_penalty": True,
    "tox_alpha": 1.0,       # weight *= (1-ml_score)**alpha
    "use_kcat_factor": True,
    "kcat_beta": 2.0,       # weight *= (kcat_norm)**beta
    # 如果 logs.csv 已经提供 kcat_norm(0~1)，优先使用；否则用原始 kcat 做本批次 min-max 归一化
}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    set_seed(seed)

def _length_reward(seq_len: int, center: float = 260.0, sigma: float = 0.5) -> float:
    """长度奖励（高斯），以 center 为峰值。"""
    x = (seq_len / center) - 1.0
    return math.exp(-((x * x) / (sigma * sigma)))

def _minmax_norm(v, vmin, vmax):
    if vmax <= vmin:
        return 1.0
    return max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))

def reward_len(completions, **kwargs):
    # 你原来的占位函数，保持不变（GRPO内部会用到）
    return 0

def format_sequence(sequence, label):
    return f"<sep><start>{sequence}<end><|endoftext|>"

def generate_dataset(iteration_num, label, tox_quantile=0.70, use_negrew=False):
    """
    从 logs.csv 读取上一轮 (iteration_num - 1) 的数据，计算综合 weight。
    若某行缺失 ml_score 或 kcat，则直接丢弃。
    """
    df = pd.read_csv("logs.csv")
    df = df[df["iteration_num"] == (iteration_num - 1)]

    # 必须列
    need_cols = ["TM_norm_que", "sequence", "algn"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"[ERROR] logs.csv missing required column: {c}")

    # 转换数值
    df["TM_norm_que"] = pd.to_numeric(df["TM_norm_que"], errors="coerce")
    df["algn"] = pd.to_numeric(df["algn"], errors="coerce")

    # 毒性 & kcat
    has_tox = CONFIG["use_tox_penalty"] and ("ml_score" in df.columns)
    has_kcat_norm = CONFIG["use_kcat_factor"] and ("kcat_norm" in df.columns)
    has_kcat_raw = CONFIG["use_kcat_factor"] and (("kcat" in df.columns) or ("kcat_raw" in df.columns))

    # kcat 归一化
    if CONFIG["use_kcat_factor"] and not has_kcat_norm and has_kcat_raw:
        kcol = "kcat" if "kcat" in df.columns else "kcat_raw"
        df[kcol] = pd.to_numeric(df[kcol], errors="coerce")
        valid = df[kcol].dropna()
        if len(valid) > 0:
            kmin, kmax = valid.min(), valid.max()
            df["kcat_norm"] = df[kcol].apply(lambda v: _minmax_norm(v, kmin, kmax) if pd.notna(v) else np.nan)
            has_kcat_norm = True
            
    tox_thr = df["ml_score"].quantile(tox_quantile)

    rows = []
    for _, entry in df.iterrows():
        sequence = str(entry.get("sequence", ""))
        TM_norm_que = entry.get("TM_norm_que")
        algn = entry.get("algn")

        if not sequence or pd.isna(TM_norm_que) or pd.isna(algn):
            continue

        # 毒性：如果需要但缺失 -> 丢弃
        if has_tox:
            ml_score = entry.get("ml_score", np.nan)
            if pd.isna(ml_score):
                continue
            ml_score = max(0.0, min(1.0, float(ml_score)))
            tox_factor = (1.0 - ml_score) ** float(CONFIG["tox_alpha"])
            if ml_score>=tox_thr:
                tox_factor = 0
            else:
                tox_factor = 1.0
        else:
            tox_factor = 1.0

        # kcat：如果需要但缺失 -> 丢弃
        kcat_factor = 1.0
        if CONFIG["use_kcat_factor"]:
            if has_kcat_norm:
                kcat_norm = entry.get("kcat_norm", np.nan)
                if pd.isna(kcat_norm):
                    continue
                kcat_norm = max(0.0, min(1.0, float(kcat_norm)))
                kcat_factor = (kcat_norm ** float(CONFIG["kcat_beta"]))
            else:
                # 没有任何 kcat 信息 -> 丢弃
                continue

        # 长度奖励
        length_rew = _length_reward(len(sequence), CONFIG["len_center"], CONFIG["len_sigma"])

        # 基础权重
        base_weight = (float(TM_norm_que) + (float(algn) / 100.0)) * length_rew

        if not use_negrew:
            weight = base_weight * tox_factor * kcat_factor
        else:
            a1, a2 = 3, 0.5
            base = base_weight * kcat_factor
            if ml_score>=tox_thr:
                g = max((ml-thr)/(1-thr),0)
                weight = -max(a1*g-a2*base, 0)
            else:
                weight = base

        rows.append({
            "prompt": label,
            "completion": format_sequence(sequence, label),
            "reward": float(weight),
            "weight": float(weight),
            "TM_norm_que": float(TM_norm_que),
            "algn": float(algn),
            "length_rew": float(length_rew),
            "tox_factor": float(tox_factor),
            "kcat_factor": float(kcat_factor),
        })

    if not rows:
        print("[WARN] No rows kept for dataset; maybe missing ml_score or kcat for all rows?")
    return Dataset.from_list(rows)


# ----------------- 训练主流程（与你原来一致） -----------------
seed_everything(CONFIG["seed"])

# create dataset
root_dir = os.path.dirname(os.path.abspath(__file__))
seq_dir = os.path.join(root_dir, "data", "inputs")
fasta_file = os.path.join(seq_dir, f"seq_gen_{args.label}_iteration{args.iteration_num-1}.fasta")

dataset = generate_dataset(args.iteration_num, args.label)
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)
train_dataset = split['train']
eval_dataset  = split['test']

tokenizer_dir = args.model_dir
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                          add_eos_token=False,  # 训练时不要自动加eos
                                          add_bos_token=False,
                                          use_fast=True)
tokenizer.eos_token_id = 1
tokenizer.pad_token_id = 0

if args.iteration_num > 1:
    model = f"output_iteration{args.iteration_num-1}"
    checkpoint = checkpoint_load(f"output_iteration{args.iteration_num-1}")
else:
    model = args.model_dir
    checkpoint = None

lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)
optimizer, model, scheduler = load_optimizer_scheduler(model, checkpoint, lr_list[args.iteration_num-1].item(), CONFIG)

training_args = GRPOConfig(
    output_dir=f"output_iteration{args.iteration_num}",
    logging_steps=100,
    beta=CONFIG["beta"],
    num_train_epochs=CONFIG["num_epochs"],
    learning_rate=lr_list[args.iteration_num-1].item(),
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
    torch_compile=False
)

print("model ", model)
trainer = pLM_GRPOTrainer(
    model=model,
    ref_model=args.model_dir,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, scheduler)
)

trainer.lr_scheduler       = scheduler
trainer.lr_scheduler_state = None

from src.utils import _optimizer_to_device

# 有些环境里 accelerate 的 device 要在这时取
try:
    train_device = trainer.accelerator.device
except Exception:
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_optimizer_to_device(optimizer, train_device)

print("fixed LR (optimizer) before training:", trainer.optimizer.param_groups[0]["lr"])
trainer.train()
trainer.save_model()
print("fixed LR (optimizer) after traning:", trainer.optimizer.param_groups[0]["lr"])

torch.cuda.empty_cache()
