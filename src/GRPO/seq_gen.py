# seq_gen_hydra.py —— 结构化配置（Hydra + dataclass）版本
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel


# =======================
# 1) 结构化配置（Schema）
# =======================
@dataclass
class ModelConfig:
    # 初始（第 0 轮）使用的底座模型目录（包含权重）
    base_model_dir: str = MISSING
    # 分词器目录（通常与 base_model_dir 相同）
    tokenizer_dir: Optional[str] = None
    # 设备：None 则自动选择 cuda/cpu
    device: Optional[str] = None


@dataclass
class PathsConfig:
    # 输出 FASTA 的目录；相对路径则基于 Hydra 的 run 目录
    out_dir: str = "."
    # 输出文件名模板
    fasta_name_tmpl: str = "seq_gen_{label}_iteration{iteration_num}.fasta"


@dataclass
class GenConfig:
    # 迭代号与标签（与流水线一致）
    iteration_num: int = MISSING
    label: str = MISSING

    # 采样相关参数（可按需暴露更多）
    top_k: int = 9
    repetition_penalty: float = 1.2
    max_length: int = 1014          # 注意：包含 prompt 的总长度
    num_return_sequences: int = 20  # 每次生成的序列数
    num_batches: int = 10           # 重复生成的批次数

    # 特殊 token（用于清理）
    special_tokens: Tuple[str, ...] = ("<start>", "<end>", "<|endoftext|>", "<pad>", " ", "<sep>")

    # 仅允许的氨基酸集合（过滤）
    allowed_aas: str = "ACDEFGHIKLMNPQRSTVWY"

    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        base_model_dir=MISSING,
        tokenizer_dir=None,
        device=None,
    ))
    paths: PathsConfig = field(default_factory=PathsConfig)


# 将 schema 注册给 Hydra
cs = ConfigStore.instance()
cs.store(name="seq_gen_schema", node=GenConfig)


# =======================
# 2) 工具函数
# =======================
def _resolve_device(name: Optional[str]) -> torch.device:
    if name is None:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def remove_characters(sequence: str, char_list: List[str]) -> str:
    """
    移除训练时使用的特殊 token。
    原逻辑：输入是一段包含 '<sep>' 的文本，取第二段作为序列，再做替换。
    """
    columns = sequence.split("<sep>")
    seq = columns[1] if len(columns) > 1 else sequence
    for ch in char_list:
        seq = seq.replace(ch, "")
    return seq


@torch.no_grad()
def calculate_perplexity(input_ids: torch.Tensor, model: GPT2LMHeadModel) -> float:
    """
    计算单条序列的困惑度（PPL）。
    注意：此处把 input_ids 作为 labels 计算自回归 loss。
    """
    outputs = model(input_ids.unsqueeze(0), labels=input_ids)
    loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs.loss
    return math.exp(loss.item())


def _all_chars_allowed(seq: str, allowed: set[str]) -> bool:
    return all(ch in allowed for ch in seq)


# =======================
# 3) 主流程
# =======================
@hydra.main(version_base=None, config_path=None, config_name="seq_gen_schema")
def main(cfg: GenConfig):
    # run_dir = Path.cwd()
    run_dir = Path(cfg.run_dir)

    # 解析路径
    out_root = Path(cfg.paths.out_dir)
    if not out_root.is_absolute():
        out_root = run_dir / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    fasta_name = cfg.paths.fasta_name_tmpl.format(
        label=cfg.label, iteration_num=cfg.iteration_num
    )
    fasta_path = out_root / fasta_name

    # 设备
    device = _resolve_device(cfg.model.device)

    # 模型与分词器目录
    base_model_dir = (
        cfg.model.base_model_dir
        if os.path.isabs(cfg.model.base_model_dir)
        else to_absolute_path(cfg.model.base_model_dir)
    )
    tokenizer_dir = (
        cfg.model.tokenizer_dir or cfg.model.base_model_dir
    )
    tokenizer_dir = (
        tokenizer_dir if os.path.isabs(tokenizer_dir) else to_absolute_path(tokenizer_dir)
    )

    # 选择权重来源：
    # - 第 0 轮：用 base_model_dir
    # - 第 >0 轮：用本轮（已训练完成）的输出目录，即 run_dir
    if cfg.iteration_num == 0:
        model_dir_for_gen = base_model_dir
    else:
        # 你的流水线里：训练完成后本轮产物就在 run_dir（例如 output_iteration{i}）
        model_dir_for_gen = str(run_dir)

    print(f"[seq_gen] tokenizer_dir = {tokenizer_dir}")
    print(f"[seq_gen] model_dir_for_gen = {model_dir_for_gen}")
    print(f"[seq_gen] fasta_out = {fasta_path}")

    # 加载分词器与模型
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model = GPT2LMHeadModel.from_pretrained(model_dir_for_gen).to(device)
    model.eval()

    # 生成设置
    eos_id = tokenizer.eos_token_id or 1
    pad_id = tokenizer.pad_token_id or 0
    label_prompt = cfg.label.strip()
    allowed_set = set(cfg.allowed_aas)

    # 逐批生成
    all_sequences: List[Dict] = []
    for b in tqdm(range(cfg.num_batches), desc="Generating"):
        input_ids = tokenizer.encode(label_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            max_length=cfg.max_length,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            do_sample=True,
            num_return_sequences=cfg.num_return_sequences,
        )

        # 仅保留未被截断（以 pad 结尾）的序列
        new_outputs = [o for o in outputs if int(o[-1].item()) == pad_id]
        if not new_outputs:
            print("[WARN] this batch produced no short (unpadded-end) sequences")

        # 计算 perplexity
        batch_items: List[Tuple[str, float]] = []
        for output in new_outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=False)
            ppl = calculate_perplexity(output, model)
            clean_seq = remove_characters(decoded, cfg.special_tokens)
            if _all_chars_allowed(clean_seq, allowed_set):
                batch_items.append((clean_seq, float(ppl)))

        # 构造 fasta 片段
        for idx, (seq, ppl) in enumerate(batch_items):
            fasta_rec = f">{cfg.label}_{b}_{idx}_iteration{cfg.iteration_num}\t{ppl}\n{seq}\n"
            all_sequences.append(
                {
                    "label": cfg.label,
                    "batch": b,
                    "index": idx,
                    "pepr": ppl,
                    "fasta": fasta_rec,
                }
            )

        # 释放显存
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 写出 FASTA
    fasta_content = "".join(item["fasta"] for item in all_sequences)
    fasta_path.write_text(fasta_content, encoding="utf-8")

    print(f"[DONE] total_records={len(all_sequences)} -> {fasta_path}")


if __name__ == "__main__":
    main()

'''
python seq_gen_hydra.py \
  iteration_num=${i} \
  label="${label}" \
  model.base_model_dir="/root/zymGFN/zymCTRL/" \
  model.tokenizer_dir="/root/zymGFN/zymCTRL/" \
  paths.out_dir="." \
  hydra.run.dir="${folder_path}output_iteration${i}"

'''