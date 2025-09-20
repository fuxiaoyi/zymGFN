#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, gc, math, pickle
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch

# === 你已有的依赖（来自 SMILES Transformer 项目）===
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split  # SMILES 分词

# === ProtT5（来自 ProtTrans / HF Transformers）===
from transformers import T5EncoderModel, T5Tokenizer


# --------------------------
# 工具：读 FASTA（无第三方依赖）
# --------------------------
def read_fasta(path: str) -> List[Tuple[str, str]]:
    ids, seqs = [], []
    cur_id, cur_seq = None, []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur_seq))
                cur_id = line[1:].split()[0]   # 取 '>' 后首个 token 作为 ID
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        ids.append(cur_id)
        seqs.append("".join(cur_seq))
    return list(zip(ids, seqs))


# --------------------------
# ProtT5 序列嵌入（批处理 + 均值池化）
# --------------------------
def seqs_to_vec(
    seqs: List[str],
    prot_t5_path: str = "prot_t5_xl_uniref50",  # 若你用 HF Hub，建议改成 "Rostlab/prot_t5_xl_uniref50"
    device_str: str = None,
    batch_size: int = 4,
    max_len_clip: int = 1000,
) -> np.ndarray:
    """
    返回形状 [N, D] 的均值池化 embedding（与你现有 Seq_to_vec 保持一致逻辑：>1000 截断为头500+尾500；UZOB -> X）
    """
    if device_str is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 预处理：长度裁剪 + 替换无效氨基酸
    proc = []
    for s in seqs:
        if len(s) > max_len_clip:
            s = s[: max_len_clip // 2] + s[-max_len_clip // 2 :]
        s = re.sub(r"[UZOB]", "X", s)
        # ProtT5 习惯用空格分隔字母
        proc.append(" ".join(list(s)))

    # 加载模型（一次）
    tokenizer = T5Tokenizer.from_pretrained(prot_t5_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(prot_t5_path)
    model.to(device).eval()

    feats = []
    with torch.no_grad():
        for i in range(0, len(proc), batch_size):
            batch = proc[i : i + batch_size]
            enc = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, T, H]
            # 均值池化（去掉 padding；与旧代码保持一致，去掉最后一个 token）
            for b in range(out.size(0)):
                valid_len = int(attention_mask[b].sum().item())
                # 保险起见，限制 >0
                take = max(valid_len - 1, 1)
                mean_vec = out[b, :take, :].mean(dim=0).detach().cpu().numpy()
                feats.append(mean_vec)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(feats, axis=0)


# --------------------------
# SMILES 嵌入（批处理）
# --------------------------
class SmilesEmbedder:
    def __init__(self, vocab_path="vocab.pkl", trfm_path="trfm_12_23000.pkl"):
        self.vocab = WordVocab.load_vocab(vocab_path)
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3

        self.trfm = TrfmSeq2seq(len(self.vocab), 256, len(self.vocab), 4)
        self.trfm.load_state_dict(torch.load(trfm_path, map_location="cpu"))
        self.trfm.eval()

    def _encode_one(self, sm_tokens: List[str], seq_len: int = 220) -> Tuple[List[int], List[int]]:
        # 与你现有函数保持一致：过长-> 截断为 109+109
        if len(sm_tokens) > 218:
            sm_tokens = sm_tokens[:109] + sm_tokens[-109:]
        ids = [self.vocab.stoi.get(tok, self.unk_index) for tok in sm_tokens]
        ids = [self.sos_index] + ids + [self.eos_index]
        seg = [1] * len(ids)
        padding = [self.pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        return ids, seg

    def embed(self, smiles_list: List[str]) -> np.ndarray:
        x_id, x_seg = [], []
        for sm in smiles_list:
            toks = split(sm)  # SMILES 分词
            ids, seg = self._encode_one(toks)
            x_id.append(ids)
            x_seg.append(seg)
        xid = torch.tensor(x_id)          # [N, L]
        # SMILES Transformer 代码里用了转置后 encode
        X = self.trfm.encode(torch.t(xid))  # [L, N, D] or [N, D]? 依赖实现；与你现有 smiles_to_vec 保持一致
        # 若 encode 返回 [N, D]，下面直接转 numpy；若是张量别的形状，请按你本地实现保留
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return X


# --------------------------
# 主流程
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Predict kinetic label for all sequences in a FASTA.")
    ap.add_argument("--fasta", required=True, help="Input FASTA file.")
    # 二选一：--smiles（所有序列同一底物）或 --smiles_map（CSV 两列：id,smiles）
    ap.add_argument("--smiles", default=None, help="One SMILES used for ALL sequences.")
    ap.add_argument("--smiles_map", default=None, help="CSV with columns: id,smiles (align by FASTA IDs).")
    # UniKP 模型选择
    ap.add_argument("--task", choices=["kcat", "Km", "kcat_Km"], default="kcat")
    ap.add_argument("--model_pickle", default=None, help="Override path to UniKP pickle (if not using default).")
    # ProtT5 / 设备 / 批量
    ap.add_argument("--prot_t5_path", default="prot_t5_xl_uniref50",
                    help='Path or HF id for ProtT5 (e.g. "Rostlab/prot_t5_xl_uniref50").')
    ap.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu" (auto if not set).')
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for ProtT5 embedding.")
    # 输出
    ap.add_argument("--out", required=True, help="Output file (.csv or .xlsx).")
    ap.add_argument("--write_linear", action="store_true",
                    help="Write linear value (10**log10). Always writes raw_log10 too.")
    args = ap.parse_args()

    # 读 FASTA
    fasta_items = read_fasta(args.fasta)  # [(id, seq), ...]
    if len(fasta_items) == 0:
        raise RuntimeError("FASTA 为空或无法解析。")

    ids = [i for i, _ in fasta_items]
    sequences = [s for _, s in fasta_items]

    # 准备 SMILES 列表
    if args.smiles_map:
        df_map = pd.read_csv(args.smiles_map)
        if not {"id", "smiles"}.issubset(df_map.columns.str.lower()):
            # 容错：大小写不敏感
            cols = {c.lower(): c for c in df_map.columns}
            id_col = cols.get("id")
            sm_col = cols.get("smiles")
            if id_col is None or sm_col is None:
                raise ValueError("`--smiles_map` 必须包含列：id,smiles")
        else:
            id_col = "id"
            sm_col = "smiles"

        mp = dict(zip(df_map[id_col], df_map[sm_col]))
        smiles_list = []
        missing = []
        for _id in ids:
            if _id in mp:
                smiles_list.append(mp[_id])
            else:
                missing.append(_id)
        if missing:
            raise ValueError(f"以下 FASTA ID 在 --smiles_map 中没有匹配的 SMILES：{missing[:10]} ... 共 {len(missing)} 条")
    else:
        if not args.smiles:
            raise ValueError("需要提供 --smiles（所有序列同一底物）或 --smiles_map（二选一）。")
        smiles_list = [args.smiles] * len(ids)

    # 嵌入：序列（ProtT5）
    seq_vec = seqs_to_vec(
        sequences,
        prot_t5_path=args.prot_t5_path,
        device_str=args.device,
        batch_size=args.batch_size,
    )  # [N, D1]

    # 嵌入：SMILES
    sm_embedder = SmilesEmbedder(vocab_path="vocab.pkl", trfm_path="trfm_12_23000.pkl")
    smiles_vec = sm_embedder.embed(smiles_list)  # [N, D2]

    if seq_vec.shape[0] != smiles_vec.shape[0]:
        raise RuntimeError(f"样本数不一致：seq_vec={seq_vec.shape}, smiles_vec={smiles_vec.shape}")

    fused = np.concatenate([smiles_vec, seq_vec], axis=1)  # [N, D1+D2]

    # 加载 UniKP 模型
    if args.model_pickle:
        model_path = args.model_pickle
    else:
        # 按任务选择默认 pickle 文件名
        default_map = {
            "kcat": "UniKP/UniKP for kcat.pkl",
            "Km": "UniKP/UniKP for Km.pkl",
            "kcat_Km": "UniKP/UniKP for kcat_Km.pkl",
        }
        model_path = default_map[args.task]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 预测
    y_log10 = model.predict(fused)  # UniKP 输出为 log10
    if isinstance(y_log10, (list, tuple)):
        y_log10 = np.asarray(y_log10)
    out_df = pd.DataFrame({
        "id": ids,
        "sequence": sequences,
        "smiles": smiles_list,
        "pred_log10": y_log10,
    })
    if args.write_linear:
        out_df["pred_linear"] = np.power(10.0, y_log10)

    # 保存
    if args.out.lower().endswith(".xlsx"):
        out_df.to_excel(args.out, index=False)
    else:
        out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  (N={len(out_df)})")


if __name__ == "__main__":
    main()
'''
python predict_kinetics_from_fasta.py \
  --fasta inputs/proteins.fasta \
  --smiles "OC1=CC=C(C[C@@H](C(O)=O)N)C=C1" \
  --task kcat \
  --out results_kcat.csv \
  --write_linear


# smiles_map.csv 两列：id,smiles；id 必须与 FASTA 的 >ID 对齐
python predict_kinetics_from_fasta.py \
  --fasta inputs/proteins.fasta \
  --smiles_map inputs/smiles_map.csv \
  --task Km \
  --out results_Km.xlsx \
  --write_linear


'''