# seq_gen_with_conf_filter_batched.py
import os
import math
import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm import tqdm


def remove_characters(sequence: str, char_list: List[str]) -> str:
    """移除训练时的特殊标记并取出 <sep> 后的主体序列（沿用你的逻辑）。"""
    columns = sequence.split("<sep>")
    seq = columns[1] if len(columns) > 1 else sequence
    for ch in char_list:
        seq = seq.replace(ch, "")
    return seq


@torch.inference_mode()
def calculate_perplexity(input_ids: torch.Tensor, model: GPT2LMHeadModel) -> float:
    """teacher-forcing PPL。input_ids: [L]"""
    outputs = model(input_ids.unsqueeze(0), labels=input_ids)
    loss = outputs.loss
    return math.exp(loss.item())


def token_confidence_from_scores(
    scores: List[torch.Tensor],
    sequences: torch.Tensor,
    prompt_len: int,
    *,
    k: int = 5,
    eos_id: int = 1,
    pad_id: int = 0,
) -> List[np.ndarray]:
    """
    根据论文定义计算逐步 token confidence：
        C_i = - (1/k) * sum_{j=1..k} log P_i(j)
    返回长度为 B 的列表，每个元素是该样本每步的 C_i 数组（到 EOS/末尾为止）。
    """
    B = sequences.size(0)
    T = len(scores)
    traces = [[] for _ in range(B)]
    for t in range(T):
        logprobs = F.log_softmax(scores[t], dim=-1)  # [B, V]
        topk_vals, _ = torch.topk(logprobs, k=k, dim=-1)  # [B, k]
        Ci = -topk_vals.mean(dim=-1)  # [B]
        step_token = sequences[:, prompt_len + t]
        for b in range(B):
            if step_token[b].item() == pad_id:
                continue
            traces[b].append(Ci[b].item())
    return [np.asarray(t, dtype=np.float32) for t in traces]


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if x.size < w:
        return np.array([], dtype=np.float32)
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0))
    return (c[w:] - c[:-w]) / w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--cwd", type=str, required=True)

    # 目标与批量
    parser.add_argument("--target_sequences", type=int, default=200,
                        help="希望最终保留下来的序列条数（通过筛选后计数）")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="一次 generate 的样本数，受显存限制设置")
    parser.add_argument("--max_batches", type=int, default=200,
                        help="最多尝试多少个批次（避免死循环）；总尝试数≈batch_size*max_batches")

    # 生成超参
    parser.add_argument("--top_p", type=float, default=1.0)      # nucleus 采样阈值 (0,1]
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_length", type=int, default=1014)

    # 置信度筛选超参
    parser.add_argument("--conf_k", type=int, default=5)         # token confidence 的 top-k
    parser.add_argument("--window_size", type=int, default=50)   # grouped confidence 滑窗
    parser.add_argument("--burn_in", type=int, default=50)       # 生成至少这么多步再开始判断
    # 论文的 C_i 越小越“自信”。对“窗口内 C_i 的均值”做阈值：<= conf_threshold 则通过
    parser.add_argument("--conf_threshold", type=float, default=1.8)

    args = parser.parse_args()

    iteration_num = args.iteration_num
    ec_label = args.label.strip()
    out_dir = args.cwd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Reading pretrained model and tokenizer")

    if iteration_num == 0:
        model_name = "/root/zymGFN/zymCTRL/"
    else:
        model_name = f"./output_iteration{iteration_num}"

    print(f"{model_name} loaded")
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/zymGFN/zymCTRL/"
    )
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    special_tokens = ["<start>", "<end>", "<|endoftext|>", "<pad>", " ", "<sep>"]
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    # 准备输出
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"seq_gen_{ec_label}_iteration{iteration_num}.fasta"
    out_path_cur = out_name
    out_path_cwd = os.path.join(out_dir, out_name)

    kept_records: List[str] = []  # 直接保存 FASTA 文本片段，减少内存占用
    kept_count = 0
    batches_tried = 0

    # 统一的 prompt
    base_input_ids = tokenizer.encode(ec_label, return_tensors="pt").to(device)

    pbar = tqdm(total=args.target_sequences, desc="accepted sequences")
    while kept_count < args.target_sequences and batches_tried < args.max_batches:
        batches_tried += 1

        # === 1) 单批生成 ===
        gen_out = model.generate(
            input_ids=base_input_ids,
            do_sample=True,
            top_p=max(min(args.top_p, 1.0), 1e-6),
            top_k=0,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            eos_token_id=1,
            pad_token_id=0,
            num_return_sequences=args.batch_size,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = gen_out.sequences            # [B, L]
        scores = gen_out.scores                  # list[T] of [B, V]
        B = sequences.size(0)
        prompt_len = base_input_ids.shape[1]

        # === 2) token-confidence + 滑窗筛选 ===
        traces = token_confidence_from_scores(
            scores, sequences, prompt_len,
            k=args.conf_k, eos_id=1, pad_id=0
        )
        accept_idx = []
        for b in range(B):
            Ci = traces[b]  # 越小越“自信”
            if Ci.size >= max(args.window_size, args.burn_in):
                window_mean = rolling_mean(Ci, args.window_size)
                if window_mean.size > 0 and window_mean[-1] <= args.conf_threshold:
                    accept_idx.append(b)

        # === 3) 清洗 + AA 过滤 + PPL + 追加到输出 ===
        if accept_idx:
            filtered_ids = sequences[accept_idx]
            decoded = tokenizer.batch_decode(filtered_ids, skip_special_tokens=False)
            for j, txt in zip(accept_idx, decoded):
                seq = remove_characters(txt, special_tokens)
                if not seq or not all(ch in canonical_amino_acids for ch in seq):
                    continue
                ppl = calculate_perplexity(sequences[j].to(device), model)
                record = f">{ec_label}_{batches_tried}_{j}_iteration{iteration_num}\t{ppl}\n{seq}\n"
                kept_records.append(record)
                kept_count += 1
                pbar.update(1)
                if kept_count >= args.target_sequences:
                    break

        # 及时释放分步 logits，避免显存/内存堆积
        del gen_out, sequences, scores, traces
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    # === 4) 落盘（写两份：当前目录 + 指定 cwd）===
    fasta_content = "".join(kept_records)
    with open(out_path_cur, "w") as f:
        f.write(fasta_content)
    with open(out_path_cwd, "w") as f:
        f.write(fasta_content)

    if kept_count < args.target_sequences:
        print(f"[warn] only kept {kept_count}/{args.target_sequences} after {batches_tried} batches")
    else:
        print(f"[done] kept {kept_count} sequences")

    print(f"wrote: {out_path_cur}")
    print(f"also:  {out_path_cwd}")


if __name__ == "__main__":
    main()
