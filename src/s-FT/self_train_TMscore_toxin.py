import os
import math
import random
import subprocess
import argparse
import torch
import pandas as pd
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


# ---------- helpers ----------
def _clean_name(s: str) -> str:
    """统一名称（用于对齐 FASTA 与 CSV）：
    - 去掉前后空白
    - 只取第一个 <TAB> 之前的部分
    - 去掉行首 '>' 和外层引号
    """
    return str(s).strip().split("\t")[0].lstrip(">").strip().strip('"')


def _load_tox_map(tox_csv: str):
    """读取 ToxinPred2 输出，返回 {name: {'ml': float, 'pred': str}}。
    自动识别分隔符（制表符/逗号），支持列名大小写不敏感：
    必需列：ID、ML_Score、Prediction（或相近命名）
    """
    if tox_csv is None or not os.path.exists(tox_csv):
        return {}

    try:
        df = pd.read_csv(tox_csv, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(tox_csv, sep="\t")

    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("id") or cols.get("name") or list(df.columns)[0]
    sc = cols.get("ml_score") or cols.get("score")  # ML_Score 列
    pc = cols.get("prediction")                      # Prediction 列

    m = {}
    for _, r in df.iterrows():
        name = _clean_name(r[idc])
        ml = float(r[sc]) if sc in df.columns and pd.notna(r[sc]) else 0.0
        pred = str(r[pc]).strip() if pc in df.columns and pd.notna(r[pc]) else ""
        m[name] = {"ml": ml, "pred": pred}
    return m


def _length_reward(seq_len: int, center: float = 260.0, sigma: float = 0.5) -> float:
    """长度奖励（高斯）：以 center 为峰值，默认 260 氨基酸。"""
    x = (seq_len / center) - 1.0
    return math.exp(-((x * x) / (sigma * sigma)))


# ---------- dataset builder ----------
def generate_dataset(
    generated_sequences: str,
    TMscores: str,
    tox_csv: str = None,
    tox_mode: str = "penalize",    # "penalize" or "filter"
    tox_alpha: float = 1.0,        # 惩罚强度
    use_length_rew: bool = True,
    len_center: float = 260.0,
    len_sigma: float = 0.5,
):
    """将 FASTA + Foldseek TM 结果 + ToxinPred2 结果融合成训练样本，并计算 weight。"""
    data = {"sequence": [], "seq_name": [], "TM": [], "TM_norm_que": [], "weight": []}

    # 1) 读 FASTA
    with open(generated_sequences, "r") as f:
        rep_seq = f.readlines()
    sequences_rep = {}
    cur_name = None
    for line in rep_seq:
        if line.startswith(">"):
            cur_name = _clean_name(line)
        else:
            if cur_name is not None:
                sequences_rep[cur_name] = line.strip()

    # 2) 读毒性表
    tox_map = _load_tox_map(tox_csv)

    # 3) 读 Foldseek TM 输出（query, target, alntmscore, qtmscore, ttmscore, alnlen）
    with open(TMscores, "r") as f:
        lines = f.readlines()

    # 跳过可能的表头
    if lines and lines[0].lower().startswith("query"):
        lines = lines[1:]

    for entry in lines:
        parts = entry.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue

        name = _clean_name(parts[0])

        # 名字必须能在 FASTA 映射到具体序列
        if name not in sequences_rep:
            continue

        try:
            TM = float(parts[2])          # alntmscore
            TM_norm_que = float(parts[4]) # 常用作为“query 归一化”的 TM（如需改用 qtmscore，换成 parts[3]）
            algn = int(parts[5])          # alignment length
        except Exception:
            continue

        seq = sequences_rep[name]
        length_rew = _length_reward(len(seq), len_center, len_sigma) if use_length_rew else 1.0
        base_weight = (TM_norm_que + (algn / 100.0)) * length_rew

        # 4) 并入毒性
        tox_factor = 1.0
        if name in tox_map:
            ml = max(0.0, min(1.0, float(tox_map[name]["ml"])))
            pred = str(tox_map[name]["pred"]).lower()

            if tox_mode == "filter":
                # 预测为毒性的直接丢弃（严格判断：包含 'toxin' 且不包含 'non'）
                if ("toxin" in pred) and ("non" not in pred):
                    continue
            else:
                # 惩罚：越毒（ml 越接近 1），乘子越小
                tox_factor = (1.0 - ml) ** float(tox_alpha)

        weight = base_weight * tox_factor

        data["sequence"].append(seq)
        data["seq_name"].append(name)
        data["TM"].append(TM)
        data["TM_norm_que"].append(TM_norm_que)
        data["weight"].append(weight)

    return data


# ---------- grouping & tokenization ----------
def grouper(iterable):
    """把 (长度估计, 文本) 组成的条目拼到不超过 1024-token 的块里。"""
    prev = None
    group = ''
    total_sum = 0
    for item in iterable:
        if prev is None or item[0] + total_sum < 1025:
            group += item[1]
            total_sum += item[0]
        else:
            total_sum = item[0]
            yield group
            group = item[1]
        prev = item
    if group:
        yield group


def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples["text"])
    # 这里保留原始提示信息
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked."
        )
    return output


def group_texts(examples, block_size=1024):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# ---------- main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--cwd", type=str, required=True)

    # 新增：毒性融合相关
    parser.add_argument("--tox_csv", type=str, default=None,
                        help="ToxinPred2 输出 CSV/TSV 路径（不传则不融合毒性）")
    parser.add_argument("--tox_mode", type=str, default="penalize",
                        choices=["penalize", "filter"], help="penalize=按 ML_Score 惩罚；filter=直接丢弃毒性样本")
    parser.add_argument("--tox_alpha", type=float, default=1.0, help="惩罚强度，权重乘 (1-ML_Score)^alpha")

    # 新增：长度奖励参数
    parser.add_argument("--use_length_reward", action="store_true", default=True,
                        help="是否使用长度奖励（高斯，中心=260，sigma=0.5）")
    parser.add_argument("--len_center", type=float, default=260.0)
    parser.add_argument("--len_sigma", type=float, default=0.5)

    args = parser.parse_args()

    iteration_num = int(args.iteration_num)
    label = str(args.label)
    cwd = args.cwd if args.cwd.endswith("/") else (args.cwd + "/")

    # 路径
    generated_sequences = f'{cwd}generated_sequences/seq_gen_{label}_iteration{iteration_num-1}.fasta'
    TMscores = f'{cwd}TMscores/{label}_TM_iteration{iteration_num-1}'
    tox_csv = args.tox_csv or f'{cwd}toxicity/{label}_tox_iteration{iteration_num-1}.csv'

    # 构造训练数据
    data = generate_dataset(
        generated_sequences,
        TMscores,
        tox_csv=tox_csv,
        tox_mode=args.tox_mode,
        tox_alpha=args.tox_alpha,
        use_length_rew=bool(args.use_length_reward),
        len_center=args.len_center,
        len_sigma=args.len_sigma,
    )

    df = pd.DataFrame(data)
    len_all = len(df)

    # 选 top ~200 条（按 weight）
    df = df.sort_values(by=['weight'], ascending=False)[['sequence', 'seq_name', 'weight']]
    best_sequences = df.iloc[:51, :]['sequence'].to_list()  # 约 200 条
    random.shuffle(best_sequences)
    print(f'{len(best_sequences)} sequences out of {len_all} ready to finetune.')

    # 释放内存
    del df

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/bingxing2/ailab/group/ai4earth/hantao/project/internTA/proteinGflownet/ZymCTRL_local')

    # 把序列打包到 1024 token 的块中
    processed_dataset = []
    for seq in best_sequences:
        sequence = seq.strip()
        separator = '<sep>'
        control_code_len = len(tokenizer(label + separator)['input_ids'])
        available_space = 1021 - control_code_len  # 1024 减去 special tokens

        if len(sequence) > available_space:
            total_length = control_code_len + len(sequence[:available_space]) + 1
            text = f"{label}{separator}{sequence[:available_space]}<|endoftext|>"
            processed_dataset.append((total_length, text))
        else:
            total_length = control_code_len + len(sequence) + 3
            text = f"{label}{separator}<start>{sequence}<end><|endoftext|>"
            processed_dataset.append((total_length, text))

    grouped_dataset = dict(enumerate(grouper(processed_dataset), 1))

    # 写出训练文本
    txt_path = f"{cwd}{label}_processed.txt"
    with open(txt_path, 'w') as fn:
        for _, value in grouped_dataset.items():
            padding_len = 1024 - len(tokenizer(value)['input_ids'])
            padding = "<pad>" * max(0, padding_len)
            fn.write(value + padding + "\n")

    # TOKENIZE
    data_files = {"train": txt_path}
    extension = "text"
    global tok_logger  # 被 tokenize_function 使用
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir='.')
    raw_datasets["train"] = load_dataset(
        extension, data_files=data_files, split=f"train[10%:]", cache_dir='.'
    )
    raw_datasets["validation"] = load_dataset(
        extension, data_files=data_files, split=f"train[:10%]", cache_dir='.'
    )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=32,
        remove_columns=['text'],
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    block_size = 1024
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=124,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    os.makedirs(f'{cwd}dataset', exist_ok=True)
    train_dataset.save_to_disk(f'{cwd}dataset/train2')
    eval_dataset.save_to_disk(f'{cwd}dataset/eval2')

    # 释放内存
    del tokenizer, train_dataset, eval_dataset, tokenized_datasets, lm_datasets, raw_datasets
    del best_sequences, grouped_dataset, processed_dataset

    # Finetune（保持你的原流程）
    print('starting finetune')
    os.makedirs(f"{cwd}models", exist_ok=True)

    if iteration_num == 1:
        subprocess.run([
            "python", f"{cwd}finetuner.py",
            "--tokenizer_name", "/home/bingxing2/ailab/group/ai4earth/hantao/project/internTA/proteinGflownet/ZymCTRL_local",
            "--model_name_or_path", "/home/bingxing2/ailab/group/ai4earth/hantao/project/internTA/proteinGflownet/ZymCTRL_local",
            "--load_best_model_at_end",
            "--do_train", "--do_eval",
            "--output_dir", f"{cwd}models/{label}_model{iteration_num}",
            "--evaluation_strategy", "steps",
            "--eval_steps", "10", "--logging_steps", "2", "--save_steps", "10",
            "--num_train_epochs", "25",
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--cache_dir", ".",
            "--learning_rate", "0.8e-06",
            "--dataloader_drop_last", "False",
            "--save_total_limit", "1",
        ])
    else:
        ckpt_dir = f'{cwd}models/{label}_model{iteration_num-1}'
        checkpoint_folder = [x for x in os.listdir(ckpt_dir) if 'checkpoint' in x][0]
        ckpt_path = f"{ckpt_dir}/{checkpoint_folder}"
        subprocess.run([
            "python", f"{cwd}scripts/finetuner.py",
            "--tokenizer_name", ckpt_path,
            "--model_name_or_path", ckpt_path,
            "--load_best_model_at_end",
            "--do_train", "--do_eval",
            "--output_dir", f"{cwd}models/{label}_model{iteration_num}",
            "--evaluation_strategy", "steps",
            "--eval_steps", "10", "--logging_steps", "2", "--save_steps", "10",
            "--num_train_epochs", "25",
            "--per_device_train_batch_size", "4",
            "--per_device_eval_batch_size", "1",
            "--cache_dir", ".",
            "--learning_rate", "0.8e-06",
            "--dataloader_drop_last", "False",
            "--save_total_limit", "1",
        ])

    print(f'round {iteration_num} of finetuning performed')
