# ESM_Fold_hydra.py —— 结构化配置（Hydra + dataclass）版本
from __future__ import annotations

import os
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING
from transformers import AutoTokenizer, EsmForProteinFolding


# =======================
# 1) 结构化配置（Schema）
# =======================
@dataclass
class ESMConfig:
    # Hugging Face 兼容目录（包含 tokenizer 和 folding 模型）
    model_dir: str = MISSING
    # 可选：强制设备，留空则自动选择 cuda/cpu
    device: Optional[str] = None
    # 内存优化选项
    max_sequence_length: int = 1000  # 最大序列长度，超过则跳过
    batch_size: int = 1  # 批处理大小
    use_mixed_precision: bool = True  # 使用混合精度
    clear_cache_frequency: int = 5  # 每处理多少个序列清理一次缓存


@dataclass
class PathsConfig:
    # 如果提供完整 FASTA 路径则优先使用；否则按模板在 sequences_dir 下拼接
    fasta_path: Optional[str] = None
    # FASTA 所在目录（相对 Hydra run 目录或绝对路径）
    sequences_dir: str = "."
    # 输出的 PDB 目录名（相对 Hydra run 目录），也可给绝对路径
    pdb_dir: str = "PDB"


@dataclass
class FoldConfig:
    iteration_num: int = MISSING
    label: str = MISSING
    esm: ESMConfig = field(default_factory=ESMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# 将 schema 注册给 Hydra
cs = ConfigStore.instance()
cs.store(name="fold_schema", node=FoldConfig)


# =======================
# 2) 主逻辑
# =======================
def _resolve_device(name: Optional[str]) -> torch.device:
    if name is None:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _get_gpu_memory_info():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    return 0, 0, 0


def _clear_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def _check_memory_available(required_gb: float = 2.0) -> bool:
    """检查是否有足够的内存可用"""
    if not torch.cuda.is_available():
        return True
    
    allocated, reserved, total = _get_gpu_memory_info()
    free = total - allocated
    return free >= required_gb


def _read_fasta_simple(fasta_file: Path) -> dict[str, str]:
    """
    简单读取 FASTA（与你原逻辑一致：一行序列）。
    如果你的 FASTA 可能多行序列，这里可升级为拼接多行。
    """
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_file}")
    sequences: dict[str, str] = {}
    name: Optional[str] = None
    with fasta_file.open("r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                name = line.strip()
                sequences[name] = ""
            else:
                if name is None:
                    continue
                sequences[name] = line.strip()
    return sequences


@hydra.main(version_base=None, config_path="conf", config_name="grpo_fold")
def main(cfg: FoldConfig):
    # -------- 目录与输入解析（默认相对 Hydra 的 run 目录 = Path.cwd()）--------
    run_dir = Path.cwd()
    # run_dir = Path(cfg.run_dir)  # This field doesn't exist in FoldConfig

    if cfg.paths.fasta_path:
        fasta_file = Path(to_absolute_path(cfg.paths.fasta_path)) \
                     if not os.path.isabs(cfg.paths.fasta_path) else Path(cfg.paths.fasta_path)
    else:
        fasta_name = f"seq_gen_{cfg.label}_iteration{cfg.iteration_num}.fasta"
        fasta_root = Path(cfg.paths.sequences_dir)
        if not fasta_root.is_absolute():
            fasta_root = run_dir / fasta_root
        fasta_file = fasta_root / fasta_name

    pdb_root = Path(cfg.paths.pdb_dir)
    if not pdb_root.is_absolute():
        pdb_root = run_dir / pdb_root
    pdb_root.mkdir(parents=True, exist_ok=True)

    print(f"[Hydra] run_dir  = {run_dir}")
    print(f"[Hydra] fasta   = {fasta_file}")
    print(f"[Hydra] pdb_dir = {pdb_root}")

    # -------- 设备 & 模型加载（目录从配置读取）--------
    device = _resolve_device(cfg.esm.device)
    model_dir_abs = to_absolute_path(cfg.esm.model_dir) \
                    if not os.path.isabs(cfg.esm.model_dir) else cfg.esm.model_dir

    print(f"[ESM] loading from: {model_dir_abs}")
    print(f"[MEMORY] Initial GPU memory: {_get_gpu_memory_info()}")
    
    # 设置PyTorch内存管理
    if device.type == "cuda":
        torch.cuda.empty_cache()
        # 设置内存分配策略
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    tokenizer_esm = AutoTokenizer.from_pretrained(model_dir_abs)
    model_esm = EsmForProteinFolding.from_pretrained(model_dir_abs).to(device)
    
    # 启用混合精度
    if cfg.esm.use_mixed_precision and device.type == "cuda":
        model_esm = model_esm.half()
        print("[MEMORY] Using mixed precision (FP16)")
    
    model_esm.eval()
    print(f"[MEMORY] After model loading: {_get_gpu_memory_info()}")

    # -------- 读取 FASTA 并折叠 --------
    sequences = _read_fasta_simple(fasta_file)
    print(f"[FASTA] {len(sequences)} sequences loaded")

    count, error, skipped = 0, 0, 0
    for name, sequence in sequences.items():
        try:
            count += 1
            
            # 检查序列长度
            if len(sequence) > cfg.esm.max_sequence_length:
                print(f"[SKIP] sequence '{name}' too long ({len(sequence)} > {cfg.esm.max_sequence_length})")
                skipped += 1
                continue
            
            # 检查内存可用性
            if not _check_memory_available(required_gb=2.0):
                print(f"[SKIP] sequence '{name}' - insufficient memory")
                skipped += 1
                _clear_memory()
                continue
            
            print(f"[PROCESSING] {name} (length: {len(sequence)})")
            print(f"[MEMORY] Before processing: {_get_gpu_memory_info()}")
            
            with torch.no_grad():
                # 使用混合精度进行推理
                if cfg.esm.use_mixed_precision and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        output_pdb = model_esm.infer_pdb(sequence)
                else:
                    output_pdb = model_esm.infer_pdb(sequence)
                
                # 清洗 FASTA 名（去掉 '>' 与可能的制表符）
                clean = name[1:] if name.startswith(">") else name
                clean = clean.split("\t")[0]

                out_path = pdb_root / f"{clean}.pdb"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w") as f:
                    f.write(output_pdb)

            # 定期清理内存
            if count % cfg.esm.clear_cache_frequency == 0:
                _clear_memory()
                print(f"[MEMORY] After cleanup: {_get_gpu_memory_info()}")

            if count % 10 == 0 or count == len(sequences):
                print(f"[PROGRESS] processed {count}/{len(sequences)}, errors: {error}, skipped: {skipped}")

        except Exception as e:
            error += 1
            print(f"[ERROR] sequence '{name}' failed: {e!r}")
            _clear_memory()

    # 释放模型
    del model_esm
    _clear_memory()

    print(f"[DONE] total={len(sequences)}, processed={count}, ok={count-error-skipped}, error={error}, skipped={skipped}")
    print(f"[MEMORY] Final GPU memory: {_get_gpu_memory_info()}")


if __name__ == "__main__":
    main()
'''
python ESM_Fold_hydra.py \
  iteration_num=${i} \
  label="${label}" \
  esm.model_dir="/home/.../esm_fold" \
  paths.sequences_dir="." \
  paths.pdb_dir="PDB" \
  hydra.run.dir="${folder_path}output_iteration${i}"
'''