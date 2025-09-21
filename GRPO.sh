#!/bin/bash
##################
# slurm settings #
##################
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=40000
#SBATCH --job-name sft_TMscore_4.2.1.1

start_epoch=$(date +%s)
echo [$(date +"%Y-%m-%d %H:%M:%S")] starting on $(hostname)

set -eu

###################
# set environment #
###################
# eval "$(conda shell.bash hook)"

###############
# run command #
###############
folder_path="/root/zymGFN/src/GRPO/"
label="3.1.1.1"
smiles='CC(=O)Oc1ccc(cc1)[N+](=O)[O-]'

echo "self-training of ZymCTRL for TMscore with ${label} started"

# ===== 从 0 开始；第 0 轮不训练，只产出 logs.csv 供第 1 轮训练 =====
for i in $(seq 0 30); do
  echo "========== Iteration ${i} =========="
  run_dir="${folder_path}output_iteration${i}"
  export RUN_DIR="${run_dir}"   # ← 每轮更新并导出
  LABEL="${label}"
  export LABEL                       # 让 YAML 能读到
  export ITER="${i}"                 # 每轮更新迭代号
  prev=$((i-1))
  prev_dir="${folder_path}output_iteration${prev}"
  export PREV_DIR="${prev_dir}"

  # -------- 训练（仅 i>0 时运行；读取上一轮 prev_dir/logs.csv）--------
  if [ "${i}" -gt 0 ]; then
    # conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_pt260
    # module unload compilers/cuda cudnn compilers/gcc
    # source /home/bingxing2/apps/package/pytorch/2.6.0-cu124-cp311/env.sh

    python "${folder_path}GRPO_train.py" \
      --config-path "${folder_path}conf" --config-name "grpo_train" \
      iteration_num="${i}" label="${label}" \
      hydra.run.dir="${run_dir}"

    # conda deactivate
  else
    echo "[Iter 0] cold-start: skip training; generating data for next iter"
  fi

  # -------- 生成序列（FASTA 写到 run_dir）--------
  echo "[Iter ${i}] Sequence generation"
  # module unload compilers/cuda cudnn compilers/gcc
  # module load compilers/gcc/11.3.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x
  # conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_cp310

  python "${folder_path}seq_gen.py" \
    --config-path "${folder_path}conf" --config-name "grpo_seq_gen" \
    iteration_num="${i}" label="${label}" \
    hydra.run.dir="${run_dir}"

  # -------- 折叠结构（PDB 写到 run_dir/PDB）--------
  echo "[Iter ${i}] Folding (ESMFold)"
  python "${folder_path}../utils/ESM_Fold.py" \
    --config-path "${folder_path}conf" --config-name "grpo_fold" \
    iteration_num="${i}" label="${label}" \
    hydra.run.dir="${run_dir}"

  # -------- Foldseek TM-score（输出 TM 文件到 run_dir）--------
  echo "[Iter ${i}] Foldseek TM-score vs 7atl (alpha)"
  tmp_dir="${run_dir}/tmp"
  mkdir -p "${tmp_dir}"

  foldseek easy-search \
    "${run_dir}/PDB" \
    "${folder_path}../../PDB/7atl.pdb" \
    "${run_dir}/alpha_${label}_TM_iteration${i}" \
    "${tmp_dir}" \
    --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" \
    --exhaustive-search 1 -e inf --tmscore-threshold 0.0

  # -------- 聚类（输入 FASTA 从 run_dir，输出到 run_dir/clustering）--------
  echo "[Iter ${i}] Alignments and cluster"
  tmp_dir="${run_dir}/tmp"
  mkdir -p "${tmp_dir}" "${run_dir}/clustering"

  mmseqs easy-cluster \
    "${run_dir}/seq_gen_${label}_iteration${i}.fasta" \
    "${run_dir}/clustering/clustResult_0.9_seq_gen_${label}_iteration${i}" \
    "${tmp_dir}" --min-seq-id 0.9

  rm -rf "${tmp_dir}"

  # -------- 毒性预测（ToxinPred2，输出到 run_dir）--------
  echo "[Iter ${i}] ToxinPred2"
  # conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/envs/matrics_yxu
  # cd /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/proteinGflownet/toxinpred/toxinpred2

  python "${folder_path}../utils/toxinpred2.py" \
    -i "${run_dir}/seq_gen_${label}_iteration${i}.fasta" \
    -o "${run_dir}/outfile${i}.csv" \
    -d 2

  # conda deactivate
  echo "ToxinPred2 skipped - conda environment not available"

  # -------- UniKP（kcat，输出到 run_dir）--------
  echo "[Iter ${i}] UniKP kcat"
  # conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/envs/UniKP_yxu_v2
  # cd /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/proteinGflownet/UniKP

  # GOMP=$(/home/bingxing2/ailab/scxlab0094/hantao/project/internTA/envs/UniKP_yxu_v2/bin/python - <<'PY'
# import sys, glob, os
# c = glob.glob(os.path.join(sys.prefix, "lib", "libgomp*.so*"))
# print(c[0] if c else "")
# PY
# )
  # LD_PRELOAD="$GOMP" \
  # PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "${folder_path}../utils/uniKP_fasta_train.py" \
    --fasta "${run_dir}/seq_gen_${label}_iteration${i}.fasta" \
    --smiles "${smiles}" \
    --task kcat \
    --out "${run_dir}/results_kcat${i}.csv" \
    --write_linear

  # conda deactivate
  echo "UniKP kcat skipped - conda environment not available"

  # -------- 合并生成当轮 logs.csv（写到 run_dir）--------
  echo "[Iter ${i}] dataset generation"
  # conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_pt260
  # module unload compilers/cuda cudnn compilers/gcc
  # source /home/bingxing2/apps/package/pytorch/2.6.0-cu124-cp311/env.sh

  python "${folder_path}../utils/dataset_gen_toxUnikp.py" \
    --config-path "${folder_path}conf" --config-name "grpo_dataset" \
    hydra.run.dir="${run_dir}"

  # conda deactivate

done

###############
# end message #
###############
end_epoch=$(date +%s)
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname) after $((end_epoch-start_epoch)) seconds
