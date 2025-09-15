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

set -euo pipefail

###################
# set environment #
###################
# 初始化 conda（按你本机 Anaconda/Miniconda 路径修改）
eval "$(conda shell.bash hook)"

# foldseek 安装目录（把下面改成你的 foldseek 根目录）
#your_local_foldseek_path="/home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_cp310"
#export PATH="${your_local_foldseek_path}/foldseek/bin:${PATH}"
#command -v foldseek >/dev/null || { echo "ERROR: foldseek 未在 PATH 中"; exit 1; }

###############
# run command #
###############
folder_path="/home/bingxing2/ailab/scxlab0094/hantao/project/internTA/proteinGflownet/ProtRL/experiments/Self-finetuning/s-FT_TM-score/"
label="3.1.1.1"
smiles='CC(=O)Oc1ccc(cc1)[N+](=O)[O-]'


echo "self-training of ZymCTRL for TMscore with ${label} started"

for i in $(seq 0 30); do
  echo "========== Iteration ${i} =========="

  # -------- 训练（从第 1 轮开始，用上一轮 i-1 的 CSV） --------
  if [ "${i}" -ne 0 ]; then
    prev=$((i-1))
    echo "[Iter ${i}] Train started (use tox/kcat from iter ${prev})"
    conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_cp310
    cd "${folder_path}"

    python "${folder_path}self_train_TMscore_toxin_unikp.py" \
      --iteration_num "${i}" \
      --label "${label}" \
      --cwd "${folder_path}" \
      --tox_csv "${folder_path}generated_sequences/outfile${prev}.csv" \
      --tox_mode penalize \
      --tox_alpha 1.0 \
      --use_length_reward \
      --len_center 260 --len_sigma 0.5 \
      --kcat_csv "${folder_path}generated_sequences/results_kcat${prev}.csv" \
      --kcat_col pred_linear \
      --kcat_beta 1.0

    conda deactivate
  else
    # 第 0 轮：创建目录
    echo "[Iter 0] Init folders"
    mkdir -p "${folder_path}generated_sequences" \
             "${folder_path}PDB" \
             "${folder_path}TMscores" \
             "${folder_path}models" \
             "${folder_path}dataset"
  fi

  # -------- 生成序列 --------
  echo "[Iter ${i}] Sequence generation"
  conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_cp310
  cd "${folder_path}"

  python "${folder_path}seq_gen.py" \
    --iteration_num "${i}" \
    --label "${label}" \
    --cwd "${folder_path}"

  # -------- 折叠结构（ESMFold）--------
  echo "[Iter ${i}] Folding (ESMFold)"
  python "${folder_path}esmfold.py" \
    --iteration_num "${i}" \
    --label "${label}" \
    --cwd "${folder_path}"

  # -------- 计算 TM 分数（Foldseek）--------
  echo "[Iter ${i}] Foldseek TM-score vs 7atl (alpha)"
  tmp_dir="${folder_path}tmp_${i}"
  mkdir -p "${tmp_dir}"
  foldseek easy-search \
    "${folder_path}PDB/${label}_output_iteration${i}" \
    "${folder_path}7atl.pdb" \
    "${folder_path}TMscores/${label}_TM_iteration${i}" \
    "${tmp_dir}" \
    --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" \
    --exhaustive-search 1 -e inf --tmscore-threshold 0.0
  rm -rf "${tmp_dir}"

  conda deactivate

  # -------- 毒性预测（ToxinPred2）--------
  echo "[Iter ${i}] ToxinPred2"
  conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/envs/matrics_yxu
  cd /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/proteinGflownet/toxinpred/toxinpred2

  python toxinpred2.py \
    -i "${folder_path}generated_sequences/seq_gen_${label}_iteration${i}.fasta" \
    -o "${folder_path}generated_sequences/outfile${i}.csv" \
    -d 2
  conda deactivate

  # -------- UniKP（kcat）--------
  echo "[Iter ${i}] UniKP kcat"
  conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/envs/UniKP_yxu_v2
  cd /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/proteinGflownet/UniKP

  GOMP=$(/home/bingxing2/ailab/scxlab0094/hantao/project/internTA/envs/UniKP_yxu_v2/bin/python - <<'PY'
import sys, glob, os
c = glob.glob(os.path.join(sys.prefix, "lib", "libgomp*.so*"))
print(c[0] if c else "")
PY
)
  LD_PRELOAD="$GOMP" \
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/envs/UniKP_yxu_v2/bin/python \
    /home/bingxing2/ailab/group/ai4earth/hantao/project/internTA/proteinGflownet/UniKP/UniKP_fasta_train.py \
    --fasta "${folder_path}generated_sequences/seq_gen_${label}_iteration${i}.fasta" \
    --smiles "${smiles}" \
    --task kcat \
    --out "${folder_path}generated_sequences/results_kcat${i}.csv" \
    --write_linear

  conda deactivate
done

###############
# end message #
###############
cgroup_dir=$(awk -F: '{print $NF}' /proc/self/cgroup)
peak_mem=$(cat /sys/fs/cgroup${cgroup_dir}/memory.peak)
echo [$(date +"%Y-%m-%d %H:%M:%S")] peak memory is $peak_mem bytes
end_epoch=$(date +%s)
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname) after $((end_epoch-start_epoch)) seconds
