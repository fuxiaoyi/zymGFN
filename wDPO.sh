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


###############
# run command #
###############
folder_path="/home/bingxing2/ailab/scxlab0094/hantao/project/internTA/proteinGflownet/ProtRL/example/ZymCTRL-fold/wDPO_backup/"
label="3.1.1.1"
smiles='CC(=O)Oc1ccc(cc1)[N+](=O)[O-]'
DPO_mode="weighted"


echo "self-training of ZymCTRL for TMscore with ${label} started"
conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_pt260
module unload compilers/cuda cudnn compilers/gcc
source /home/bingxing2/apps/package/pytorch/2.6.0-cu124-cp311/env.sh

for i in $(seq 1 30); do
  echo "========== Iteration ${i} =========="

  # -------- 训练（从第 1 轮开始，用上一轮 i-1 的 CSV） --------
  if [ "${i}" -ne 0 ]; then
    prev=$((i-1))
    cd "${folder_path}"
    echo "[Iter ${i}] Train started (use tox/kcat from iter ${prev})"

    python "${folder_path}train.py" \
      --iteration_num "${i}" \
      --label "${label}" \
      --model_dir "/home/bingxing2/ailab/group/ai4earth/hantao/project/internTA/proteinGflownet/ZymCTRL_local" \
      --max_iteration_num 30 \
      --mode $DPO_mode

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
  module unload compilers/cuda cudnn compilers/gcc
#   compilers/cuda/11.8   4) cudnn/8.8.1.3_cuda11.x   5) nccl/2.11.4-1_cuda11.8
  module load compilers/gcc/11.3.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x
  conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_cp310
  cd "${folder_path}"

  if [ "${i}" -ne 0 ]; then
    python "${folder_path}seq_gen.py" \
      --iteration_num "${i}" \
      --label "${label}" \
      --cwd "${folder_path}"
  
    # -------- 折叠结构（ESMFold）--------
    echo "[Iter ${i}] Folding (ESMFold)"
    python "${folder_path}ESM_Fold.py" \
      --iteration_num "${i}" \
      --label "${label}" 
  fi

  # -------- 计算 TM 分数（Foldseek）--------
  echo "[Iter ${i}] Foldseek TM-score vs 7atl (alpha)"
  tmp_dir="${folder_path}tmp_${i}"
  mkdir -p "${tmp_dir}"
  foldseek easy-search \
    "${folder_path}output_iteration$i/PDB" \
    "${folder_path}7atl.pdb" \
    "alpha_${label}_TM_iteration${i}" \
    "${tmp_dir}" \
    --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" \
    --exhaustive-search 1 -e inf --tmscore-threshold 0.0
  rm -rf "${tmp_dir}"

  echo Aligments and cluster 
  mkdir -p "${folder_path}clustering"
  tmp_dir="${folder_path}tmp_${i}"
  mkdir -p "${tmp_dir}"
  mmseqs easy-cluster "${folder_path}seq_gen_${label}_iteration${i}.fasta" "${folder_path}clustering/clustResult_0.9_seq_gen_${label}_iteration${i}" "${tmp_dir}" --min-seq-id 0.9
  rm -rf "${tmp_dir}"

  conda deactivate

  # -------- 毒性预测（ToxinPred2）--------
  echo "[Iter ${i}] ToxinPred2"
  conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/envs/matrics_yxu
  cd /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/proteinGflownet/toxinpred/toxinpred2

  python toxinpred2.py \
    -i "${folder_path}seq_gen_${label}_iteration${i}.fasta" \
    -o "${folder_path}outfile${i}.csv" \
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
    --fasta "${folder_path}seq_gen_${label}_iteration${i}.fasta" \
    --smiles "${smiles}" \
    --task kcat \
    --out "${folder_path}results_kcat${i}.csv" \
    --write_linear

  conda deactivate
  
  conda activate /home/bingxing2/ailab/scxlab0094/hantao/project/internTA/conda_env/proteinRL_pt260
  module unload compilers/cuda cudnn compilers/gcc
  source /home/bingxing2/apps/package/pytorch/2.6.0-cu124-cp311/env.sh
  cd "${folder_path}"

  echo dataset generation 
  cd ${folder_path}
  python ${folder_path}dataset_gen_toxUnikp.py --iteration_num $i --label $label --model_dir "/home/bingxing2/ailab/group/ai4earth/hantao/project/internTA/proteinGflownet/ZymCTRL_local" --tox_csv "${folder_path}outfile${i}.csv" --kcat_csv "${folder_path}results_kcat${i}.csv"

  
done

###############
# end message #
###############
end_epoch=$(date +%s)
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname) after $((end_epoch-start_epoch)) seconds
