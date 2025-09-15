#!/bin/bash -l 

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=HF_wDPO
#SBATCH --time=10:00:00
#SBATCH --export=ALL 
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80

set -e
set -u
set -o pipefail


###################
# set environment #
###################

module load python
module load cuda/12.6.1
source /home/woody/b114cb/b114cb23/.test_env/bin/activate

#export http_proxy=http://proxy:80
# export specific path 
#export https_proxy=http://proxy:80
    
###############
# run command #
###############

label="4.2.1.1"
model_directory="/home/woody/b114cb/b114cb23/models/ZymCTRL" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
DPO_mode="weighted" # choose between paired, ranked and weighted 
MAX_ITERATION_NUM=30
export CUDA_LAUNCH_BLOCKING=1


echo RL for the enzyme class $label


for i in $(seq 0 $MAX_ITERATION_NUM);

do

    echo Starting iteration $i    
    if [ $i != 0 ]; then
    
      echo Train started
      CUDA_LAUNCH_BLOCKING=1 python train.py --iteration_num $i --label $label --mode $DPO_mode --model_dir $model_directory --max_iteration_num  $MAX_ITERATION_NUM
    
    fi

    echo Sequence generation started
    CUDA_LAUNCH_BLOCKING=1 python seq_gen.py --iteration_num $i --label $label

   
    # Fold the sequences with ESM fold
    echo Folding started
    python ESM_Fold.py --iteration_num $i  --label $label
    

    # Calculate TM Score
    echo foldseek started for 2vvb
    export PATH=/home/woody/b114cb/b114cb23/foldseek/bin/:$PATH
    foldseek easy-search output_iteration$i/PDB  '2vvb.pdb' alpha_${label}_TM_iteration$((i)) tm --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0 --gpu 1
      
    echo foldseek started for 1i6p
    foldseek easy-search output_iteration$i/PDB  '1i6p.pdb' beta_${label}_TM_iteration$((i)) tm --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0 --gpu 1
    # Calculate aligment and clusters
    echo Aligments and cluster 
    export PATH=/home/woody/b114cb/b114cb23/mmseqs/bin/:$PATH
    mmseqs easy-cluster seq_gen_${label}_iteration$((i)).fasta clustering/clustResult_0.9_seq_gen_${label}_iteration$((i)) tmp --min-seq-id 0.9
    mmseqs easy-cluster seq_gen_${label}_iteration$((i)).fasta clustering/clustResult_0.5_seq_gen_${label}_iteration$((i)) tmp --min-seq-id 0.5
    mmseqs easy-search  seq_gen_${label}_iteration$((i)).fasta /home/woody/b114cb/b114cb23/Filippo/brenda_dataset/database_${label}.fasta alignment/alnResult_seq_gen_${label}_iteration$((i)).m8 tmp    
   
    echo dataset generation 
    python dataset_gen.py --iteration_num $i --label $label --mode $DPO_mode --model_dir $model_directory
    
    python plot.py 
    
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
