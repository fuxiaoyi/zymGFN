#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=HF_GRPO 
#SBATCH --time=24:00:00

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
#export https_proxy=http://proxy:80
    
###############
# run command #
###############

label="4.2.1.1"
model_directory="/home/woody/b114cb/b114cb23/models/ZymCTRL" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
max_iteration_num=30
export CUDA_LAUNCH_BLOCKING=1

echo RL for the enzyme class $label


for i in $(seq 15 $max_iteration_num);

do

    echo Starting iteration $i    
    if [ $i != 0 ]; then
    
      echo Train started
      python train.py --iteration_num $i --label $label --model_dir $model_directory --max_iteration_num $max_iteration_num
    
    fi

    echo Sequence generation started
    python seq_gen.py --iteration_num $i --label $label

   
    # Fold the sequences with ESM fold
    echo Folding started
    python ESM_Fold.py --iteration_num $i  --label $label
      
    
    # Calculate TM Score
    echo foldseek started for 2vvb
    export PATH=/home/woody/b114cb/b114cb23/foldseek/bin/:$PATH
    foldseek easy-search output_iteration$i/PDB  '2vvb.pdb' alpha_${label}_TM_iteration$((i)) $TMPDIR --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0 --gpu 1
      
    echo foldseek started for 1i6p
    foldseek easy-search output_iteration$i/PDB  '1i6p.pdb' beta_${label}_TM_iteration$((i)) $TMPDIR --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0 --gpu 1
    # Calculate aligment and clusters
    echo Aligments and cluster 
    export PATH=/home/woody/b114cb/b114cb23/mmseqs/bin/:$PATH
    mmseqs easy-cluster seq_gen_${label}_iteration$((i)).fasta clustering/clustResult_0.9_seq_gen_${label}_iteration$((i)) $TMPDIR --min-seq-id 0.9
    mmseqs easy-cluster seq_gen_${label}_iteration$((i)).fasta clustering/clustResult_0.5_seq_gen_${label}_iteration$((i)) $TMPDIR --min-seq-id 0.5
    mmseqs easy-search  seq_gen_${label}_iteration$((i)).fasta /home/woody/b114cb/b114cb23/Filippo/brenda_dataset/database_${label}.fasta alignment/alnResult_seq_gen_${label}_iteration$((i)).m8 $TMPDIR    

    echo dataset generation 
    python dataset_gen.py --iteration_num $i --label $label --model_dir $model_directory 
    
    python plot.py 
    
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
