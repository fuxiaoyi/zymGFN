
from src.utils import *
from src.pLM_weigtedDPO import weighted_DPO
from src.pLM_GRPO import pLM_GRPOTrainer

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel)
from trl.trainer.utils import pad
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import argparse
import torch
import numpy as np
import random
import pandas as pd
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)


args = parser.parse_args()


CONFIG = {
    "beta": 0.1,
    "seed": 42,
    "learning_rate": 2e-6,
    # "batch_size": 15,
    "batch_size": 2,
    "num_epochs": 1,
    "split_percent": 0.2,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "adam_decay": 0.1,
}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    set_seed(seed)



def reward_len(completions, **kwargs):
    return 0
    

def format_sequence(sequence, label):
    return f"<sep><start>{sequence}<end><|endoftext|>"

def generate_dataset(iteration_num, label):

    df = pd.read_csv(f"logs.csv")
    df = df[df["iteration_num"] == (iteration_num - 1) ]
    
    rows = []
    for idx, entry in df.iterrows():
        TM_norm_que = float(entry["TM_norm_que"])
        sequence = entry["sequence"]
        algn = float(entry["algn"])
        lenght_rew = math.exp(-((((algn/len(sequence))-1)**2)/(0.5**2)))

        rows.append({
            "prompt": label,
            "completion": format_sequence(sequence, label),
            "reward": float(TM_norm_que + lenght_rew)
        })
    
    return Dataset.from_list(rows)


seed_everything(CONFIG["seed"])

# create dataset
root_dir = os.path.dirname(os.path.abspath(__file__))
seq_dir = os.path.join(root_dir, "data", "inputs")
fasta_file = os.path.join(seq_dir, f"seq_gen_{args.label}_iteration{args.iteration_num-1}.fasta")
dataset = generate_dataset(args.iteration_num, args.label)
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)

train_dataset = split['train']
eval_dataset   = split['test'] 
#print(train_dataset)

tokenizer_dir = args.model_dir
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                          add_eos_token=False, # NEED this for training NOT for generate() else add eos at the end of promt
                                          add_bos_token=False,
                                          use_fast=True)
tokenizer.eos_token_id = 1
tokenizer.pad_token_id = 0



if args.iteration_num > 1 : 
    model = f"output_iteration{args.iteration_num-1}"
    checkpoint = checkpoint_load(f"output_iteration{args.iteration_num-1}")

else:
    model = args.model_dir
    checkpoint = None
 
lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)
optimizer, model, scheduler  = load_optimizer_scheduler(model, checkpoint, lr_list[args.iteration_num-1].item(), CONFIG)


training_args = GRPOConfig(output_dir=f"output_iteration{args.iteration_num}", 
                           logging_steps=100,
                           beta=CONFIG["beta"],
                           num_train_epochs = CONFIG["num_epochs"],
                           learning_rate = lr_list[args.iteration_num-1].item(),
                           do_train = True, 
                           do_eval = True, 
                           eval_strategy = "epoch",
                           save_strategy = "steps",                      # ← add this line
                           eval_steps = 500, 
                           save_total_limit = 1,
                           save_steps = 5,
                           num_generations = 4,
                           bf16=True,
                           gradient_checkpointing=True,
                           torch_compile=False)

print("model ",model)
trainer = pLM_GRPOTrainer(
    model= model,
    ref_model = args.model_dir,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
    optimizers = (optimizer, scheduler))

trainer.lr_scheduler       = scheduler
trainer.lr_scheduler_state = None

print("fixed LR (optimizer) before training:", trainer.optimizer.param_groups[0]["lr"])

trainer.train()

trainer.save_model()

print("fixed LR (optimizer) after traning:", trainer.optimizer.param_groups[0]["lr"])

torch.cuda.empty_cache()
