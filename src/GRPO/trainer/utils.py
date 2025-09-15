from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel)
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import torch
import numpy as np
import random
import pandas as pd
import math
import os
from pathlib import Path


def checkpoint_load(model_path):
   
    base_path = Path(model_path)
    for p in base_path.rglob("checkpoint-*"):
            if p.is_dir():
                checkpoint = p.resolve()
    
    return checkpoint
    
def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if not st:
                continue
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)


'''
def load_optimizer_scheduler(model, checkpoint, lr, CONFIG):
    
    #model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(model)

    optimizer = AdamW(
        model.parameters(),
        lr = lr,
        betas = CONFIG["adam_betas"],
        eps = CONFIG["epsilon"],
        weight_decay = CONFIG["adam_decay"]
    )

    if checkpoint is not None:
        optim_state_path = checkpoint / "optimizer.pt"
        saved_optim_state = torch.load(optim_state_path, map_location="cpu")
        optimizer.load_state_dict(saved_optim_state)
        for group in optimizer.param_groups:
                group["lr"] = lr
                group["initial_lr"] = lr

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    return optimizer, model, scheduler
'''

def load_optimizer_scheduler(model, checkpoint, lr, CONFIG, *, use_foreach: bool = True):
    # ????????
    if isinstance(model, (str, Path)):
        model_obj = AutoModelForCausalLM.from_pretrained(str(model))
    else:
        model_obj = model

    # ??????(??? state ????)
    try:
        param_device = next(model_obj.parameters()).device
    except StopIteration:
        param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = AdamW(
        (p for p in model_obj.parameters() if p.requires_grad),
        lr=lr,
        betas=CONFIG["adam_betas"],
        eps=CONFIG["epsilon"],
        weight_decay=CONFIG["adam_decay"],
        foreach=use_foreach,   # ?????,?? False ???
    )

    if checkpoint is not None:
        opt_path = Path(checkpoint) / "optimizer.pt"
        if opt_path.exists():
            saved = torch.load(opt_path, map_location=param_device)  # ? ??:???????
            optimizer.load_state_dict(saved)
            for g in optimizer.param_groups:  # ???????????
                g["lr"] = lr
                g["initial_lr"] = lr
            _optimizer_to_device(optimizer, param_device)            # ? ??:??????

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    return optimizer, model_obj, scheduler