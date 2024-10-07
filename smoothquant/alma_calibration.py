import torch
import torch.nn as nn

from datasets import load_dataset
from datasets import Dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm
import pandas as pd
from load_data import load_WMT22Testdataset

def get_act_scales(model, tokenizer, mode='mode_1' ,num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )
    dataset_ = load_WMT22Testdataset(mode=mode, upper_bound_num_samples=num_samples)
    longest_sen_len = len(max(dataset_, key=len))
    max_length = max(seq_len, longest_sen_len)
    
    for i in tqdm(range(len(dataset_))):
        #prormpts are inside the dataset_
        input_ids = tokenizer(
            dataset_[i], return_tensors="pt", max_length=max_length, truncation=True
        ).input_ids.to(device)
        model(input_ids)


    for h in hooks:
        h.remove()

    return act_scales

