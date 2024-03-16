import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import os
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Load our tensors
data_path = "data/all_games_tensor.pt"
all_games_tensor = torch.load(data_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

p = 113
frac_train = 0.8

# Optimizer config
lr = 1e-3
wd = 1. 
betas = (0.9, 0.98)

num_epochs = 25000
checkpoint_every = 100
DATA_SEED = 598

# Split all_games into train_data and eval_data after shuffling
torch.manual_seed(DATA_SEED)
torch.random.manual_seed(DATA_SEED)
np.random.seed(DATA_SEED)

# Load all games tensor
data_path = "data/all_games_tensor.pt"
all_games_tensor = torch.load(data_path).long()

print(all_games_tensor.shape)


# Shuffle the data
shuffled_indices = torch.randperm(all_games_tensor.shape[0])
all_games_tensor_shuffled = all_games_tensor[shuffled_indices, :]

# Split the data
train_data = all_games_tensor_shuffled[:int(frac_train * len(all_games_tensor_shuffled))].to(device)
eval_data = all_games_tensor_shuffled[int(frac_train * len(all_games_tensor_shuffled)):].to(device)
print(f"Train data shape: {train_data.shape}")
print(f"Eval data shape: {eval_data.shape}")

# Save to data
torch.save(train_data, "data/train_data.pt")
torch.save(eval_data, "data/eval_data.pt")

def lm_accuracy(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute mean accuracy for the language model predictions.
    
    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
    
    Returns:
        torch.Tensor: Mean accuracy across all predictions.
    """
    # Convert logits to predictions: select the max logit index (predicted token) for each position
    predictions = torch.argmax(logits[..., :-1, :], dim=-1)
    
    # Shift tokens to align with prediction targets
    # We ignore the first token since we cannot predict it, aligning predictions and actual next tokens
    actual_next_tokens = tokens[..., 1:]
    
    # Compute matches by comparing predictions with actual next tokens
    matches = predictions == actual_next_tokens
    
    # Calculate mean accuracy by averaging the matches tensor
    mean_accuracy = matches.float().mean()
    
    return mean_accuracy

def is_legal_move(logits, src):
    pred_tokens = logits.argmax(-1)
    batch_size, seq_len = src.size()
    legal_moves = torch.zeros_like(src, dtype=torch.bool)
    for i in range(seq_len):
        legal_moves[:, i] = (src[:, :i] != pred_tokens[:, i].unsqueeze(1)).all(dim=1)
    legal_moves = legal_moves.float()

    # Create a mask of 1s where predicted token == actual token == 9
    nines_mask = ((src == 9) & (pred_tokens == 9)).float()
    legal_moves += nines_mask
    legal_moves = torch.clamp(legal_moves, max=1)


    total_legal_moves = legal_moves.sum().item()
    total_predictions = (seq_len) * batch_size
    legal_move_percentage = (total_legal_moves / total_predictions) * 100
    return legal_move_percentage

# Create a DataLoader for the training data
batch_size = 16384
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


cfg = HookedTransformerConfig(
    attn_only=True,
    n_layers=2,
    n_heads=2,
    d_model=32,
    d_head=8,
    d_vocab=10,
    act_fn='relu',
    n_ctx=9,
    init_weights=True,
    device=device,
    seed=999,
    original_architecture="gpt2",
    attention_dir='causal',
)

model = HookedTransformer(cfg)
# Print number of parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

train_losses, eval_losses = [], []
epochs = 20
# Before we start, print eval loss and accuracy
model.eval()
eval_loss = 0
eval_logits, eval_loss = model(eval_data, return_type='both')
# Print devices
print(f"eval data device: {eval_data.device}, eval logits device: {eval_logits.device}")
eval_accuracy = lm_accuracy(eval_logits, eval_data)
eval_percent_legal = is_legal_move(eval_logits, eval_data)
print(f"Initial eval loss: {eval_loss.item():.4f}, eval accuracy: {eval_accuracy*100:.2f}%, eval legal moves: {eval_percent_legal:.2f}%")
for epoch in range(epochs):
    batch_loss = 0
    model.train()
    total_batches = 0
    for batch in train_loader:
        total_batches += 1
        optimizer.zero_grad()
        logits, loss = model(batch.long(), return_type='both')
        #loss = loss_fn(logits, batch)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
    train_losses.append(batch_loss / total_batches)

    # Eval loss
    model.eval()
    eval_loss = 0
    eval_logits, eval_loss = model(eval_data.long(), return_type='both')
    #eval_loss = loss_fn(eval_logits, eval_data)
    eval_losses.append(eval_loss.item())

    # Eval accuracy
    eval_accuracy = lm_accuracy(eval_logits, eval_data)

    # Eval legal moves
    eval_percent_legal = is_legal_move(eval_logits, eval_data)

    print(f"Epoch {epoch} | Train loss: {train_losses[-1]:.4f}, Eval loss: {eval_losses[-1]:.4f}, Eval accuracy: {eval_accuracy*100:.2f}%, Eval legal moves: {eval_percent_legal:.2f}%")


# Save model
PTH_LOCATION = "data/transformer_lens.pth"
torch.save(
    {
        "model":model.state_dict(),
        "config": model.cfg,
        "eval_losses": eval_losses,
        "train_losses": train_losses,
    },
    PTH_LOCATION)