import re

import torch
from torch import nn
import torch.nn.functional as F


IGNORE_INDEX = -100

def is_number_regex(input_str):
    return bool(re.match(r"^[0-9]+$", input_str))


def l2_dist(x, y):
    B, L = x.shape
    # shape: x (BL) y (V)
    dist = (x.reshape(-1)[:, None] - y[None, :]) ** 2
    dist = dist.reshape(B, L, -1)
    return dist


def get_digit_loss(
    loss_sft,
    tokenizer,
    logits,
    labels,
    attention_mask=None,
    target_temperature=2.0,
    beta: float = 1.0,
):
    # shift
    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    digits = [(k, v) for k, v in tokenizer.vocab.items() if is_number_regex(k)]
    digit_values = (
        torch.tensor([int(k) for k, v in digits])
        .to(shift_labels.device)
        .to(logits.dtype)
    )
    digit_indices = torch.tensor([v for k, v in digits]).to(shift_labels.device)

    mask = shift_labels[..., None] == digit_indices[None, None, :]
    any_mask = mask.any(-1).flip(-1)

    x = any_mask.long()
    # Compute the cumulative sum
    cumsum = torch.cumsum(x, dim=-1)

    # Create a mask to identify positions where cumulative sum should reset
    reset_mask = x == 0

    # Compute the differences in the cumulative sum at reset points
    reset_diff = torch.zeros_like(cumsum)
    reset_diff[:, 1:] = cumsum[:, :-1] * reset_mask[:, 1:].to(dtype=cumsum.dtype)
    # Compute the cumulative sum again after reset
    adjusted_cumsum = cumsum - reset_diff.cummax(-1).values
    pos = adjusted_cumsum.flip(-1)

    data_targets = mask.long().argmax(-1)
    data_targets = digit_values[data_targets.reshape(-1)].reshape(*data_targets.shape)
    digit_targets = l2_dist(data_targets, digit_values)
    digit_targets = (-digit_targets / target_temperature).softmax(-1)

    digit_logits = shift_logits[..., digit_indices]
    loss_fn = nn.KLDivLoss(reduction="none")
    loss = loss_fn(digit_logits.log_softmax(-1), digit_targets)  # BLV
    loss = loss.sum(-1)  # definition of KL

    loss_coeff = pos.to(loss.dtype)
    loss = (loss * loss_coeff).sum() / loss_coeff.sum()

    loss = loss_sft + beta * loss

    stats = {
        "loss/total": loss.detach(),
        "loss/sft": loss_sft.detach(),
        "loss/digit": loss.detach(),
    }
    return loss, stats