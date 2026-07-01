from typing import Optional

import torch
import torch.nn.functional as F
from torch._tensor import Tensor


class NumberTokenSelector_RLM:
    '''
    Select number tokens
    '''
    def __init__(self, tokenizer, device): # nvocab):
        self.tokenizer = tokenizer
        vocab_size = 13
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        for id in range(3,13):
            self.nvocab[id] = id-3

        # Extract indices and values of number tokens
        self.number_token_mask = ~torch.isnan(self.nvocab)
        self.number_token_indices = torch.nonzero(self.number_token_mask, as_tuple=False).squeeze()

        self.number_token_values = self.nvocab[self.number_token_indices]

    def select_number_tokens(self, logits: Tensor):
        # Create a mask to filter out non-digit tokens and labels

        number_token_mask = self.number_token_mask.to(logits.device)
        logits = logits[:, :, number_token_mask]
        return logits, number_token_mask

class NumberTokenLoss:
    def __init__(self, tokenizer, device, loss_function=F.mse_loss, weight=0.3):
        self.loss_function = loss_function
        self.weight = weight

        self.selector = NumberTokenSelector_RLM(tokenizer, device) # self.nvocab)
        self.nvocab = self.selector.nvocab # torch.full((vocab_size,), float("nan"), device=device)


    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        position_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
         NTL-WAS
        Args:
            logits: logits (B, L, V)
            labels: token (B, L)
            position_weights:  CE reweight  (L,)  (B, L)
                 token
        Returns:
            token
        """
        B, L, V = logits.shape

        # 1. token
        # is_number_pos  (B, L)
        device = logits.device

        # 2.  labels
        labels = labels.to(device)

        # 3.  __init__
        number_token_mask_dev = self.selector.number_token_mask.to(device)
        nvocab_dev = self.selector.nvocab.to(device)
        number_token_indices_dev = self.selector.number_token_indices.to(device)
        number_token_values_dev = self.selector.number_token_values.to(device)
         # 1. token
        # is_number_pos  (B, L)
        # <--- :  mask
        is_number_pos = number_token_mask_dev[labels]

        # token0
        if not is_number_pos.any():
            # <--- :
            return torch.tensor(0.0, device=device, dtype=logits.dtype)

        # 2. token
        # a. logitslabels
        # relevant_logits: (N, V), Ntoken
        relevant_logits = logits[is_number_pos]
        # relevant_labels: (N,)
        relevant_labels = labels[is_number_pos]

        # b.  y
        # y_true_values: (N,)
        # <--- :  nvocab
        y_true_values = nvocab_dev[relevant_labels]

        # c. logitssoftmax y_hat
        # probs: (N, V)
        probs = F.softmax(relevant_logits, dim=-1)

        # d. token y_hat_j
        # number_probs: (N, num_number_tokens)
        # <--- :  indices
        number_probs = probs[:, number_token_indices_dev]

        # e. token V_j
        # all_number_values: (num_number_tokens,)
        # <--- :  values
        all_number_values = number_token_values_dev

        # 5. (4): sum_j (y_hat_j * |y - V_j|)
        #
        # y_true_values  (N, 1)
        # all_number_values  (1, num_number_tokens)
        # abs_diff  (N, num_number_tokens)
        # <--- :
        abs_diff = torch.abs(y_true_values.unsqueeze(-1) - all_number_values.unsqueeze(0))

        #
        # per_token_loss: (N,)
        per_token_loss = torch.sum(number_probs * abs_diff, dim=-1)

        # 6.  token  position_weights  CE
        if position_weights is None:
            loss = per_token_loss.mean()
        else:
            pw = position_weights.to(device=device, dtype=logits.dtype)
            if pw.dim() == 1:
                pw = pw.view(1, L).expand(B, L)
            elif pw.shape != (B, L):
                raise ValueError(
                    f"position_weights must be (L,) or (B, L), got {tuple(pw.shape)}"
                )
            w_sel = pw[is_number_pos]
            loss = (per_token_loss * w_sel).sum() / w_sel.sum().clamp(min=1e-8)

        return loss