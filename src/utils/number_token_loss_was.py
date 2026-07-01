import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from src.model.regress_lm.vocabs import DecoderVocab
from src.utils.number_token_selector import NumberTokenSelector


class NumberTokenLoss_WAS:
    def __init__(
        self, vocab: DecoderVocab[float], device, loss_function=F.mse_loss, weight=0.5
    ):
        self.loss_function = loss_function
        self.weight = weight

        self.selector = NumberTokenSelector(vocab, device)  # self.nvocab)
        self.nvocab = (
            self.selector.nvocab
        )  # torch.full((vocab_size,), float("nan"), device=device)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
         NTL-WAS
        Args:
            logits: logits (B, L, V)
            labels: token (B, L)
        Returns:
            token
        """
        B, L, V = logits.shape

        # 1. token
        # is_number_pos  (B, L)
        is_number_pos = self.selector.number_token_mask[labels]

        # token0
        if not is_number_pos.any():
            return torch.tensor(0.0, device=self.device, dtype=logits.dtype)

        # 2. token
        # a. logitslabels
        # relevant_logits: (N, V), Ntoken
        relevant_logits = logits[is_number_pos]
        # relevant_labels: (N,)
        relevant_labels = labels[is_number_pos]

        # b.  y
        # y_true_values: (N,)
        y_true_values = self.selector.nvocab[relevant_labels]

        # c. logitssoftmax y_hat
        # probs: (N, V)
        probs = F.softmax(relevant_logits, dim=-1)

        # d. token y_hat_j
        # number_probs: (N, num_number_tokens)
        number_probs = probs[:, self.selector.number_token_indices]

        # e. token V_j
        # all_number_values: (num_number_tokens,)
        all_number_values = self.selector.number_token_values

        # 5. (4): sum_j (y_hat_j * |y - V_j|)
        #
        # y_true_values  (N, 1)
        # all_number_values  (1, num_number_tokens)
        # abs_diff  (N, num_number_tokens)
        abs_diff = torch.abs(y_true_values.unsqueeze(-1) - all_number_values.unsqueeze(0))

        #
        # per_token_loss: (N,)
        per_token_loss = torch.sum(number_probs * abs_diff, dim=-1)

        # 6. token
        loss = per_token_loss.mean()

        return loss