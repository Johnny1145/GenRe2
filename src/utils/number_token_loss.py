import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from src.model.regress_lm.vocabs import DecoderVocab
from src.utils.number_token_selector import NumberTokenSelector


class NumberTokenLoss:
    def __init__(
        self, vocab: DecoderVocab[float], device, loss_function=F.mse_loss, weight=0.5
    ):
        self.loss_function = loss_function
        self.weight = weight

        self.selector = NumberTokenSelector(vocab, device)  # self.nvocab)
        self.nvocab = (
            self.selector.nvocab
        )  # torch.full((vocab_size,), float("nan"), device=device)

    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")
        #  masking
        #  - token0-9
        #  - token<E*>

        # vocab-level masks and values
        digit_mask = self.selector.digit_token_mask  # (V,)
        exp_mask = self.selector.exponent_token_mask  # (V,)
        digit_values = self.selector.digit_vocab  # (V,)
        exp_values = self.selector.exponent_vocab  # (V,)
        # print("digit_mask",digit_mask)
        # print("exp_mask",exp_mask)
        # print("digit_values",digit_values)
        # print("exp_values",exp_values)
        # assert False

        B, L, V = logits.shape

        # labelslabeltokentoken
        # pad
        is_exp_pos = exp_mask[labels]  # (B, L)
        is_digit_pos = (~is_exp_pos) & digit_mask[labels]  # (B, L)
        # print("is_exp_pos",is_exp_pos)
        # print("is_digit_pos",is_digit_pos)
        # assert False

        # masked logits
        very_negative = -1e9
        logits_digit = logits.clone()
        logits_digit[..., ~digit_mask] = very_negative
        logits_exp = logits.clone()
        logits_exp[..., ~exp_mask] = very_negative

        #  softmax
        probs_digit = F.softmax(logits_digit, dim=-1)
        probs_exp = F.softmax(logits_exp, dim=-1)

        #  NaN
        digit_values_safe = torch.where(digit_mask, digit_values, torch.zeros_like(digit_values))
        exp_values_safe = torch.where(exp_mask, exp_values, torch.zeros_like(exp_values))
        yhat_digit = torch.sum(probs_digit * digit_values_safe.view(1, 1, V), dim=-1)  # (B, L)
        yhat_exp = torch.sum(probs_exp * exp_values_safe.view(1, 1, V), dim=-1)  # (B, L)

        #
        yhat = torch.where(is_exp_pos, yhat_exp, yhat_digit)  # (B, L)
        # print("yhat",yhat[0,  :])
        # print("yhat_digit",yhat_digit[0,  :])
        # print("yhat_exp",yhat_exp[0,  :])
        # assert False

        #
        y_exp = exp_values_safe[labels]
        y_digit = digit_values_safe[labels]
        y = torch.where(is_exp_pos, y_exp, y_digit)
        # print("y",y[0,  :])
        # # print("y_digit",y_digit[0,:])
        # # print("y_exp",y_exp[0,:])
        # print("yhat",yhat[0,:])
        # # print("yhat_digit",yhat_digit[0,:])
        # # print("yhat_exp",yhat_exp[0,:])
        # assert False

        #  y  yhat
        valid = torch.isfinite(y)
        valid_hat = torch.isfinite(yhat)
        # print("valid",valid)
        # print("valid_hat",valid_hat)
        # assert False
        if valid.any():
            loss = self.loss_function(yhat[valid], y[valid])
        else:
            #  NaN
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        # print("loss",loss)
        # assert False
        return loss
