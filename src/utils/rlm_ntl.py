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


    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        #  device
        device = logits.device
        labels = labels.to(device)
        logits, number_tokens = self.selector.select_number_tokens(logits)

        # Compute the weighted average of number tokens (yhat)
        softmaxed = F.softmax(logits, dim=-1)
        nvocab_on_device = self.nvocab.to(device)
        yhat = torch.sum(softmaxed * nvocab_on_device[number_tokens], dim=-1)
        y = nvocab_on_device[labels]

        loss = self.loss_function(yhat[~torch.isnan(y)], y[~torch.isnan(y)])
        return loss