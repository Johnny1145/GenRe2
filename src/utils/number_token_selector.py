import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from src.model.regress_lm.tokenizers import DecoderTokenizer, P10Tokenizer
from src.model.regress_lm.vocabs import DecoderVocab


class NumberTokenSelector:
    """
    Select number tokens
    """

    def __init__(self, vocab: DecoderVocab[float], device):  # nvocab):
        self.tokenizer = vocab.tokenizer
        self.vocab = vocab
        #
        self.nvocab = torch.full((len(vocab),), float("nan"), device=device)
        #
        self.digit_vocab = torch.full((len(vocab),), float("nan"), device=device)
        self.exponent_vocab = torch.full((len(vocab),), float("nan"), device=device)

        hashed_num_tokens = set(self.tokenizer.get_num_tokens())
        # If tokenizer exposes exponent tokens (e.g., P10 last position), include them
        exponent_tokens = []
        if hasattr(self.tokenizer, "get_exponent_tokens"):
            try:
                exponent_tokens = list(self.tokenizer.get_exponent_tokens())
            except Exception:
                exponent_tokens = []
        hashed_exp_tokens = set(exponent_tokens)

        for token, id in self.vocab.stoi.items():
            if token in hashed_num_tokens:
                value = float(self.tokenizer.token_to_number(token))
                self.nvocab[id] = value
                self.digit_vocab[id] = value
            elif token in hashed_exp_tokens and hasattr(self.tokenizer, "token_to_exponent"):
                # Map exponent tokens to their integer exponent value
                try:
                    exp_value = float(self.tokenizer.token_to_exponent(token))
                    self.nvocab[id] = exp_value
                    self.exponent_vocab[id] = exp_value
                except Exception:
                    pass

        # Extract indices and values of number tokens
        self.number_token_mask = ~torch.isnan(self.nvocab)
        self.digit_token_mask = ~torch.isnan(self.digit_vocab)
        self.exponent_token_mask = ~torch.isnan(self.exponent_vocab)
        self.number_token_indices = torch.nonzero(
            self.number_token_mask, as_tuple=False
        ).squeeze()

        self.number_token_values = self.nvocab[self.number_token_indices]
        self.digit_token_values = self.digit_vocab[self.digit_token_mask]
        self.exponent_token_values = self.exponent_vocab[self.exponent_token_mask]

    def select_number_tokens(self, logits: Tensor):
        #
        logits = logits[:, :, self.number_token_mask]
        return logits, self.number_token_mask


def main():
    tokenizer = P10Tokenizer(num_digits=4, exponent_range=10)
    vocab = DecoderVocab(tokenizer)
    device = torch.device("cuda")
    number_token_selector = NumberTokenSelector(vocab, device)
    print(number_token_selector.number_token_indices)
    print(number_token_selector.number_token_values)


if __name__ == "__main__":
    main()
