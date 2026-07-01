import torch
import torch.nn.functional as F
from torch import Tensor

class NumberTokenSelector:
    """
    Select number tokens (Copied from your prompt)
    """
    def __init__(self, vocab, device):
        self.tokenizer = vocab.tokenizer
        self.vocab = vocab
        self.nvocab = torch.full((len(vocab),), float("nan"), device=device)
        self.digit_vocab = torch.full((len(vocab),), float("nan"), device=device)
        self.exponent_vocab = torch.full((len(vocab),), float("nan"), device=device)

        hashed_num_tokens = set(self.tokenizer.get_num_tokens())
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
                try:
                    exp_value = float(self.tokenizer.token_to_exponent(token))
                    self.nvocab[id] = exp_value
                    self.exponent_vocab[id] = exp_value
                except Exception:
                    pass

        self.number_token_mask = ~torch.isnan(self.nvocab)
        self.digit_token_mask = ~torch.isnan(self.digit_vocab)
        self.exponent_token_mask = ~torch.isnan(self.exponent_vocab)

        # Get indices for efficient slicing
        self.digit_token_indices = torch.nonzero(self.digit_token_mask, as_tuple=True)[0]
        self.exponent_token_indices = torch.nonzero(self.exponent_token_mask, as_tuple=True)[0]

        # Get the corresponding values for the sliced tensors
        self.digit_token_values = self.digit_vocab[self.digit_token_indices]
        self.exponent_token_values = self.exponent_vocab[self.exponent_token_indices]

class DIST2Loss:
    """
    Implements the Discretized Distance Loss (DIST2Loss), revised version.
    This version uses efficient slicing and label remapping, inspired by the reference
    implementation, while maintaining the modularity of using NumberTokenSelector.
    """

    def __init__(
        self,
        vocab,
        device,
        temperature: float = 1.0,
        weight: float = 0.1,
    ):
        self.temperature = temperature
        self.weight = weight
        self.device = device
        self.selector = NumberTokenSelector(vocab, device)

        # --- Pre-compute label remapping tables for efficiency ---
        # This is a crucial optimization. Instead of searching for the index on every forward pass,
        # we create a lookup table at initialization.
        vocab_size = len(vocab)

        # For digits
        self.digit_global_to_local_map = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        self.digit_global_to_local_map[self.selector.digit_token_indices] = torch.arange(
            len(self.selector.digit_token_indices), device=device
        )

        # For exponents
        self.exponent_global_to_local_map = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        self.exponent_global_to_local_map[self.selector.exponent_token_indices] = torch.arange(
            len(self.selector.exponent_token_indices), device=device
        )


    def _calculate_kl_loss_for_space(
        self,
        logits: Tensor,
        labels: Tensor,
        positions_mask: Tensor,
        token_space_indices: Tensor,
        token_space_values: Tensor,
        label_remap_table: Tensor,
    ) -> Tensor:
        """
        Revised helper function to calculate KL loss using efficient slicing.
        """
        if not positions_mask.any():
            return torch.tensor(0.0, device=self.device, dtype=logits.dtype)

        # Filter logits and labels for the relevant positions (e.g., all digit positions in the batch)
        logits_pos_filtered = logits[positions_mask]  # (num_valid_tokens, V)
        labels_pos_filtered = labels[positions_mask]  # (num_valid_tokens,)

        # --- Step 1: Slice logits to the relevant subspace (e.g., only digit logits) ---
        # This is the key efficiency improvement.
        # Shape: (num_valid_tokens, num_space_tokens) e.g., (..., 10) for digits 0-9
        logits_sliced = logits_pos_filtered[:, token_space_indices]

        # --- Step 2: Remap ground truth labels to the local indices of the subspace ---
        # E.g., global token ID 3 (for "1") becomes local ID 1 in a [0-9] space.
        labels_local = label_remap_table[labels_pos_filtered]

        # --- Step 3: Build the distance-aware soft target distribution (p_d) ---
        # All calculations are now in the small, local space.

        # Get numerical values of the target tokens in the local space
        # Shape: (num_valid_tokens, 1)
        target_values = token_space_values[labels_local].unsqueeze(-1)

        # Get numerical values of all tokens in the local space
        # Shape: (1, num_space_tokens)
        all_token_values = token_space_values.unsqueeze(0)

        # Calculate squared Euclidean distance: d(v, y) = (v - y)^2
        # Broadcasting: (num_valid_tokens, 1) - (1, num_space_tokens) -> (num_valid_tokens, num_space_tokens)
        distances_sq = (all_token_values - target_values) ** 2

        # Create the soft target distribution. No masking needed before softmax
        # because we are already in the correct subspace.
        # User-requested: exponentiate the negated distance before normalizing.
        exp_weights = torch.exp(-distances_sq / self.temperature)
        soft_targets = exp_weights / exp_weights.sum(dim=-1, keepdim=True)

        # --- Step 4: Calculate the model's predicted distribution (p_theta) ---
        # Use log_softmax for numerical stability with KL divergence.
        log_probs_model = F.log_softmax(logits_sliced, dim=-1)

        # --- Step 5: Compute the KL Divergence Loss: KL(p_d || p_theta) ---
        kl_div = F.kl_div(
            log_probs_model,
            soft_targets,
            reduction='batchmean',
            log_target=False
        )

        return kl_div

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.numel() == 0:
            raise ValueError("Logits passed to the DIST2Loss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the DIST2Loss are empty!")

        # 1. Identify which positions are digits and which are exponents
        # These are (B, L) boolean masks
        is_exp_pos = self.selector.exponent_token_mask[labels]
        is_digit_pos = (~is_exp_pos) & self.selector.digit_token_mask[labels]

        # 2. Calculate loss for digit positions
        loss_digit = self._calculate_kl_loss_for_space(
            logits=logits,
            labels=labels,
            positions_mask=is_digit_pos,
            token_space_indices=self.selector.digit_token_indices,
            token_space_values=self.selector.digit_token_values,
            label_remap_table=self.digit_global_to_local_map,
        )

        # 3. Calculate loss for exponent positions
        # loss_exp = self._calculate_kl_loss_for_space(
        #     logits=logits,
        #     labels=labels,
        #     positions_mask=is_exp_pos,
        #     token_space_indices=self.selector.exponent_token_indices,
        #     token_space_values=self.selector.exponent_token_values,
        #     label_remap_table=self.exponent_global_to_local_map,
        # )

        # print(loss_digit)
        # print(loss_exp)
        # assert 0

        # 4. Combine the losses
        total_loss = loss_digit

        return total_loss
