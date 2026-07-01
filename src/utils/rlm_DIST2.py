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

class DIST2Loss:
    """
    Implements the Discretized Distance Loss (DIST2Loss), revised version.
    This version uses efficient slicing and label remapping, inspired by the reference
    implementation, while maintaining the modularity of using NumberTokenSelector.
    """

    def __init__(self, tokenizer, device, loss_function=F.mse_loss, weight=0.3):
        self.weight = weight
        self.device = device
        self.selector = NumberTokenSelector_RLM(tokenizer, device)
        self.temperature = 1.0

        # --- Pre-compute label remapping tables for efficiency ---
        # This is a crucial optimization. Instead of searching for the index on every forward pass,
        # we create a lookup table at initialization.
        vocab_size = 13

        # For digits
        self.digit_global_to_local_map = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        self.digit_global_to_local_map[self.selector.number_token_indices] = torch.arange(
            len(self.selector.number_token_indices), device=device
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
        unnormalized_log_probs = -distances_sq / self.temperature
        soft_targets = F.softmax(unnormalized_log_probs, dim=-1)

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
        device = logits.device
        labels = labels.to(device)
        is_digit_pos = self.selector.number_token_mask.to(device)[labels]

        # 4.
        digit_token_indices_dev = self.selector.number_token_indices.to(device)
        digit_token_values_dev = self.selector.number_token_values.to(device)
        digit_remap_table_dev = self.digit_global_to_local_map.to(device)

        # <---  --->

        # 5.
        loss_digit = self._calculate_kl_loss_for_space(
            logits=logits,
            labels=labels,
            positions_mask=is_digit_pos,
            token_space_indices=digit_token_indices_dev, # <---
            token_space_values=digit_token_values_dev,   # <---
            label_remap_table=digit_remap_table_dev,     # <---
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