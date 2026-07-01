from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

def debug_tensor(name: str, tensor: torch.Tensor):
    """"""

    grad_fn_name = tensor.grad_fn.__class__.__name__ if tensor.grad_fn is not None else "None"

    print(
        f"[DEBUG] Tensor '{name}':\n"
        f"  - Shape: {tensor.shape}\n"
        f"  - Device: {tensor.device}\n"
        f"  - Dtype: {tensor.dtype}\n"
        f"  - Requires Grad: {tensor.requires_grad}\n"
        f"  - Grad Fn: {grad_fn_name}"
    )

class Remax:
    """
    """

    def __init__(
        self,
        temperature: float = 1.0,
        num_samples: int = 8,
        reward_scale: float = 1.0,
        kernel_sigma: float = 1.0,
        kernel_reduction: str = "mean",
        kl_weight: float = 0.0,
        ratio_clip: float = 0.2,
        expert_ce_weight: float = 0.0,
        y_quantile_transformer: Optional[Any] = None,
    ):
        self.temperature = temperature
        self.num_samples = num_samples
        self.reward_scale = reward_scale
        self.kernel_sigma = kernel_sigma
        self.kernel_reduction = kernel_reduction
        self.kl_weight = kl_weight
        self.ratio_clip = ratio_clip
        self.expert_ce_weight = expert_ce_weight
        self.y_quantile_transformer = y_quantile_transformer

    def compute_reward(self, predictions: torch.Tensor, targets: torch.Tensor, nan_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """reward NaN """
        if predictions.dim() > targets.dim():
            targets_expanded = targets.unsqueeze(1).expand_as(predictions)
        else:
            targets_expanded = targets

        #  NaN  0
        predictions_for_dist = predictions.clone()
        if nan_mask is not None and nan_mask.any():
            predictions_for_dist[nan_mask] = targets_expanded[nan_mask]

        dist2 = (predictions_for_dist - targets_expanded) ** 2
        rewards = torch.clamp(dist2, max=50)
        rewards = -rewards

        #  NaN
        if nan_mask is not None and nan_mask.any():
            #  -10
            #  NaN
            rewards[nan_mask] = -100.0

        return rewards
    # def compute_reward(self, predictions: torch.Tensor, targets: torch.Tensor, nan_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     """reward NaN """
    #     if predictions.dim() > targets.dim():
    #         targets_expanded = targets.unsqueeze(1).expand_as(predictions)
    #     else:
    #         targets_expanded = targets

    #     #  NaN  0
    #     predictions_for_dist = predictions.clone()
    #     if nan_mask is not None and nan_mask.any():
    #         predictions_for_dist[nan_mask] = targets_expanded[nan_mask]

    #     dist2 = torch.abs(predictions_for_dist - targets_expanded)
    #     rewards = torch.clamp(dist2, max=10)
    #     rewards = -rewards

    #     #  NaN
    #     if nan_mask is not None and nan_mask.any():
    #         #  -10
    #         #  NaN
    #         rewards[nan_mask] = -20.0

    #     return rewards

    def compute_kl_divergence(self, current_logits: torch.Tensor, ref_logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ KL(current || ref)"""
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        # KL(p||q) = sum(p * (log p - log q))
        kl_div_per_token = (current_log_probs.exp() * (current_log_probs - ref_log_probs)).sum(dim=-1)

        if mask is not None:
            kl_div_per_token = kl_div_per_token * mask.float()
            # Average over non-masked tokens
            kl_loss = kl_div_per_token.sum() / mask.float().sum().clamp_min(1.0)
        else:
            kl_loss = kl_div_per_token.mean()

        return kl_loss

    def _sequence_log_prob_from_logits(self, logits: torch.Tensor, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ logits  token ID """
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
        # Sum log probs over the sequence length for each sample
        seq_log_prob = (gathered * mask).sum(dim=1)
        return seq_log_prob

    def __call__(
        self, model, tokenizer, batch: Dict[str, torch.Tensor], ref_model
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        model.train() # Ensure main model is in train mode for gradients
        if ref_model is not None:
            ref_model.eval() # Reference model is always in eval mode

        # --- FIX 4: Handle missing normalization stats ---
        if "y_mean" not in batch or "y_std" not in batch:
            raise ValueError("Batch must contain 'y_mean' and 'y_std' for reward normalization.")

        device = model.device

        batch_on_device = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        #
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                batch_on_device[key] = value
        # ------------------------------------------------------------------

        targets = batch_on_device["y"].to(device)
        y_means = batch_on_device["y_mean"].to(device)
        y_stds = batch_on_device["y_std"].to(device)

        with torch.no_grad():
            #  greedy_decode
            greedy_preds_np = model.greedy_decode(batch_on_device)
            # print(greedy_preds_np)
            # print(y_means.cpu().numpy())
            # assert 0

            #  NaN
            greedy_preds_np[np.isnan(greedy_preds_np)] = y_means.cpu().numpy()

            greedy_preds = torch.tensor(greedy_preds_np, device=device, dtype=torch.float64)

            #
            if self.y_quantile_transformer is not None:
                #
                greedy_preds_np_norm = self.y_quantile_transformer.transform(greedy_preds_np.reshape(-1, 1)).flatten()
                targets_np_norm = self.y_quantile_transformer.transform(targets.cpu().numpy().reshape(-1, 1)).flatten()
                greedy_preds_norm = torch.tensor(greedy_preds_np_norm, device=device, dtype=torch.float64)
                targets_norm = torch.tensor(targets_np_norm, device=device, dtype=torch.float64)
                # print(greedy_preds_norm)
                # print(targets_norm)
                # assert 0
            else:
                #
                greedy_preds_norm = (greedy_preds - y_means) / y_stds
                targets_norm = (targets - y_means) / y_stds

            # print(greedy_preds_norm.shape, targets_norm.shape)
            # assert 0

            #  (B,)
            baseline_rewards = self.compute_reward(greedy_preds_norm, targets_norm)

        # ---  2:  ---
        #
        sampled_ids, sampled_floats, sampled_log_probs, sampled_logits, policy_entropy = model.sample_with_logprobs(
            batch_on_device, self.num_samples, self.temperature, return_logits=True, return_entropy=True
        )
        # sampled_ids: (B, S, L), sampled_log_probs: (B, S, L), sampled_logits: (B, S, L, V)
        # sampled_floats: (B, S, 1)
        # policy_entropy:

        # ---  3:  ---
        sampled_preds = torch.tensor(sampled_floats, device=device, dtype=torch.float64).squeeze(-1) # (B, S)
        # print(targets_norm)
        # print(sampled_preds)
        # print(greedy_preds)
        # assert 0

        #  NaN
        nan_mask = torch.isnan(sampled_preds)

        #
        if self.y_quantile_transformer is not None:
            #
            sampled_preds_np = sampled_preds.cpu().numpy()
            #  NaN transform  NaN
            #  NaN  reward
            nan_mask_np = np.isnan(sampled_preds_np)
            if nan_mask_np.any():
                y_means_np = y_means.cpu().numpy()
                #  y_means_np
                if y_means_np.ndim == 0:
                    y_mean_value = float(y_means_np)
                else:
                    #  y_means_np batch
                    y_mean_value = float(y_means_np[0]) if len(y_means_np) > 0 else 0.0
                #  NaN
                sampled_preds_np[nan_mask_np] = y_mean_value
            # print(sampled_preds_np)
            # print(greedy_preds_np)
            # print(targets_norm)
            sampled_preds_np_norm = self.y_quantile_transformer.transform(sampled_preds_np.reshape(-1, 1)).flatten()
            sampled_preds_norm = torch.tensor(sampled_preds_np_norm, device=device, dtype=torch.float64).reshape(sampled_preds.shape)
            #  NaN compute_reward  nan_mask  NaN
        else:
            # NaN
            sampled_preds_norm = (sampled_preds - y_means) / y_stds
        # assert 0
        # assert 0
        rewards = self.compute_reward(sampled_preds_norm, targets_norm, nan_mask=nan_mask) # (B, S)
        # print(sampled_preds_norm)
        # print(greedy_preds_norm)
        # print(targets_norm)

        # assert 0

        # ---  4:  (Advantage = R - b) ---
        #
        advantages = rewards - baseline_rewards.unsqueeze(1)
        # REINFORCE
        advantages_detached = advantages.detach()
        advantages_detached = advantages_detached.to(torch.float32)
        # print(rewards, baseline_rewards)
        # print(advantages_detached)

        # ---  5:  REINFORCE  ---
        #  log pi(a|s)
        seq_log_probs = sampled_log_probs.sum(dim=2) # (B, S)

        # REINFORCE : -E[log pi(a|s) * Advantage]
        policy_loss = -(seq_log_probs * advantages_detached).mean()
        total_loss = policy_loss + 0 #  policy_loss

        # print(total_loss)
        # debug_tensor("total_loss", total_loss)
        # assert 0
        # print(seq_log_probs)
        # print(policy_loss)
        # assert 0
        # policy_entropysample_with_logprobscompute_policy_entropy_sample
        # ---  6: KL  ---
        kl_loss = torch.tensor(0.0, device=device)
        if self.kl_weight > 0 and sampled_logits is not None:
            B, S, L, V = sampled_logits.shape

            #  logits
            current_logits_on_sampled = sampled_logits.view(B * S, L, V)

            #  logits
            with torch.no_grad():
                # Fix: sampled_ids is (B, S, L+1) where [:, :, 0] is BOS
                vectorized_sampled_ids = sampled_ids[:, :, 1:].reshape(B * S, L)
                ref_logits_on_sampled = self._get_logits_for_sequence(
                    ref_model, tokenizer, batch_on_device, vectorized_sampled_ids, self.num_samples
                )

            mask = torch.ones_like(vectorized_sampled_ids, dtype=torch.float64, device=device)
            kl_loss = self.compute_kl_divergence(current_logits_on_sampled, ref_logits_on_sampled, mask)
            total_loss = total_loss + self.kl_weight * kl_loss

        # ---  6.5:  (Expert CE Loss) ---
        expert_ce_loss = torch.tensor(0.0, device=device)
        if self.expert_ce_weight > 0:
            B, S, L, V = sampled_logits.shape

            # 1.  Ground Truth  token ID  (B, L)
            expert_token_ids = []
            for t in targets:
                #  tokenizer.float_to_token_ids  float  IDs
                if hasattr(tokenizer, 'float_to_token_ids'):
                    ids = tokenizer.float_to_token_ids(float(t))
                else:
                    #  tokenizer  ()
                    ids = [0] * L

                if len(ids) < L:
                    ids = ids + [tokenizer.pad_token_id or 0] * (L - len(ids))
                else:
                    ids = ids[:L]
                expert_token_ids.append(ids)
            gt_ids = torch.tensor(expert_token_ids, device=device) # (B, L)

            # 2.
            # gen_ids  token ( token)
            gen_ids = sampled_ids[:, :, 1:] if sampled_ids.shape[-1] == L + 1 else sampled_ids
            flat_gen_ids = gen_ids.reshape(B * S, L)
            flat_gt_ids = gt_ids.unsqueeze(1).expand(-1, S, -1).reshape(B * S, L)
            flat_logits = sampled_logits.reshape(B * S, L, V)

            # 3.  IEEE tokenizer  token ID
            tokens_to_find = ["<+>", "<->", "<0>", "<9>"]
            special_ids = tokenizer.convert_tokens_to_ids(tokens_to_find)
            pos_id, neg_id, zero_id, nine_id = special_ids

            #  token ""
            # : <+> -> 1.0, <-> -> 0.0, <0>..9> -> 0.0..9.0
            score_map = torch.zeros(V, device=device)
            score_map[pos_id] = 1.0
            score_map[neg_id] = 0.0
            digit_ids_list = []
            for i in range(10):
                d_id = tokenizer.convert_tokens_to_ids(f"<{i}>")
                if d_id != tokenizer.unk_token_id:
                    score_map[d_id] = float(i)
                    digit_ids_list.append(d_id)
            digit_ids = torch.tensor(digit_ids_list, device=device)

            dynamic_expert_targets = torch.zeros((B * S, L), dtype=torch.long, device=device)
            was_illegal = torch.zeros(B * S, dtype=torch.bool, device=device)

            # 4.
            for l in range(L):
                # 1.  token
                if l < 2:
                    is_legal_now = (flat_gen_ids[:, l] == pos_id) | (flat_gen_ids[:, l] == neg_id)
                else:
                    is_legal_now = torch.isin(flat_gen_ids[:, l], digit_ids)

                # 2.  target_l
                if l == 0:
                    #  (Number Sign) GT
                    target_l = flat_gt_ids[:, 0]
                elif l == 1:
                    #  (Exponent Sign)
                    ns_correct = (flat_gen_ids[:, 0] == flat_gt_ids[:, 0])
                    #  NS  '-'  0
                    target_l = torch.where(ns_correct, flat_gt_ids[:, 1], neg_id)
                elif l in [2, 3]:
                    #  (Exponent Digits)
                    ns_correct = (flat_gen_ids[:, 0] == flat_gt_ids[:, 0])
                    es_correct = (flat_gen_ids[:, 1] == flat_gt_ids[:, 1])
                    #  NS   ES  0
                    target_is_zero = (~ns_correct) | (ns_correct & ~es_correct)
                    target_l = torch.where(target_is_zero, zero_id, flat_gt_ids[:, l])
                else:
                    #  (Mantissa)
                    ns_correct = (flat_gen_ids[:, 0] == flat_gt_ids[:, 0])

                    #  ( index 1  l-1)
                    dec_prefix = flat_gen_ids[:, 1:l]
                    gt_prefix = flat_gt_ids[:, 1:l]

                    dec_scores = score_map[dec_prefix]
                    gt_scores = score_map[gt_prefix]

                    diffs = dec_scores - gt_scores
                    has_diff = (diffs != 0).any(dim=-1)
                    #
                    first_diff_idx = (diffs != 0).float().argmax(dim=-1, keepdim=True)
                    first_diff = torch.gather(diffs, -1, first_diff_idx).squeeze(-1)

                    is_greater = has_diff & (first_diff > 0)
                    is_less = has_diff & (first_diff < 0)

                    #  GT 0 9
                    mag_target = torch.where(has_diff,
                                            torch.where(is_greater, zero_id, nine_id),
                                            flat_gt_ids[:, l])

                    #  NS  0
                    target_l = torch.where(ns_correct, mag_target, zero_id)

                # 3.  fallback  GT
                dynamic_expert_targets[:, l] = torch.where(was_illegal | (~is_legal_now), flat_gt_ids[:, l], target_l)

                # 4.  was_illegal
                was_illegal |= (~is_legal_now)

            expert_ce_loss = F.cross_entropy(
                flat_logits.reshape(-1, V),
                dynamic_expert_targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
            )

            # --- DEBUG PRINT FOR EXPERT STRATEGY ---
            # print("\n" + "="*80)
            # print("Expert Strategy Debug (First Sample of Batch):")
            # sample_idx = 0
            # gen_ids_sample = flat_gen_ids[sample_idx]
            # gt_ids_sample = flat_gt_ids[sample_idx]
            # expert_targets_sample = dynamic_expert_targets[sample_idx]

            # gen_tokens = tokenizer.convert_ids_to_tokens(gen_ids_sample.tolist())
            # gt_tokens = tokenizer.convert_ids_to_tokens(gt_ids_sample.tolist())
            # expert_tokens = tokenizer.convert_ids_to_tokens(expert_targets_sample.tolist())

            # print(f"Generated IDs:    {gen_ids_sample.tolist()}")
            # print(f"Generated Tokens: {gen_tokens}")
            # print(f"Ground Truth IDs: {gt_ids_sample.tolist()}")
            # print(f"GT Tokens:        {gt_tokens}")
            # print(f"Expert Target IDs:{expert_targets_sample.tolist()}")
            # print(f"Expert Tokens:    {expert_tokens}")
            # print("="*80 + "\n")

            # assert 0
            # ----------------------------------------

            total_loss = total_loss + self.expert_ce_weight * expert_ce_loss

        # ---  7:  ---
        with torch.no_grad():
            metrics = {
                "policy_loss": policy_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "expert_ce_loss": expert_ce_loss.detach(),
                "total_loss": total_loss.detach(),
                "mean_reward_sampled": rewards.mean(),
                "mean_reward_baseline": baseline_rewards.mean(),
                "mean_advantage": advantages.mean(),
                "policy_entropy": policy_entropy.detach() if policy_entropy is not None else 0,
            }

        return total_loss, metrics

    def _get_logits_for_sequence(
        self, model, tokenizer, batch: Dict[str, torch.Tensor], decoded_ids: torch.Tensor, num_samples: int
    ) -> torch.Tensor:

        device = model.device
        # batch  on_device
        base_inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        batch_size = base_inputs["input_ids"].shape[0]

        expanded_inputs = {
            k: v.repeat_interleave(num_samples, dim=0) for k, v in base_inputs.items()
        }

        decoder_start_token_id = getattr(model.model.config, "decoder_start_token_id", tokenizer.bos_token_id)
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id or 0

        #  decoded_ids  ( __call__ )
        decoded_ids_on_device = decoded_ids.to(device)

        decoder_input_ids = torch.cat(
            [
                torch.full((batch_size * num_samples, 1), decoder_start_token_id, dtype=torch.long, device=device),
                decoded_ids_on_device[:, :-1]
            ],
            dim=1
        )

        model_inputs = expanded_inputs
        model_inputs["decoder_input_ids"] = decoder_input_ids

        # print("\n--- GRPO_loss DEBUGGING ---")
        # print(f"Model's expected device: {device}")
        # print(f"Actual model device: {next(model.model.parameters()).device}")
        # print("Checking devices of tensors in model_inputs:")
        # for key, value in model_inputs.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"  - Tensor '{key}' is on device: {value.device}")
        # print("--- END DEBUGGING ---\n")
        # assert 0

        outputs = model.model(**model_inputs)

        return outputs.logits