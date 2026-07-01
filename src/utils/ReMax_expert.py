from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class Remax_mse:
    """REINFORCEKL"""

    def __init__(
        self,
        temperature: float = 1.0,
        num_samples: int = 8,
        reward_scale: float = 1.0,
        kernel_sigma: float = 1.0,
        kernel_reduction: str = "mean",
        kl_weight: float = 0.0,
        entropy_weight: float = 0,
        expert_ce_weight: float = 0.0,
        # ref_modelinit__call__
    ):
        """
        REINFORCE

        Args:
            temperature:
            num_samples:
            reward_scale: reward
            kernel_sigma: sigma
            kernel_reduction: ()'mean''sum'
            kl_weight: KL
            entropy_weight:
            expert_ce_weight:
        """
        self.temperature = temperature
        self.num_samples = num_samples
        self.reward_scale = reward_scale
        self.kernel_sigma = kernel_sigma
        self.kernel_reduction = kernel_reduction
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.expert_ce_weight = expert_ce_weight
        # self.ref_model = ref_model # <--- ref_model

    def compute_reward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        rewardexp(-||y_hat - y||^2 / (2*sigma^2))

         (B,S)  (B,S,D)kernel_reduction

        Args:
            predictions:  (B, S)  (B, S, D)
            targets:  (B,)
        Returns:
            rewards:  (B, S)
        """
        sigma2 = float(self.kernel_sigma) ** 2 + 1e-12
        if predictions.dim() == 3:
            # (B, S, D)
            targets_expanded = (
                targets.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, predictions.size(1), predictions.size(2))
            )
            diff2 = (predictions - targets_expanded) ** 2  # (B, S, D)
            if self.kernel_reduction == "sum":
                dist2 = diff2.sum(dim=2)
            else:
                dist2 = diff2.mean(dim=2)
        else:
            # (B, S)
            targets_expanded = targets.unsqueeze(1).expand(-1, predictions.size(1))
            dist2 = (predictions - targets_expanded) ** 2  # (B, S)
        return -dist2

    # <--- : KL
    def compute_kl_divergence(
        self,
        current_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        KL: KL(current || ref)

        Args:
            current_logits: logits (B*S, L, V)
            ref_logits: logits (B*S, L, V)
            mask: tokenmask (B*S, L)

        Returns:
            kl_loss: KL
        """
        # log
        current_log_probs = F.log_softmax(current_logits, dim=-1)  # (B*S, L, V)
        ref_probs = F.softmax(ref_logits, dim=-1)  # (B*S, L, V)

        # F.kl_divKL: KL(P||Q)
        # F.kl_divlog_probprob
        kl_div = F.kl_div(
            current_log_probs,
            ref_probs,
            reduction='none'
        )  # (B*S, L, V)

        # tokenKL
        kl_div_per_token = kl_div.sum(dim=-1)  # (B*S, L)

        # masktokenKL
        if mask is not None:
            kl_div_per_token = kl_div_per_token * mask.float()
            kl_loss = kl_div_per_token.sum() / mask.float().sum()
        else:
            kl_loss = kl_div_per_token.mean()

        return kl_loss

    def compute_reinforce_loss(
        self, log_probs: torch.Tensor, rewards: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        """
        REINFORCE

        Args:
            log_probs:  shape (batch_size, num_samples, seq_len)
            rewards: reward shape (batch_size, num_samples)
            baseline: reward shape (batch_size, num_samples)

        Returns:
            loss: REINFORCE
        """
        #
        sequence_log_probs = log_probs.sum(dim=2)  # (batch_size, num_samples)

        # advantage
        advantage = rewards - baseline  # (batch_size, num_samples)
        # advantage0
        num_nonzero_advantage = (advantage != 0).sum().item()
        # detach advantageREINFORCE
        advantage = advantage.detach()
        advantages_std_per_batch = advantage.std(dim=1)  # (B,) - batchSrolloutstd
        mean_advantages_std = advantages_std_per_batch.mean()  # scalar - batchstd

        # REINFORCE
        # rewardlog_prob * advantage
        reinforce_loss = -(sequence_log_probs * advantage).mean()

        return reinforce_loss, mean_advantages_std, num_nonzero_advantage

    def compute_policy_entropy(
        self, model, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        token

        Args:
            model:
            batch: batch

        Returns:
            policy_entropy:
        """
        batch_size = batch["encoder_input"].shape[0]
        device = batch["encoder_input"].device

        #
        # sample_with_logprobslogitslog_prob
        with torch.no_grad():
            # Encode
            memory, memory_key_padding_mask = model.encoder_decoder.encode(batch["encoder_input"])

            #
            # batch_sizebatch_size * num_samples

            # decoderBOS
            current_tgt_ids = torch.full(
                (batch_size, 1),
                model.decoder_vocab.bos_pad_id,
                dtype=torch.long,
                device=device,
            )

            #
            step_entropies = torch.zeros(
                (batch_size, model.decode_len),
                dtype=torch.float32,
                device=device,
            )

            for step_idx in range(model.decode_len):
                # logits
                logits = model.encoder_decoder.next_token_logits(
                    current_tgt_ids, memory, memory_key_padding_mask
                )

                # mask
                # mask -
                curr_mask = model.decoder_constraint_masks[step_idx, :].unsqueeze(0)  # (1, V)
                #
                MASK_VALUE = -1e7
                masked_logits = (1.0 - curr_mask) * MASK_VALUE + curr_mask * logits

                #
                scaled_logits = masked_logits / self.temperature
                probs = F.softmax(scaled_logits, dim=-1)  # (B, V)
                log_probs = F.log_softmax(scaled_logits, dim=-1)  # (B, V)

                # H = -sum(p * log(p))
                # 0*inf=nan
                p_log_p = probs * log_probs
                # NaN0
                p_log_p = torch.nan_to_num(p_log_p, nan=0.0, posinf=0.0, neginf=0.0)
                entropy = -torch.sum(p_log_p, dim=-1)  # (B,)
                step_entropies[:, step_idx] = entropy

                #
                next_token_ids = torch.argmax(scaled_logits, dim=-1, keepdim=True)  # (B, 1)

                #
                if step_idx < model.decode_len - 1:
                    current_tgt_ids = torch.cat([current_tgt_ids, next_token_ids], dim=1)

            #
            policy_entropy = step_entropies.mean()

        return policy_entropy

    def compute_policy_entropy_sample(
        self, model, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        token

        Args:
            model:
            batch: batch

        Returns:
            policy_entropy:
        """
        batch_size = batch["encoder_input"].shape[0]
        device = batch["encoder_input"].device

        #
        with torch.no_grad():
            # Encode
            memory, memory_key_padding_mask = model.encoder_decoder.encode(batch["encoder_input"])

            #  B*S batch
            num_samples = self.num_samples
            expanded_memory = (
                memory.unsqueeze(1)
                .expand(-1, num_samples, -1, -1)
                .reshape(batch_size * num_samples, memory.size(1), memory.size(2))
            )

            if memory_key_padding_mask is not None:
                expanded_memory_key_padding_mask = (
                    memory_key_padding_mask.unsqueeze(1)
                    .expand(-1, num_samples, -1)
                    .reshape(batch_size * num_samples, memory_key_padding_mask.size(1))
                )
            else:
                expanded_memory_key_padding_mask = None

            # decoderBOS
            current_tgt_ids = torch.full(
                (batch_size * num_samples, 1),
                model.decoder_vocab.bos_pad_id,
                dtype=torch.long,
                device=device,
            )

            #
            step_entropies = torch.zeros(
                (batch_size * num_samples, model.decode_len),
                dtype=torch.float32,
                device=device,
            )

            for step_idx in range(model.decode_len):
                # logits
                logits = model.encoder_decoder.next_token_logits(
                    current_tgt_ids,
                    expanded_memory,
                    expanded_memory_key_padding_mask
                )

                # mask
                curr_mask = model.decoder_constraint_masks[step_idx, :].unsqueeze(0)  # (1, V)
                                #
                MASK_VALUE = -1e7
                masked_logits = (1.0 - curr_mask) * MASK_VALUE + curr_mask * logits

                #
                scaled_logits = masked_logits / self.temperature
                probs = F.softmax(scaled_logits, dim=-1)  # (B*S, V)
                log_probs = F.log_softmax(scaled_logits, dim=-1)  # (B*S, V)

                # H = -sum(p * log(p))
                # 0*inf=nan
                p_log_p = probs * log_probs
                # NaN0
                p_log_p = torch.nan_to_num(p_log_p, nan=0.0, posinf=0.0, neginf=0.0)
                entropy = -torch.sum(p_log_p, dim=-1)  # (B*S,)
                step_entropies[:, step_idx] = entropy

                # tokensample_with_logprobs
                next_token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)

                #
                if step_idx < model.decode_len - 1:
                    current_tgt_ids = torch.cat([current_tgt_ids, next_token_ids], dim=1)

            #  (B, S, L_decode)
            step_entropies = step_entropies.reshape(batch_size, num_samples, model.decode_len)

            #  (B, S)
            sample_entropies = step_entropies.mean(dim=-1)

            #  (B)
            batch_entropies = sample_entropies.mean(dim=-1)

            #  ()
            policy_entropy = batch_entropies.mean()

        return policy_entropy

    def __call__(
        self, model, batch: Dict[str, torch.Tensor], ref_model=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        REINFORCEKL

        Args:
            model:
            batch: batch
            ref_model: KL

        Returns:
            loss:
            metrics:
        """
        model.eval()
        if ref_model is not None:
            ref_model.eval()
        batch_size = batch["encoder_input"].shape[0]
        device = batch["encoder_input"].device

        # 1.
        # log prob no_grad
        decoded_ids, output_floats, step_log_probs, step_logits = model.sample_with_logprobs(
            batch, self.num_samples, self.temperature, return_logits=True
        )
        # decoded_ids: (B, S, L), output_floats: (B, S),
        # step_log_probs: (B, S, L), step_logits: (B, S, L, V)

        # 2. Reward
        predictions = torch.tensor(
            output_floats, device=device, dtype=torch.float32
        )  # (B, S)
        targets = batch["y"]  # (B,)
        y_max = batch["y_max"]
        y_min = batch["y_min"]
        y_max_bs = y_max.unsqueeze(1)
        y_min_bs = y_min.unsqueeze(1)
        #  (B,)  (B,1)  (B,S)
        predictions_true = predictions.squeeze(-1)
        predictions_norm = predictions_true * (y_max_bs - y_min_bs + 1e-8) + y_min_bs
        targets_norm = targets * (y_max - y_min + 1e-8) + y_min
        rewards = self.compute_reward(predictions_norm, targets_norm)  # (B, S)

        # batchrolloutstd
        reward_std_per_batch = rewards.std(dim=1)  # (B,) - batchSrolloutstd
        mean_reward_std = reward_std_per_batch.mean()  # scalar - batchstd

        # 3. Baseline
        with torch.no_grad():
            greedy_decoded_ids, greedy_output_floats = model.greedy_decode(
                batch, num_samples=1
            )
        greedy_predictions = torch.tensor(
            greedy_output_floats, device=device, dtype=torch.float32
        )  # (batch_size, 1, max_num_objs)
        greedy_predictions = greedy_predictions[:, :, 0]  # (batch_size, 1)
        greedy_predictions = greedy_predictions.expand(-1, self.num_samples) # (B, S)
        greedy_predictions_norm = greedy_predictions * (y_max_bs - y_min_bs + 1e-8) + y_min_bs
        baseline = self.compute_reward(greedy_predictions_norm, targets_norm)  # (B, S)
        # baselineSgreedy_predictionsexpand
        # baseline
        mean_baseline = baseline.mean().detach()  # batchmean
        # 4. REINFORCE
        log_probs = step_log_probs  # (B, S, L)
        # rewardsbaselinebaseline
        num_rewards_ge_baseline = (rewards >= baseline).sum().item()
        num_rewards_lt_baseline = (rewards < baseline).sum().item()
        reinforce_loss, mean_advantages_std, num_nonzero_advantage = self.compute_reinforce_loss(log_probs, rewards, baseline)

        # ---  ---
        with torch.no_grad():
            #  sample  reward (B,)
            # mean_reward_per_sample = rewards.mean(dim=1)
            #  sample  advantage : mean(|reward - baseline|)
            # baseline  S
            mean_abs_advantage_per_sample = (rewards - baseline).abs().mean(dim=1)
            #  sample  baseline
            abs_baseline_per_sample = baseline[:, 0].abs()
            # : mean(|advantage|) / (|baseline| + eps)
            relative_exploration_rate = mean_abs_advantage_per_sample / (abs_baseline_per_sample + 1e-8)

            # : reward < -0.1   < 0.05
            low_exploration_mask = (baseline[:, 0] < -0.1) & (relative_exploration_rate < 0.1)
            num_low_exploration_samples = low_exploration_mask.sum().item()
            low_exploration_proportion = num_low_exploration_samples / batch_size
            mean_rel_exploration_rate = relative_exploration_rate.mean().item()

        # 5. KL ()
        total_loss = reinforce_loss
        kl_loss = torch.tensor(0.0, device=device) # <--- :  requires_grad

        if ref_model is not None and self.kl_weight > 0:
            # logits
            # : (B, S, L, V) -> (B*S, L, V)
            _, _, seq_len, vocab_size = step_logits.shape
            current_logits = step_logits.view(batch_size * self.num_samples, seq_len, vocab_size)

            # ID
            # (B, S, L) -> (B*S, L)
            vectorized_decoded_ids = decoded_ids.view(batch_size * self.num_samples, seq_len)

            # maskpadding tokens
            mask = (vectorized_decoded_ids != model.decoder_vocab.bos_pad_id).float()

            # logits
            with torch.no_grad():
                ref_logits = self._get_logits_for_sequence(
                    ref_model, batch, vectorized_decoded_ids, self.num_samples
                )

            # KL
            kl_loss = self.compute_kl_divergence(current_logits, ref_logits, mask)

            total_loss = total_loss + self.kl_weight * kl_loss

        # 6.  (Differentiable)
        #  step_logits
        entropy_mask = (decoded_ids != model.decoder_vocab.bos_pad_id).float()

        #  token : H = -sum(p * log_p)
        probs = F.softmax(step_logits, dim=-1)
        log_probs = F.log_softmax(step_logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, S, L)
        token_entropy = torch.nan_to_num(token_entropy, nan=0.0)

        # 1.  (:  token )
        if entropy_mask.sum() > 0:
            policy_entropy = (token_entropy * entropy_mask).sum() / entropy_mask.sum()
        else:
            policy_entropy = token_entropy.mean()

        # 2.  ""  ()
        low_exp_filter = low_exploration_mask.view(batch_size, 1, 1).float()
        combined_entropy_mask = entropy_mask * low_exp_filter

        if combined_entropy_mask.sum() > 0:
            filtered_entropy = (token_entropy * combined_entropy_mask).sum() / combined_entropy_mask.sum()
        else:
            filtered_entropy = torch.tensor(0.0, device=device)

        if self.entropy_weight > 0:
            total_loss = total_loss - self.entropy_weight * filtered_entropy

        # 6.5  CE  (Expert CE Loss)
        expert_ce_loss = torch.tensor(0.0, device=device)
        expert_low_prob_ratio = torch.tensor(0.0, device=device)
        if self.expert_ce_weight > 0:
            #
            B, S, L, V = step_logits.shape

            # 1.  (B*S, L)
            flat_decoded_ids = decoded_ids.reshape(B * S, L)
            flat_logits = step_logits.reshape(B * S, L, V)

            # 2.  ground truth  token  (B, L)
            expert_token_ids = []
            for t in targets:
                #  vocab.to_token_ids  float  IDs tokenizer.to_tokens
                ids = model.decoder_vocab.to_token_ids(t.item())
                if len(ids) < L:
                    ids = ids + [model.decoder_vocab.bos_pad_id] * (L - len(ids))
                else:
                    ids = ids[:L]
                expert_token_ids.append(ids)
            gt_ids = torch.tensor(expert_token_ids, device=device) # (B, L)
            flat_gt_ids = gt_ids.unsqueeze(1).expand(-1, S, -1).reshape(B * S, L)

            # 3.  token id ( tokenizer )
            min_token_str = model.decoder_vocab.tokenizer.get_min_digit_token()
            max_token_str = model.decoder_vocab.tokenizer.get_max_digit_token()
            min_tid = model.decoder_vocab.stoi[min_token_str]
            max_tid = model.decoder_vocab.stoi[max_token_str]

            dynamic_expert_targets = torch.zeros((B * S, L), dtype=torch.long, device=device)

            # 4.
            for l in range(L):
                if l == 0:
                    # prefix  gt
                    dynamic_expert_targets[:, 0] = flat_gt_ids[:, 0]
                else:
                    #  (B*S, l)
                    curr_prefixes = flat_decoded_ids[:, :l]
                    gt_prefixes = flat_gt_ids[:, :l]

                    #  gt
                    diffs = curr_prefixes.float() - gt_prefixes.float()
                    has_diff = (diffs != 0).any(dim=-1)

                    #
                    first_diff_idx = (diffs != 0).float().argmax(dim=-1, keepdim=True)
                    first_diff = torch.gather(diffs, -1, first_diff_idx).squeeze(-1)

                    is_equal = ~has_diff
                    is_greater = has_diff & (first_diff > 0)
                    is_less = has_diff & (first_diff < 0)

                    l_gt = flat_gt_ids[:, l]
                    l_min = torch.full((B * S,), min_tid, device=device, dtype=torch.long)
                    l_max = torch.full((B * S,), max_tid, device=device, dtype=torch.long)

                    step_target = torch.where(is_equal, l_gt,
                                            torch.where(is_greater, l_min, l_max))
                    dynamic_expert_targets[:, l] = step_target

            #  ID
            # print("\n" + "="*80)
            # print("Expert CE Debug (First Sample of First Batch):")
            # print(f"Ground Truth (y):   {targets[2].item():.6f}")
            # print(f"Rollout IDs:        {flat_decoded_ids[2].tolist()}")
            # print(f"Static GT IDs:      {flat_gt_ids[2].tolist()}")
            # print(f"Dynamic Target IDs: {dynamic_expert_targets[2].tolist()}")

            # #  IDs  tokens
            # #  itos  token  from_token_ids float
            # rollout_tokens = [model.decoder_vocab.itos[i] for i in flat_decoded_ids[2].tolist()]
            # gt_tokens = [model.decoder_vocab.itos[i] for i in flat_gt_ids[2].tolist()]
            # dynamic_tokens = [model.decoder_vocab.itos[i] for i in dynamic_expert_targets[2].tolist()]
            # print(f"Rollout Tokens:     {rollout_tokens}")
            # print(f"Static GT Tokens:   {gt_tokens}")
            # print(f"Dynamic Tokens:     {dynamic_tokens}")
            # print("="*80 + "\n")

            # assert False
            # 5.
            expert_ce_loss = F.cross_entropy(
                flat_logits.reshape(-1, V),
                dynamic_expert_targets.reshape(-1),
                ignore_index=model.decoder_vocab.bos_pad_id
            )
            total_loss = total_loss + self.expert_ce_weight * expert_ce_loss

            # 5.5  Expert  Token pi_k < exp(-H)
            with torch.no_grad():
                flat_probs_all = F.softmax(flat_logits, dim=-1)          # (B*S, L, V)
                flat_log_probs_all = F.log_softmax(flat_logits, dim=-1)  # (B*S, L, V)
                p_log_p_all = flat_probs_all * flat_log_probs_all
                p_log_p_all = torch.nan_to_num(p_log_p_all, nan=0.0, posinf=0.0, neginf=0.0)
                H_per_pos = -p_log_p_all.sum(dim=-1)  # (B*S, L)
                expert_target_probs = torch.gather(
                    flat_probs_all,
                    dim=-1,
                    index=dynamic_expert_targets.unsqueeze(-1)
                ).squeeze(-1)  # (B*S, L)
                entropy_inc_threshold = torch.exp(-H_per_pos)  # (B*S, L)
                expert_valid_mask = (dynamic_expert_targets != model.decoder_vocab.bos_pad_id)
                below_threshold_mask = (expert_target_probs < entropy_inc_threshold) & expert_valid_mask
                num_valid = expert_valid_mask.float().sum().clamp(min=1)
                expert_low_prob_ratio = below_threshold_mask.float().sum() / num_valid

        # 7.
        metrics = {
            "reinforce_loss": reinforce_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "entropy_loss": (-self.entropy_weight * filtered_entropy).detach() if self.entropy_weight > 0 else torch.tensor(0.0, device=device),
            "expert_ce_loss": expert_ce_loss.detach(),
            "total_loss": total_loss.detach(),
            "mean_reward": rewards.mean().detach(),
            "max_reward": rewards.max().detach(),
            "min_reward": rewards.min().detach(),
            "reward_std": rewards.std().detach(),
            "mean_reward_std": mean_reward_std.detach(),  # batchrolloutstd
            "mean_advantages_std": mean_advantages_std.detach(),  # batchrolloutstd
            "mean_baseline": mean_baseline,  # baseline
            "num_nonzero_advantage": num_nonzero_advantage,  # advantage0
            "num_rewards_ge_baseline": num_rewards_ge_baseline,  # rewardsbaseline
            "num_rewards_lt_baseline": num_rewards_lt_baseline,  # rewardsbaseline
            "num_low_exploration_samples": num_low_exploration_samples,  # reward < -0.1   < 0.05
            "low_exploration_proportion": low_exploration_proportion,  #
            "mean_relative_exploration_rate": mean_rel_exploration_rate,  #
            "mean_prediction": predictions.mean().detach(),
            "prediction_std": predictions.std().detach(),
            "policy_entropy": policy_entropy.detach(),  #  ()
            "filtered_entropy": filtered_entropy.detach(),  #  ( regularization)
            "expert_low_prob_ratio": expert_low_prob_ratio.detach(),  # expert  token  pi_k < exp(-H)
        }

        model.train()

        return total_loss, metrics


    # <--- : logits
    def _get_logits_for_sequence(
        self, model, batch: Dict[str, torch.Tensor], decoded_ids: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """
        logits ()

        Args:
            model:  (ref_model)
            batch: batchencoder_input
            decoded_ids: token ids (B*S, L)
            num_samples: encoder_input

        Returns:
            logits: logits (B*S, L, V)
        """
        encoder_input = batch["encoder_input"]
        batch_size = encoder_input.shape[0]
        device = encoder_input.device

        # encoder_inputdecoded_idsbatch size
        # (B, L_enc) -> (B*S, L_enc)
        expanded_encoder_input = encoder_input.repeat_interleave(num_samples, dim=0)

        # sample_with_logprobs
        # encodelogits
        memory, memory_key_padding_mask = model.encoder_decoder.encode(expanded_encoder_input)

        # decoderBOSsample_with_logprobs
        current_tgt_ids = torch.full(
            (batch_size * num_samples, 1),
            model.decoder_vocab.bos_pad_id,
            dtype=torch.long,
            device=device,
        )

        step_logits = torch.zeros(
            (batch_size * num_samples, decoded_ids.shape[1], len(model.decoder_vocab)),
            dtype=torch.float32,
            device=device,
        )

        # logitssample_with_logprobs
        for step_idx in range(decoded_ids.shape[1]):
            # logits
            logits = model.encoder_decoder.next_token_logits(
                current_tgt_ids, memory, memory_key_padding_mask
            )
            step_logits[:, step_idx, :] = logits

            # token
            if step_idx < decoded_ids.shape[1] - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, decoded_ids[:, step_idx:step_idx+1]], dim=1)

        return step_logits