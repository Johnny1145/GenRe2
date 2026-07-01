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
        kl_weight: float = 0.1,
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
        """
        self.temperature = temperature
        self.num_samples = num_samples
        self.reward_scale = reward_scale
        self.kernel_sigma = kernel_sigma
        self.kernel_reduction = kernel_reduction
        self.kl_weight = kl_weight
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

            # # logits
            # print(f"Current logits shape: {current_logits.shape}")
            # print(f"Ref logits shape: {ref_logits.shape}")
            # print(f"Current logits sample 0: {current_logits[0, :5, :5]}")
            # print(f"Ref logits sample 0: {ref_logits[0, :5, :5]}")
            # print(f"Current logits sample 1: {current_logits[1, :5, :5]}")
            # print(f"Ref logits sample 1: {ref_logits[1, :5, :5]}")
            # assert False

            # KL
            kl_loss = self.compute_kl_divergence(current_logits, ref_logits, mask)

            # batchKL0
            # print(f"KL divergence: {kl_loss.item():.6f}")
            # print(f"Current logits mean: {current_logits.mean().item():.4f}, std: {current_logits.std().item():.4f}")
            # print(f"Ref logits mean: {ref_logits.mean().item():.4f}, std: {ref_logits.std().item():.4f}")
            # print(f"Current logits min/max: {current_logits.min().item():.4f}/{current_logits.max().item():.4f}")
            # print(f"Ref logits min/max: {ref_logits.min().item():.4f}/{ref_logits.max().item():.4f}")
            # print(f"Mask sum: {mask.sum().item()}, Mask mean: {mask.mean().item():.4f}")

            # logits
            # logits_diff = torch.abs(current_logits - ref_logits).mean()
            # print(f"Mean absolute difference between current and ref logits: {logits_diff.item():.6f}")

            # #
            # current_params = list(model.parameters())
            # ref_params = list(ref_model.parameters())
            # if len(current_params) == len(ref_params):
            #     param_diffs = []
            #     for cp, rp in zip(current_params, ref_params):
            #         param_diff = torch.abs(cp - rp).mean()
            #         param_diffs.append(param_diff.item())
            #     print(f"Mean parameter differences: {sum(param_diffs)/len(param_diffs):.6f}")
            # assert False

            total_loss = total_loss + self.kl_weight * kl_loss
        #
        policy_entropy = self.compute_policy_entropy_sample(model, batch)
        # 6.
        metrics = {
            "reinforce_loss": reinforce_loss.detach(),
            "kl_loss": kl_loss.detach(),
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
            "mean_prediction": predictions.mean().detach(),
            "prediction_std": predictions.std().detach(),
            "policy_entropy": policy_entropy.detach(),  #
        }

        # print(kl_loss.item())
        # assert False
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