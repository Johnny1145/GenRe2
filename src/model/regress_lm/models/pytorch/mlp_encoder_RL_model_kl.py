# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch implementation of a RegressLM."""

import math
import re
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from src.model.regress_lm import core, vocabs
from src.utils.number_token_loss import NumberTokenLoss
from src.utils.number_token_selector import NumberTokenSelector

from .. import base as model_base
from . import architecture

NEG_INF = -1.0e7

# Dict Keys: "encoder_input", "decoder_input", "decoder_target"
Tensor = torch.Tensor

max_len = -1
all_lengths = []


class PyTorchModel(nn.Module, model_base.Model[Tensor]):
    """PyTorch implementation of a RegressLM."""

    def __init__(
        self,
        encoder_vocab: vocabs.EncoderVocab[str],
        decoder_vocab: vocabs.DecoderVocab[float],
        max_input_len: int = 2048,
        max_num_objs: int = 1,
        learning_rate: float = 1e-4,
        z_loss_coef: float | None = None,
        if_ntl: bool = False,
        encoder_type: str = "vanilla",
        plot: bool = False,
        **architecture_kwargs,
    ):
        super().__init__()
        self.max_input_len = max_input_len
        self.max_num_objs = max_num_objs
        self.z_loss_coef = z_loss_coef
        self.plot = plot
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self.encoder_type = encoder_type
        self.encoder_decoder = architecture.CustomEncoderDecoder(
            custom_encoder=architecture.MLPEncoder(**architecture_kwargs),
            decoder_vocab_size=len(self.decoder_vocab),
            encoder_pad_idx=self.encoder_vocab.pad_id,
            max_decoder_len=self.decode_len + 1,
            plot=self.plot,
            **architecture_kwargs,
        )
        self.if_ntl = if_ntl

        # Pre-compute the constraint masks for the decoder.
        self.register_buffer(
            "decoder_constraint_masks", self._create_decoder_constraint_masks()
        )

    @property
    def decode_len(self) -> int:
        return self.max_num_objs * self.decoder_vocab.num_tokens_per_obj

    def compute_loss_and_metrics(
        self, examples: dict[str, Tensor], NTL: NumberTokenLoss | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # print('hello')
        metrics = {}
        if "number_mask" in examples:
            logits = self.encoder_decoder.forward(
                examples["encoder_input"],
                examples["decoder_input"],
                examples["number_mask"],
            )
        else:
            logits = self.encoder_decoder.forward(
                examples["encoder_input"], examples["decoder_input"]
            )
        targets = examples["decoder_target"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),  # (B * L_decode, V)
            targets.reshape(-1),  # Reshape to (B * L_decode)
            ignore_index=self.decoder_vocab.bos_pad_id,
        )
        metrics["ce_loss"] = loss.detach()

        if self.z_loss_coef is not None:
            # Calculate z_loss (log-softmax normalization constant).
            log_z = torch.logsumexp(logits, dim=-1)  # (B * L_decode)
            z_loss_per_token = self.z_loss_coef * (log_z**2)

            # Calculate the mean z_loss over the real (non-padded) tokens.
            loss_mask = (targets != self.decoder_vocab.bos_pad_id).float()
            z_loss = (z_loss_per_token * loss_mask).sum() / loss_mask.sum()
            metrics["z_loss"] = z_loss.detach()
            loss += z_loss

        metrics["loss"] = loss.detach()
        return loss, metrics

    def compute_loss_and_metrics_with_reinforce(
        self,
        examples: dict[str, Tensor],
        NTL: NumberTokenLoss | None = None,
        reinforce_loss_fn=None,
        ref_model=None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """REINFORCEKL"""
        reinforce_loss, reinforce_metrics = reinforce_loss_fn(self, examples, ref_model)

        #
        metrics = {**reinforce_metrics}
        metrics["total_loss"] = reinforce_loss.detach()

        return reinforce_loss, metrics

    @torch.no_grad()
    def get_mean_pooling_embedding(self, examples: dict[str, Tensor]) -> Tensor:
        self.encoder_decoder.eval()
        encoder_input = examples["encoder_input"]
        memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)
        return self.encoder_decoder.mean_pooling(memory, memory_key_padding_mask)

    @torch.no_grad()
    def decode(
        self,
        inputs: dict[str, Tensor],
        num_samples: int,
        temperature: float = 1.0,
    ) -> tuple[Tensor, np.ndarray]:
        self.encoder_decoder.eval()
        encoder_input = inputs["encoder_input"]  # (B, L_src)
        device = encoder_input.device
        batch_size = encoder_input.shape[0]
        # memory: (B, L_src, D_model), memory_key_padding_mask: (B, L_src)
        memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

        # Expand encoder outputs and masks for num_samples
        # Effectively, new batch_size = B * num_samples
        # memory: (B, L_src, D) -> (B, 1, L_src, D) -> (B, S, L_src, D)
        # -> (B*S, L_src, D)
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

        # Initialize decoder input for the expanded batch, start with <pad>.
        current_tgt_ids = torch.full(
            (batch_size * num_samples, 1),
            self.decoder_vocab.bos_pad_id,
            dtype=torch.long,
            device=device,
        )

        # Store all generated token IDs for all sequences in the expanded batch
        generated_sequences_ids = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.long,
            device=device,
        )

        # Batched autoregressive decoding loop
        for step_idx in range(self.decode_len):
            # Get logits for the next token for all (B * num_samples) sequences
            # Shape: (B*S, V)
            logits = self.encoder_decoder.next_token_logits(
                current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
            )

            # Apply constraints using the pre-computed mask
            curr_mask = self.decoder_constraint_masks[step_idx, :]  # (V,)
            curr_mask = curr_mask.unsqueeze(0)  # (1, V)
            masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

            # Apply temperature sampling, 1 token for each of the B*S sequences
            probs = F.softmax(masked_logits / temperature, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)
            # Store the predicted token IDs
            generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

            # Prepare input for the next step, but only if not the last float token
            if step_idx < self.decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        # Reshape outputs back to (B, num_samples, L_decode)
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, self.decode_len
        )

        # Compute equivalent floats.
        output_floats = np.zeros(
            (batch_size, num_samples, self.max_num_objs), dtype=float
        )
        for b in range(batch_size):
            for s_idx in range(num_samples):
                output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                    final_decoded_ids[b, s_idx, :].tolist()
                )

        return final_decoded_ids, output_floats

    @torch.no_grad()
    def decode_with_embeddings(
        self,
        inputs: dict[str, Tensor],
        num_samples: int,
        temperature: float = 1.0,
    ) -> tuple[Tensor, np.ndarray, Dict[str, np.ndarray]]:
        """Decode with embedding extraction for analysis."""
        self.encoder_decoder.eval()
        encoder_input = inputs["encoder_input"]  # (B, L_src)
        device = encoder_input.device
        batch_size = encoder_input.shape[0]

        # Encode input
        memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

        # Expand memory for multiple samples
        expanded_memory = memory.repeat_interleave(num_samples, dim=0)
        expanded_memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(
            num_samples, dim=0
        )

        # Initialize target sequence (start with padding token)
        current_tgt_ids = torch.full(
            (batch_size * num_samples, 1),
            self.decoder_vocab.pad_idx,
            device=device,
            dtype=torch.long,
        )

        # Store all generated token IDs for all sequences in the expanded batch
        generated_sequences_ids = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.long,
            device=device,
        )

        # embedding
        step_embeddings = {}

        # Batched autoregressive decoding loop
        for step_idx in range(self.decode_len):
            # Get logits and embeddings for the next token
            logits, embeddings = self.encoder_decoder.next_token_logits(
                current_tgt_ids,
                expanded_memory,
                expanded_memory_key_padding_mask,
                return_embedding=True,
            )

            # embedding
            step_embeddings[f"step_{step_idx}"] = embeddings.cpu().numpy()

            # Apply constraints using the pre-computed mask
            curr_mask = self.decoder_constraint_masks[step_idx, :]  # (V,)
            curr_mask = curr_mask.unsqueeze(0)  # (1, V)
            masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

            # Apply temperature sampling, 1 token for each of the B*S sequences
            probs = F.softmax(masked_logits / temperature, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)
            # Store the predicted token IDs
            generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

            # Prepare input for the next step, but only if not the last float token
            if step_idx < self.decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        # Reshape outputs back to (B, num_samples, L_decode)
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, self.decode_len
        )

        # Compute equivalent floats.
        output_floats = np.zeros(
            (batch_size, num_samples, self.max_num_objs), dtype=float
        )
        for b in range(batch_size):
            for s_idx in range(num_samples):
                output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                    final_decoded_ids[b, s_idx, :].tolist()
                )

        return final_decoded_ids, output_floats, step_embeddings

    @torch.no_grad()
    def greedy_decode(
        self,
        inputs: dict[str, Tensor],
        num_samples: int,
        temperature: float = 1.0,
    ) -> tuple[Tensor, np.ndarray]:
        self.encoder_decoder.eval()
        encoder_input = inputs["encoder_input"]  # (B, L_src)
        device = encoder_input.device
        batch_size = encoder_input.shape[0]
        # memory: (B, L_src, D_model), memory_key_padding_mask: (B, L_src)
        memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

        # Expand encoder outputs and masks for num_samples
        # Effectively, new batch_size = B * num_samples
        # memory: (B, L_src, D) -> (B, 1, L_src, D) -> (B, S, L_src, D)
        # -> (B*S, L_src, D)
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

        # Initialize decoder input for the expanded batch, start with <pad>.
        current_tgt_ids = torch.full(
            (batch_size * num_samples, 1),
            self.decoder_vocab.bos_pad_id,
            dtype=torch.long,
            device=device,
        )

        # Store all generated token IDs for all sequences in the expanded batch
        generated_sequences_ids = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.long,
            device=device,
        )

        # Batched autoregressive decoding loop
        for step_idx in range(self.decode_len):
            # Get logits for the next token for all (B * num_samples) sequences
            # Shape: (B*S, V)
            logits = self.encoder_decoder.next_token_logits(
                current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
            )

            # Apply constraints using the pre-computed mask
            curr_mask = self.decoder_constraint_masks[step_idx, :]  # (V,)
            curr_mask = curr_mask.unsqueeze(0)  # (1, V)
            masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

            # Greedy decoding: select the token with highest probability
            token_ids = torch.argmax(masked_logits, dim=-1, keepdim=True)  # (B*S, 1)
            # Store the predicted token IDs
            generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

            # Prepare input for the next step, but only if not the last float token
            if step_idx < self.decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        # Reshape outputs back to (B, num_samples, L_decode)
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, self.decode_len
        )

        # Compute equivalent floats.
        output_floats = np.zeros(
            (batch_size, num_samples, self.max_num_objs), dtype=float
        )
        for b in range(batch_size):
            for s_idx in range(num_samples):
                output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                    final_decoded_ids[b, s_idx, :].tolist()
                )

        return final_decoded_ids, output_floats

    @torch.no_grad()
    def decode_with_mlp_encoder(
        self,
        inputs: dict[str, Tensor],
        num_samples: int,
        temperature: float = 1.0,
    ) -> tuple[Tensor, np.ndarray]:
        self.encoder_decoder.eval()
        encoder_input = inputs["encoder_input"]  # (B, L_src)
        device = encoder_input.device
        batch_size = encoder_input.shape[0]
        # memory: (B, L_src, D_model), memory_key_padding_mask: (B, L_src)
        memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

        # Expand encoder outputs and masks for num_samples
        # Effectively, new batch_size = B * num_samples
        # memory: (B, L_src, D) -> (B, 1, L_src, D) -> (B, S, L_src, D)
        # -> (B*S, L_src, D)
        expanded_memory = (
            memory.unsqueeze(1)
            .expand(-1, num_samples, -1, -1)
            .reshape(batch_size * num_samples, memory.size(1), memory.size(2))
        )

        # Handle the case where memory_key_padding_mask might be None (for MLP encoder)
        if memory_key_padding_mask is not None:
            expanded_memory_key_padding_mask = (
                memory_key_padding_mask.unsqueeze(1)
                .expand(-1, num_samples, -1)
                .reshape(batch_size * num_samples, memory_key_padding_mask.size(1))
            )
        else:
            expanded_memory_key_padding_mask = None

        # Initialize decoder input for the expanded batch, start with <pad>.
        current_tgt_ids = torch.full(
            (batch_size * num_samples, 1),
            self.decoder_vocab.bos_pad_id,
            dtype=torch.long,
            device=device,
        )

        # Store all generated token IDs for all sequences in the expanded batch
        generated_sequences_ids = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.long,
            device=device,
        )

        # Batched autoregressive decoding loop
        for step_idx in range(self.decode_len):
            # Get logits for the next token for all (B * num_samples) sequences
            # Shape: (B*S, V)
            logits = self.encoder_decoder.next_token_logits(
                current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
            )

            # Apply constraints using the pre-computed mask
            curr_mask = self.decoder_constraint_masks[step_idx, :]  # (V,)
            curr_mask = curr_mask.unsqueeze(0)  # (1, V)
            masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

            # Apply temperature sampling, 1 token for each of the B*S sequences
            probs = F.softmax(masked_logits / temperature, dim=-1)
            # print(probs)
            # assert 0
            token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)
            # Store the predicted token IDs
            generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

            # Prepare input for the next step, but only if not the last float token
            if step_idx < self.decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        # Reshape outputs back to (B, num_samples, L_decode)
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, self.decode_len
        )
        # print(final_decoded_ids)
        # assert 0

        # Compute equivalent floats.
        output_floats = np.zeros(
            (batch_size, num_samples, self.max_num_objs), dtype=float
        )
        # print(final_decoded_ids.shape)
        # assert 0
        for b in range(batch_size):
            for s_idx in range(num_samples):
                output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                    final_decoded_ids[b, s_idx, :].tolist()
                )
        # print(output_floats)

        return final_decoded_ids, output_floats

    def sample_with_logprobs(
        self,
        inputs: dict[str, Tensor],
        num_samples: int,
        temperature: float = 1.0,
        return_logits: bool = True,
    ) -> tuple[Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        tokenlog prob


          - final_decoded_ids: (B, S, L_decode) token ids
          - output_floats: np.ndarray (B, S, max_num_objs) token ids
          - step_log_probs: (B, S, L_decode) tokenlog prob
          - step_logits: (B, S, L_decode, V) logits
        """
        # eval()/REINFORCElog prob
        encoder_input = inputs["encoder_input"]  # (B, L_src)
        device = encoder_input.device
        batch_size = encoder_input.shape[0]

        # Encode
        memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

        #  B*S batch
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
            self.decoder_vocab.bos_pad_id,
            dtype=torch.long,
            device=device,
        )

        # tokenlog prob
        generated_sequences_ids = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.long,
            device=device,
        )
        step_log_probs = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.float32,
            device=device,
        )
        if return_logits:
            step_logits = torch.zeros(
                (batch_size * num_samples, self.decode_len, len(self.decoder_vocab)),
                dtype=torch.float32,
                device=device,
            )

        for step_idx in range(self.decode_len):
            # (B*S, V)
            logits = self.encoder_decoder.next_token_logits(
                current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
            )

            # mask
            curr_mask = self.decoder_constraint_masks[step_idx, :].unsqueeze(
                0
            )  # (1, V)
            masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

            #
            scaled_logits = masked_logits / temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)  # (B*S, V)
            probs = torch.exp(log_probs)

            #
            token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)
            generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

            # tokenlog prob
            chosen_log_prob = torch.gather(log_probs, 1, token_ids).squeeze(1)  # (B*S)
            step_log_probs[:, step_idx] = chosen_log_prob

            # logitslogitsmask
            if return_logits:
                step_logits[:, step_idx, :] = logits

            #
            if step_idx < self.decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        #
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, self.decode_len
        )
        step_log_probs = step_log_probs.view(batch_size, num_samples, self.decode_len)
        if return_logits:
            step_logits = step_logits.view(batch_size, num_samples, self.decode_len, -1)
        else:
            step_logits = None

        # token ids
        output_floats = np.zeros(
            (batch_size, num_samples, self.max_num_objs), dtype=float
        )
        for b in range(batch_size):
            for s_idx in range(num_samples):
                output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                    final_decoded_ids[b, s_idx, :].tolist()
                )

        return final_decoded_ids, output_floats, step_log_probs, step_logits

    def sample_with_logprobs_reject(
        self,
        inputs: dict[str, Tensor],
        num_samples: int,
        temperature: float = 1.0,
        return_logits: bool = True,
        max_retries: int = 5,

    ) -> tuple[Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        tokenlog prob


          - final_decoded_ids: (B, S, L_decode) token ids
          - output_floats: np.ndarray (B, S, max_num_objs) token ids
          - step_log_probs: (B, S, L_decode) tokenlog prob
          - step_logits: (B, S, L_decode, V) logits
        """
        # eval()/REINFORCElog prob
        encoder_input = inputs["encoder_input"]  # (B, L_src)
        device = encoder_input.device
        batch_size = encoder_input.shape[0]

        # Encode
        memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

        #  B*S batch
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
            self.decoder_vocab.bos_pad_id,
            dtype=torch.long,
            device=device,
        )

        # tokenlog prob
        generated_sequences_ids = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.long,
            device=device,
        )
        step_log_probs = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.float32,
            device=device,
        )
        if return_logits:
            step_logits = torch.zeros(
                (batch_size * num_samples, self.decode_len, len(self.decoder_vocab)),
                dtype=torch.float32,
                device=device,
            )

        for step_idx in range(self.decode_len):
            # (B*S, V)
            logits = self.encoder_decoder.next_token_logits(
                current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
            )

            # mask
            curr_mask = self.decoder_constraint_masks[step_idx, :].unsqueeze(
                0
            )  # (1, V)
            masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

            #
            scaled_logits = masked_logits / temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)  # (B*S, V)
            probs = torch.exp(log_probs)

            #
            token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)
            generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

            # tokenlog prob
            chosen_log_prob = torch.gather(log_probs, 1, token_ids).squeeze(1)  # (B*S)
            step_log_probs[:, step_idx] = chosen_log_prob

            # logitslogitsmask
            if return_logits:
                step_logits[:, step_idx, :] = logits

            #
            if step_idx < self.decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        #
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, self.decode_len
        )
        step_log_probs = step_log_probs.view(batch_size, num_samples, self.decode_len)
        if return_logits:
            step_logits = step_logits.view(batch_size, num_samples, self.decode_len, -1)
        else:
            step_logits = None

        # token ids
        output_floats = np.zeros(
            (batch_size, num_samples, self.max_num_objs), dtype=float
        )
        for b in range(batch_size):
            for s_idx in range(num_samples):
                output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                    final_decoded_ids[b, s_idx, :].tolist()
                )

        #
        if ("y_mean" in inputs) and ("y_std" in inputs) and max_retries > 0:
            y_mean = inputs["y_mean"].to(device)
            y_std = inputs["y_std"].to(device)

            #  (B, S)
            lower = (y_mean - 4.0 * y_std).unsqueeze(1).expand(-1, num_samples)
            upper = (y_mean + 4.0 * y_std).unsqueeze(1).expand(-1, num_samples)

            # tensor (B, S, O)
            output_floats_t = torch.tensor(output_floats, device=device, dtype=torch.float32)

            #
            below = (output_floats_t < lower.unsqueeze(-1))
            above = (output_floats_t > upper.unsqueeze(-1))
            violating = (below | above).any(dim=-1)  # (B, S)

            retries = 0
            while violating.any() and retries < max_retries:
                #  B*S
                violating_indices = violating.nonzero(as_tuple=False)
                expanded_rows = violating_indices[:, 0] * num_samples + violating_indices[:, 1]

                # mask
                sub_memory = expanded_memory.index_select(0, expanded_rows)
                if expanded_memory_key_padding_mask is not None:
                    sub_mask = expanded_memory_key_padding_mask.index_select(0, expanded_rows)
                else:
                    sub_mask = None

                #
                sub_current_tgt_ids = torch.full(
                    (expanded_rows.shape[0], 1),
                    self.decoder_vocab.bos_pad_id,
                    dtype=torch.long,
                    device=device,
                )
                sub_generated_ids = torch.zeros(
                    (expanded_rows.shape[0], self.decode_len),
                    dtype=torch.long,
                    device=device,
                )
                sub_step_log_probs = torch.zeros(
                    (expanded_rows.shape[0], self.decode_len),
                    dtype=torch.float32,
                    device=device,
                )
                if return_logits:
                    sub_step_logits = torch.zeros(
                        (expanded_rows.shape[0], self.decode_len, len(self.decoder_vocab)),
                        dtype=torch.float32,
                        device=device,
                    )

                #
                for step_idx in range(self.decode_len):
                    sub_logits = self.encoder_decoder.next_token_logits(
                        sub_current_tgt_ids, sub_memory, sub_mask
                    )
                    curr_mask = self.decoder_constraint_masks[step_idx, :].unsqueeze(0)
                    masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * sub_logits
                    scaled_logits = masked_logits / temperature
                    sub_log_probs = F.log_softmax(scaled_logits, dim=-1)
                    sub_probs = torch.exp(sub_log_probs)
                    sub_token_ids = torch.multinomial(sub_probs, num_samples=1)
                    sub_generated_ids[:, step_idx] = sub_token_ids.squeeze(-1)
                    sub_chosen_log_prob = torch.gather(sub_log_probs, 1, sub_token_ids).squeeze(1)
                    sub_step_log_probs[:, step_idx] = sub_chosen_log_prob
                    if return_logits:
                        sub_step_logits[:, step_idx, :] = sub_logits
                    if step_idx < self.decode_len - 1:
                        sub_current_tgt_ids = torch.cat([sub_current_tgt_ids, sub_token_ids], dim=1)

                #
                generated_sequences_ids.index_copy_(0, expanded_rows, sub_generated_ids)
                #  (B*S, L)  log probs
                flat_step_log_probs = step_log_probs.view(batch_size * num_samples, self.decode_len)
                flat_step_log_probs.index_copy_(0, expanded_rows, sub_step_log_probs)
                #  logits (B*S, L, V)
                if return_logits:
                    vocab_size = len(self.decoder_vocab)
                    flat_step_logits = step_logits.view(batch_size * num_samples, self.decode_len, vocab_size)
                    flat_step_logits.index_copy_(0, expanded_rows, sub_step_logits)
                    #
                    step_logits = flat_step_logits.view(batch_size, num_samples, self.decode_len, vocab_size)
                #  log probs
                step_log_probs = flat_step_log_probs.view(batch_size, num_samples, self.decode_len)
                #
                final_decoded_ids = generated_sequences_ids.view(batch_size, num_samples, self.decode_len)

                for idx in range(violating_indices.shape[0]):
                    b = int(violating_indices[idx, 0].item())
                    s_idx = int(violating_indices[idx, 1].item())
                    output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                        final_decoded_ids[b, s_idx, :].tolist()
                    )

                output_floats_t = torch.tensor(output_floats, device=device, dtype=torch.float32)
                below = (output_floats_t < lower.unsqueeze(-1))
                above = (output_floats_t > upper.unsqueeze(-1))
                violating = (below | above).any(dim=-1)
                retries += 1

        return final_decoded_ids, output_floats, step_log_probs, step_logits

    def sample_with_logprobs_topp(
        self,
        inputs: dict[str, Tensor],
        num_samples: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        return_logits: bool = True,
    ) -> tuple[Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        tokenlog prob
        top-ptop-k


          - top_p: top-pptoken
          - top_k: top-kktoken0top-k
          - return_logits: logits


          - final_decoded_ids: (B, S, L_decode) token ids
          - output_floats: np.ndarray (B, S, max_num_objs) token ids
          - step_log_probs: (B, S, L_decode) tokenlog prob
          - step_logits: (B, S, L_decode, V) logits
        """
        # eval()/REINFORCElog prob
        encoder_input = inputs["encoder_input"]  # (B, L_src)
        device = encoder_input.device
        batch_size = encoder_input.shape[0]

        # Encode
        memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

        #  B*S batch
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
            self.decoder_vocab.bos_pad_id,
            dtype=torch.long,
            device=device,
        )

        # tokenlog prob
        generated_sequences_ids = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.long,
            device=device,
        )
        step_log_probs = torch.zeros(
            (batch_size * num_samples, self.decode_len),
            dtype=torch.float32,
            device=device,
        )
        if return_logits:
            step_logits = torch.zeros(
                (batch_size * num_samples, self.decode_len, len(self.decoder_vocab)),
                dtype=torch.float32,
                device=device,
            )

        for step_idx in range(self.decode_len):
            # (B*S, V)
            logits = self.encoder_decoder.next_token_logits(
                current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
            )

            # mask
            curr_mask = self.decoder_constraint_masks[step_idx, :].unsqueeze(
                0
            )  # (1, V)
            masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

            #
            scaled_logits = masked_logits / temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)  # (B*S, V)
            probs = torch.exp(log_probs)

            # top-k
            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                # top-k mask
                top_k_mask = torch.zeros_like(probs)
                top_k_mask.scatter_(-1, top_k_indices, 1.0)
                probs = probs * top_k_mask
                #
                probs = probs / probs.sum(dim=-1, keepdim=True)
                log_probs = torch.log(probs + 1e-8)  # log(0)

            # top-p
            if top_p < 1.0:
                #
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                #
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # top_ptoken0
                sorted_probs[sorted_indices_to_remove] = 0.0
                #
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                #
                probs = torch.zeros_like(probs)
                probs.scatter_(-1, sorted_indices, sorted_probs)
                log_probs = torch.log(probs + 1e-8)  # log(0)

            #
            token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)
            generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

            # tokenlog prob
            chosen_log_prob = torch.gather(log_probs, 1, token_ids).squeeze(1)  # (B*S)
            step_log_probs[:, step_idx] = chosen_log_prob

            # logitslogitsmask
            if return_logits:
                step_logits[:, step_idx, :] = logits

            #
            if step_idx < self.decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        #
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, self.decode_len
        )
        step_log_probs = step_log_probs.view(batch_size, num_samples, self.decode_len)
        if return_logits:
            step_logits = step_logits.view(batch_size, num_samples, self.decode_len, -1)
        else:
            step_logits = None

        # token ids
        output_floats = np.zeros(
            (batch_size, num_samples, self.max_num_objs), dtype=float
        )
        for b in range(batch_size):
            for s_idx in range(num_samples):
                output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                    final_decoded_ids[b, s_idx, :].tolist()
                )

        return final_decoded_ids, output_floats, step_log_probs, step_logits

    def log_prob(self, examples: dict[str, Tensor]) -> Tensor:
        self.encoder_decoder.eval()
        enc_input = examples["encoder_input"]
        dec_input = examples["decoder_input"]
        dec_target = examples["decoder_target"]

        logits = self.encoder_decoder.forward(enc_input, dec_input)
        log_probs = F.log_softmax(logits, dim=-1)

        true_log_probs = torch.gather(log_probs, dim=2, index=dec_target.unsqueeze(-1))

        pad_mask = dec_target != self.decoder_vocab.bos_pad_id
        true_log_probs_masked = true_log_probs.squeeze(-1) * pad_mask
        sequence_sum_log_probs = true_log_probs_masked.sum(dim=1)
        return sequence_sum_log_probs

    def convert_inputs(self, inputs: Sequence[core.ExampleInput]) -> dict[str, Tensor]:
        strings = [example.x for example in inputs]
        encoder_token_ids = [self.encoder_vocab.to_token_ids(s) for s in strings]
        encoder_input = [self._pad_or_truncate(t) for t in encoder_token_ids]
        return {"encoder_input": torch.tensor(encoder_input)}

    def convert_numeric_inputs(
        self, inputs: Sequence[core.ExampleInputNumeric]
    ) -> dict[str, Tensor]:
        # For MLPEncoder
        encoder_input = [example.x for example in inputs]
        return {"encoder_input": torch.tensor(encoder_input)}

    def convert_number_inputs(
        self, inputs: Sequence[core.ExampleInput]
    ) -> dict[str, Tensor]:
        def extract_and_replace_numbers(text: str) -> tuple[str, list[float]]:
            pattern = r"(x\d*:)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            numbers = []

            def replace_func(match):
                number = float(match.group(2))
                numbers.append(number)
                return f"{match.group(1)}<num>"

            processed_text = re.sub(pattern, replace_func, text)
            return processed_text, numbers

        #
        processed_data = [extract_and_replace_numbers(example.x) for example in inputs]
        strings = [text for text, _ in processed_data]
        numbers_list = [nums for _, nums in processed_data]
        encoder_token_ids = [self.encoder_vocab.to_token_ids(s) for s in strings]
        encoder_input = [self._pad_or_truncate(t) for t in encoder_token_ids]
        # tensor
        encoder_tensor = torch.tensor(encoder_input)

        # number_mask<num>0
        number_mask = torch.zeros_like(encoder_tensor, dtype=torch.float32)
        num_positions = encoder_tensor == self.encoder_vocab.num_token_id
        number_mask[num_positions] = torch.tensor(
            [n for nums in numbers_list for n in nums]
        )

        return {"encoder_input": encoder_tensor, "number_mask": number_mask}

    def convert_number_examples(
        self, examples: Sequence[core.Example]
    ) -> dict[str, Tensor]:
        y_values = [example.y for example in examples]
        y_tokens_list = [self.decoder_vocab.to_token_ids(y) for y in y_values]

        decoder_inputs = []
        decoder_targets = []
        pad_id = self.decoder_vocab.bos_pad_id

        for t in y_tokens_list:
            padding_needed = self.decode_len - len(t)
            # Input: [pad, t_1, ..., t_n, pad, ..., pad]
            decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
            # Target: [t_1, ..., t_n, pad, ..., pad]
            decoder_targets.append(t + [pad_id] * (padding_needed + 1))

        decoder_out = {
            "decoder_input": torch.tensor(decoder_inputs),
            "decoder_target": torch.tensor(decoder_targets),
            "y": torch.tensor(y_values),
        }
        return self.convert_number_inputs(examples) | decoder_out

    def convert_examples(self, examples: Sequence[core.Example]) -> dict[str, Tensor]:
        y_values = [example.y for example in examples]
        y_tokens_list = [self.decoder_vocab.to_token_ids(y) for y in y_values]

        decoder_inputs = []
        decoder_targets = []
        pad_id = self.decoder_vocab.bos_pad_id

        for t in y_tokens_list:
            padding_needed = self.decode_len - len(t)
            # Input: [pad, t_1, ..., t_n, pad, ..., pad]
            decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
            # Target: [t_1, ..., t_n, pad, ..., pad]
            decoder_targets.append(t + [pad_id] * (padding_needed + 1))

        decoder_out = {
            "decoder_input": torch.tensor(decoder_inputs),
            "decoder_target": torch.tensor(decoder_targets),
            "y": torch.tensor(y_values),
        }
        return self.convert_inputs(examples) | decoder_out

    def convert_RL_examples(
        self, examples: Sequence[core.ExampleRL]
    ) -> dict[str, Tensor]:
        y_values = [example.y for example in examples]
        y_medians = [example.y_median for example in examples]
        q1s = [example.q1 for example in examples]
        q3s = [example.q3 for example in examples]
        y_tokens_list = [self.decoder_vocab.to_token_ids(y) for y in y_values]

        decoder_inputs = []
        decoder_targets = []
        pad_id = self.decoder_vocab.bos_pad_id

        for t in y_tokens_list:
            padding_needed = self.decode_len - len(t)
            # Input: [pad, t_1, ..., t_n, pad, ..., pad]
            decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
            # Target: [t_1, ..., t_n, pad, ..., pad]
            decoder_targets.append(t + [pad_id] * (padding_needed + 1))

        decoder_out = {
            "decoder_input": torch.tensor(decoder_inputs),
            "decoder_target": torch.tensor(decoder_targets),
            "y": torch.tensor(y_values),
            "y_median": torch.tensor(y_medians),
            "q1": torch.tensor(q1s),
            "q3": torch.tensor(q3s),
        }
        return self.convert_inputs(examples) | decoder_out

    def convert_normalized_RL_examples(
        self, examples: Sequence[core.ExamplebyteRLNumeric]
    ) -> dict[str, Tensor]:
        y_values = [example.y for example in examples]
        y_max = [example.y_max for example in examples]
        y_min = [example.y_min for example in examples]
        y_tokens_list = [self.decoder_vocab.to_token_ids(y) for y in y_values]
        # print(y_values[0])
        y_values = [self.decoder_vocab.from_token_ids(t)[0] for t in y_tokens_list]
        # print(y_values[0])
        # assert 0
        decoder_inputs = []
        decoder_targets = []
        pad_id = self.decoder_vocab.bos_pad_id

        for t in y_tokens_list:
            padding_needed = self.decode_len - len(t)
            # Input: [pad, t_1, ..., t_n, pad, ..., pad]
            decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
            # Target: [t_1, ..., t_n, pad, ..., pad]
            decoder_targets.append(t + [pad_id] * (padding_needed + 1))

        decoder_out = {
            "decoder_input": torch.tensor(decoder_inputs),
            "decoder_target": torch.tensor(decoder_targets),
            "y": torch.tensor(y_values),
            "y_max": torch.tensor(y_max),
            "y_min": torch.tensor(y_min),
        }
        return self.convert_numeric_inputs(examples) | decoder_out

    def convert_normalized_RL_examples_test(
        self, examples: Sequence[core.ExamplebyteRLNumeric]
    ) -> dict[str, Tensor]:
        y_values = [example.y for example in examples]
        y_max = [example.y_max for example in examples]
        y_min = [example.y_min for example in examples]
        y_tokens_list = [self.decoder_vocab.to_token_ids(y) for y in y_values]
        decoder_inputs = []
        decoder_targets = []
        pad_id = self.decoder_vocab.bos_pad_id

        for t in y_tokens_list:
            padding_needed = self.decode_len - len(t)
            # Input: [pad, t_1, ..., t_n, pad, ..., pad]
            decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
            # Target: [t_1, ..., t_n, pad, ..., pad]
            decoder_targets.append(t + [pad_id] * (padding_needed + 1))

        decoder_out = {
            "decoder_input": torch.tensor(decoder_inputs),
            "decoder_target": torch.tensor(decoder_targets),
            "y": torch.tensor(y_values),
            "y_max": torch.tensor(y_max),
            "y_min": torch.tensor(y_min),
        }
        return self.convert_numeric_inputs(examples) | decoder_out

    def convert_numeric_examples(
        self, examples: Sequence[core.ExampleNumeric]
    ) -> dict[str, Tensor]:
        y_values = [example.y for example in examples]
        y_tokens_list = [self.decoder_vocab.to_token_ids(y) for y in y_values]

        decoder_inputs = []
        decoder_targets = []
        pad_id = self.decoder_vocab.bos_pad_id

        for t in y_tokens_list:
            padding_needed = self.decode_len - len(t)
            # Input: [pad, t_1, ..., t_n, pad, ..., pad]
            decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
            # Target: [t_1, ..., t_n, pad, ..., pad]
            decoder_targets.append(t + [pad_id] * (padding_needed + 1))

        decoder_out = {
            "decoder_input": torch.tensor(decoder_inputs),
            "decoder_target": torch.tensor(decoder_targets),
            "y": torch.tensor(y_values),
        }
        return self.convert_numeric_inputs(examples) | decoder_out

    def convert_numeric_RL_examples(
        self, examples: Sequence[core.ExampleRLNumeric]
    ) -> dict[str, Tensor]:
        y_values = [example.y for example in examples]
        y_medians = [example.y_median for example in examples]
        q1s = [example.q1 for example in examples]
        q3s = [example.q3 for example in examples]
        y_means = [example.y_mean for example in examples]
        y_stds = [example.y_std for example in examples]
        y_tokens_list = [self.decoder_vocab.to_token_ids(y) for y in y_values]

        decoder_inputs = []
        decoder_targets = []
        pad_id = self.decoder_vocab.bos_pad_id

        for t in y_tokens_list:
            padding_needed = self.decode_len - len(t)
            # Input: [pad, t_1, ..., t_n, pad, ..., pad]
            decoder_inputs.append([pad_id] + t + [pad_id] * padding_needed)
            # Target: [t_1, ..., t_n, pad, ..., pad]
            decoder_targets.append(t + [pad_id] * (padding_needed + 1))

        decoder_out = {
            "decoder_input": torch.tensor(decoder_inputs),
            "decoder_target": torch.tensor(decoder_targets),
            "y": torch.tensor(y_values),
            "y_median": torch.tensor(y_medians),
            "q1": torch.tensor(q1s),
            "q3": torch.tensor(q3s),
            "y_mean": torch.tensor(y_means),
            "y_std": torch.tensor(y_stds),
        }
        return self.convert_numeric_inputs(examples) | decoder_out

    def _pad_or_truncate(self, token_ids: list[int]) -> list[int]:
        encoder_pad_idx = self.encoder_vocab.pad_id
        if len(token_ids) > self.max_input_len:
            return token_ids[: self.max_input_len]
        return token_ids + [encoder_pad_idx] * (self.max_input_len - len(token_ids))

    def _create_decoder_constraint_masks(self) -> torch.Tensor:
        vocab_size = len(self.decoder_vocab)
        masks = np.zeros((self.decode_len, vocab_size), dtype=np.float32)
        for step_idx in range(self.decode_len):
            for allowed_token_id in self.decoder_vocab.token_ids_at_index(step_idx):
                masks[step_idx, allowed_token_id] = 1.0
        return torch.from_numpy(masks)


def _detect_overfitting(losses: Sequence[float]) -> bool:
    if len(losses) <= 1:
        return False
    return losses[-1] > losses[-2]


def _train_step(
    model: PyTorchModel,
    optimizer: optim.Optimizer,
    batch: dict[str, torch.Tensor],
):
    """Performs a single training step."""
    model.train()
    optimizer.zero_grad()
    loss, _ = model.compute_loss_and_metrics(batch)
    loss.backward()
    optimizer.step()


class PyTorchFineTuner(model_base.FineTuner):
    """PyTorch implementation of a local finetuner."""

    def __init__(self, model: PyTorchModel, optimizer: optim.Optimizer | None = None):
        self.model = model

        if optimizer is None:
            optimizer = optim.Adafactor(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4
            )
        self.optimizer = optimizer

    def fine_tune(
        self,
        examples: Sequence[core.Example],
        validation_examples: Sequence[core.Example] | None = None,
        max_epochs: int = 100,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        device = next(self.model.parameters()).device
        validation_examples = validation_examples or examples
        valid_batch = self.model.convert_examples(validation_examples)
        valid_batch = {k: v.to(device) for k, v in valid_batch.items()}

        batch_size = batch_size or len(examples)
        train_tensors = self.model.convert_examples(examples)
        rng = np.random.RandomState(seed)

        valid_losses = []
        state = self.model.state_dict()
        prev_state = state
        for _ in range(max_epochs):
            self.model.eval()  # Eval mode.
            val_loss, _ = self.model.compute_loss_and_metrics(valid_batch)
            valid_losses.append(val_loss.detach().item())

            if _detect_overfitting(valid_losses):
                state = prev_state
                break

            prev_state = state
            num_batches = math.ceil(len(examples) / batch_size)
            all_indices = rng.permutation(len(examples))
            for i in range(num_batches):
                inds = all_indices[i * batch_size : (i + 1) * batch_size]
                batch = {k: v[inds] for k, v in train_tensors.items()}
                batch = {k: v.to(device) for k, v in batch.items()}
                _train_step(self.model, self.optimizer, batch)
            state = self.model.state_dict()

        self.model.load_state_dict(state)
