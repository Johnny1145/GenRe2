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
        if encoder_type == "vanilla":
            self.encoder_decoder = architecture.EncoderDecoder(
                encoder_vocab_size=len(self.encoder_vocab),
                decoder_vocab_size=len(self.decoder_vocab),
                encoder_pad_idx=self.encoder_vocab.pad_id,
                max_encoder_len=self.max_input_len,
                max_decoder_len=self.decode_len + 1,
                **architecture_kwargs,
            )
        elif encoder_type == "mlp":
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
        if NTL is not None:
            ntl_loss = NTL.forward(logits, targets)
            metrics["ntl_loss"] = ntl_loss.detach()
            loss += 0.3 * ntl_loss
            # assert False

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
        reinforce_weight: float = 0.1,
        loss_balance: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """REINFORCE"""
        reinforce_loss, reinforce_metrics = reinforce_loss_fn(self, examples)

        #
        metrics = {**reinforce_metrics}
        metrics["total_loss"] = reinforce_loss.detach()
        metrics["reinforce_weight"] = reinforce_weight

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
        expanded_memory_key_padding_mask = (
            memory_key_padding_mask.unsqueeze(1)
            .expand(-1, num_samples, -1)
            .reshape(batch_size * num_samples, memory_key_padding_mask.size(1))
        )

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

    def sample_with_logprobs(
        self,
        inputs: dict[str, Tensor],
        num_samples: int,
        temperature: float = 1.0,
    ) -> tuple[Tensor, np.ndarray, torch.Tensor]:
        """
        tokenlog prob


          - final_decoded_ids: (B, S, L_decode) token ids
          - output_floats: np.ndarray (B, S, max_num_objs) token ids
          - step_log_probs: (B, S, L_decode) tokenlog prob
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

            #
            if step_idx < self.decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        #
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, self.decode_len
        )
        step_log_probs = step_log_probs.view(batch_size, num_samples, self.decode_len)

        # token ids
        output_floats = np.zeros(
            (batch_size, num_samples, self.max_num_objs), dtype=float
        )
        for b in range(batch_size):
            for s_idx in range(num_samples):
                output_floats[b, s_idx, :] = self.decoder_vocab.from_token_ids(
                    final_decoded_ids[b, s_idx, :].tolist()
                )

        return final_decoded_ids, output_floats, step_log_probs

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
        # print(strings[0])
        # print(numbers_list[0])
        # print(inputs[0].x)
        # assert 0

        # token ids
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
        return self.convert_inputs(examples) | decoder_out

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
