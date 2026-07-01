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
from .model import PyTorchModel

NEG_INF = -1.0e7

# Dict Keys: "encoder_input", "decoder_input", "decoder_target"
Tensor = torch.Tensor

max_len = -1
all_lengths = []


class MLPEncoderModel(PyTorchModel, model_base.Model[Tensor]):
    """PyTorch implementation of a RegressLM with MLP encoder."""

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
        super().__init__(
            encoder_vocab=encoder_vocab,
            decoder_vocab=decoder_vocab,
            max_input_len=max_input_len,
            max_num_objs=max_num_objs,
            learning_rate=learning_rate,
            z_loss_coef=z_loss_coef,
            if_ntl=if_ntl,
            encoder_type=encoder_type,
            plot=plot,
            **architecture_kwargs,
        )

        self.encoder_decoder = architecture.CustomEncoderDecoder(
            custom_encoder=architecture.MLPEncoder(**architecture_kwargs),
            decoder_vocab_size=len(self.decoder_vocab),
            encoder_pad_idx=self.encoder_vocab.pad_id,
            max_decoder_len=self.decode_len + 1,
            plot=self.plot,
            **architecture_kwargs,
        )

        # Pre-compute the constraint masks for the decoder.
        self.register_buffer(
            "decoder_constraint_masks", self._create_decoder_constraint_masks()
        )

    @property
    def decode_len(self) -> int:
        return self.max_num_objs * self.decoder_vocab.num_tokens_per_obj

    def compute_loss_and_metrics(
        self,
        examples: dict[str, Tensor],
        regression_aware_loss_fn=None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """"""
        regression_aware_loss, regression_aware_metrics = regression_aware_loss_fn(self, examples)

        #
        metrics = {**regression_aware_metrics}
        metrics["total_loss"] = regression_aware_loss.detach()

        return regression_aware_loss, metrics


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
                current_tgt_ids, expanded_memory, memory_key_padding_mask
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
                current_tgt_ids, expanded_memory, memory_key_padding_mask
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

    def convert_examples(
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
        return self.convert_inputs(examples) | decoder_out

    def convert_inputs(
        self, inputs: Sequence[core.ExampleInputNumeric]
    ) -> dict[str, Tensor]:
        # For MLPEncoder
        encoder_input = [example.x for example in inputs]
        return {"encoder_input": torch.tensor(encoder_input)}
