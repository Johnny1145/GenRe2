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

"""Default PyTorch architecture for a RegressLM."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .architecture import PositionalEncoding, RopeTransformerEncoderLayer_with_dataset_embedding

SPD_BACKENDS = [
    nn.attention.SDPBackend.FLASH_ATTENTION,
    nn.attention.SDPBackend.MATH,
    nn.attention.SDPBackend.EFFICIENT_ATTENTION,
]

class MLPEncoderDecoder(nn.Module):
    """MLP Encoder-Decoder model that accepts any custom encoder in PyTorch."""

    def __init__(
        self,
        mlp_encoder: nn.Module,
        decoder_vocab_size: int,
        encoder_pad_idx: int,
        max_decoder_len: int,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        numberic: bool = False,
        plot: bool = False,
        **kwargs,
    ):
        """
        Args:
            custom_encoder:  encoder MLPCNN  nn.Module
            decoder_vocab_size:
            encoder_pad_idx:
            max_decoder_len:
            d_model:
            nhead:
            num_decoder_layers:
            dim_feedforward:
            dropout: dropout
            numberic:
        """
        super().__init__()
        self.d_model = d_model
        self.encoder_pad_idx = encoder_pad_idx
        self.mlp_encoder = mlp_encoder
        self.numberic = numberic
        self.plot = plot

        self.tgt_tok_emb = nn.Embedding(decoder_vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        #
        self.decoder_positional_encoding = PositionalEncoding(
            d_model,
            max_len=max_decoder_len,
            dropout=dropout,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.generator = nn.Linear(d_model, decoder_vocab_size)

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def mean_pooling(self, memory: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        embeddermean poolingpadding

        Args:
            memory:  (batch_size, seq_len, d_model)
            padding_mask: padding (batch_size, seq_len)Truepadding

        Returns:
            mean pooling (batch_size, d_model)
        """
        if padding_mask is None:
            # padding_mask
            return torch.mean(memory, dim=1)

        # maskTruepadding
        valid_mask = ~padding_mask  # (batch_size, seq_len)

        # paddingembedding0
        masked_memory = memory * valid_mask.unsqueeze(-1)  # (batch_size, seq_len, d_model)

        #
        valid_lengths = valid_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

        #
        valid_lengths = torch.clamp(valid_lengths, min=1)

        #
        pooled = masked_memory.sum(dim=1) / valid_lengths  # (batch_size, d_model)

        return pooled

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor, number_mask: torch.Tensor=None) -> torch.Tensor:
        #  padding mask embedding
        src_padding_mask = src == self.encoder_pad_idx

        tgt_causal_mask = self._generate_causal_mask(tgt_input.size(1)).to(src.device)

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            #  MLP encoder
            src_embeddings = src

            if not self.plot:
                memory = self.mlp_encoder(src_embeddings)
            else:
                memory = src_embeddings.to(dtype=torch.float32)
                memory = memory.unsqueeze(1)

            #  memory  (batch_size, seq_len, d_model)
            if memory.dim() == 2:
                #  encoder  (batch_size, d_model) (batch_size, 1, d_model)
                memory = memory.unsqueeze(1)
            elif memory.dim() != 3:
                raise ValueError(f"Custom encoder output should be 2D or 3D, got {memory.dim()}D")

            #
            decoder_output = self.decoder(
                tgt=self.decoder_positional_encoding(self.tgt_tok_emb(tgt_input)),
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_padding_mask,
            )
        return self.generator(decoder_output)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ encoder """
        src_padding_mask = None
        src_emb = src

        memory = self.mlp_encoder(src_emb)

        #  memory
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        elif memory.dim() != 3:
            raise ValueError(f"Custom encoder output should be 2D or 3D, got {memory.dim()}D")

        return memory, src_padding_mask

    def next_token_logits(
        self,
        current_tgt_seq: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """"""
        tgt_causal_mask = self._generate_causal_mask(current_tgt_seq.size(1)).to(
            current_tgt_seq.device
        )
        tgt = self.decoder_positional_encoding(self.tgt_tok_emb(current_tgt_seq))

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            decoder_output_all_steps = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.generator(decoder_output_all_steps[:, -1, :])

    def plot_next_token_logits(
        self,
        current_tgt_seq: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """"""
        tgt_causal_mask = self._generate_causal_mask(current_tgt_seq.size(1)).to(
            current_tgt_seq.device
        )
        tgt = self.decoder_positional_encoding(self.tgt_tok_emb(current_tgt_seq))

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            decoder_output_all_steps = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # embeddinggenerator
        last_embedding = decoder_output_all_steps[:, -1, :]
        logits = self.generator(last_embedding)

        if return_embedding:
            return logits, last_embedding
        else:
            return logits


# MLP Encoder
class MLPEncoder(nn.Module):
    """ MLP """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        x = x.to(dtype=next(self.parameters()).dtype)

        #  3D (batch_size, seq_len, features) 2D
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x = x.view(batch_size * seq_len, features)
            output = self.mlp(x)
            #  (batch_size, seq_len, output_dim)
            return output.view(batch_size, seq_len, -1)
        else:
            return self.mlp(x).unsqueeze(1)
