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
from torch.nn.utils.rnn import pad_sequence

from typing import Optional, Tuple

class PooledCrossAttention(nn.Module):
    """
    Performs cross-attention where the query is a pooled representation of a sequence.
    Query: A sequence of shape (B, L, d_model) which will be pooled.
    Key/Value: A batch of sets of embeddings, given as a tuple of tensors.
    """
    def __init__(self, d_model: int, emb_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim * nhead != self.d_model:
            raise ValueError(f"d_model must be divisible by nhead")

        # 1.
        # Query  pooling  src d_model
        self.q_proj = nn.Linear(d_model, d_model)
        # Key  Value  dataset_embedding emb_dim
        self.k_proj = nn.Linear(emb_dim, d_model)
        self.v_proj = nn.Linear(emb_dim, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout_p = dropout

    def forward(self, x: torch.Tensor, dataset_embedding: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            x: The source sequence tensor from the encoder. Shape: (B, L, d_model).
            dataset_embedding: A tuple of tensors. Length is B, each tensor is (N_i, emb_dim).

        Returns:
            A single context vector per batch item, infused with dataset info. Shape: (B, 1, d_model).
        """
        B, L, D_model = x.shape

        # ---  1:  x  Pooling Query  ---
        #  mean pooling
        # (B, L, d_model) -> (B, d_model)
        query_vec = x.mean(dim=1)

        # ---  2:  K  V  Padding  Masking ---
        if len(dataset_embedding) != B:
            raise ValueError("Batch size of x and dataset_embedding must match.")

        #  dataset_embedding
        if not all(emb.numel() > 0 for emb in dataset_embedding):
             # embedding
            return torch.zeros(B, 1, self.d_model, device=x.device, dtype=x.dtype)

        #
        actual_emb_dim = dataset_embedding[0].shape[1]
        if actual_emb_dim != self.emb_dim:
            raise ValueError(f"Model expects emb_dim={self.emb_dim}, but got {actual_emb_dim}")

        lengths = torch.tensor([emb.shape[0] for emb in dataset_embedding], device=x.device)
        max_len_kv = int(lengths.max())

        # Padding a K/V source
        # (B, max_len_kv, emb_dim)
        kv_source = pad_sequence(list(dataset_embedding), batch_first=True, padding_value=0.0)

        # Create mask for the padded K/V source
        # (B, max_len_kv)
        mask_range = torch.arange(max_len_kv, device=x.device).expand(B, -1)
        kv_padding_mask = mask_range >= lengths.unsqueeze(1)

        # ---  3:  Cross-Attention ---
        # Project Q, K, V
        # q: (B, d_model) -> (B, d_model)
        q = self.q_proj(query_vec)
        # k, v: (B, max_len_kv, emb_dim) -> (B, max_len_kv, d_model)
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        # Reshape for multi-head attention
        # Query: (B, d_model) -> (B, 1, d_model) -> (B, nhead, 1, head_dim)
        q = q.view(B, 1, self.nhead, self.head_dim).transpose(1, 2)
        # Key/Value: (B, max_len_kv, d_model) -> (B, nhead, max_len_kv, head_dim)
        k = k.view(B, max_len_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, max_len_kv, self.nhead, self.head_dim).transpose(1, 2)

        # Attention mask needs to be broadcastable to (B, nhead, 1, max_len_kv)
        attn_mask = kv_padding_mask.view(B, 1, 1, max_len_kv)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout_p if self.training else 0.0
        )

        # Reshape output and apply final projection
        # (B, nhead, 1, head_dim) -> (B, 1, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.d_model)

        return self.out_proj(attn_output)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, d_model: int, max_len: int, base: int = 10000):
        super().__init__()
        # Note: d_model is the head dimension in our case.
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # freqs shape is (max_len, d_model / 2)
        # The cached sin/cos tables should have a feature dimension of d_model / 2
        self.register_buffer(
            "cos_cached", freqs.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", freqs.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]  # x has shape (B, n_heads, L, head_dim).
        # Return tensors of shape (1, 1, L, head_dim / 2)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to query and key tensors."""
    # q, k have shape (B, H, L, D_h)
    # cos, sin have shape (B, H, L, D_h / 2) after broadcasting

    # Reshape q and k to separate even and odd dimensions
    q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
    k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)
    q_even, q_odd = q_reshaped[..., 0], q_reshaped[..., 1]
    k_even, k_odd = k_reshaped[..., 0], k_reshaped[..., 1]
    # q_even, q_odd, k_even, k_odd have shape (B, H, L, D_h / 2)

    # Apply rotation. All tensors in this operation have a final dim of D_h / 2.
    q_out = torch.stack(
        [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], -1
    ).flatten(-2)
    k_out = torch.stack(
        [k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], -1
    ).flatten(-2)

    return q_out.type_as(q), k_out.type_as(k)


class RopeTransformerEncoderLayer(nn.Module):
    """A Transformer Encoder Layer with RoPE support."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim * nhead != self.d_model:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout_p = dropout  # For F.scaled_dot_product_attention

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def _sa_block(
        self,
        x: torch.Tensor,
        rotary_pos_emb: RotaryPositionalEmbedding,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Self-attention block with RoPE, manual projection, and correct masking."""
        B, L, _ = x.shape  # pylint: disable=invalid-name

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, L, D) -> (B, H, L, D_h)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        cos, sin = rotary_pos_emb(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Prepare attention mask
        final_attn_mask = (
            key_padding_mask.view(B, 1, 1, L) if key_padding_mask is not None else None
        )

        # Perform attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=final_attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn_output)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.ff_dropout(self.activation(self.linear1(x))))

    def forward(
        self,
        src: torch.Tensor,
        rotary_pos_emb: RotaryPositionalEmbedding,
        src_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Applies the encoder layer with the Pre-Norm structure."""
        x = src
        x = x + self.dropout1(
            self._sa_block(self.norm1(x), rotary_pos_emb, src_key_padding_mask)
        )
        x = x + self.dropout2(self._ff_block(self.norm2(x)))
        return x


class RopeTransformerEncoderLayer_with_dataset_embedding(nn.Module):
    """A Transformer Encoder Layer with RoPE support and dataset embedding cross attention."""

    def __init__(
        self,
        d_model: int,
        emb_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim * nhead != self.d_model:
            raise ValueError(f"d_model must be divisible by nhead")

        # Self-Attention Block
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.attn_dropout_p = dropout

        # Feed-Forward Block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # --- :  Cross-Attention  ---
        self.dataset_attention = PooledCrossAttention(d_model, emb_dim, nhead, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def _sa_block(
        self, x: torch.Tensor, rotary_pos_emb: RotaryPositionalEmbedding, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        cos, sin = rotary_pos_emb(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_mask = key_padding_mask.view(B, 1, 1, L) if key_padding_mask is not None else None
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout_p if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn_output)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.ff_dropout(self.activation(self.linear1(x))))

    def forward(
        self,
        src: torch.Tensor,
        rotary_pos_emb: RotaryPositionalEmbedding,
        src_key_padding_mask: Optional[torch.Tensor],
        dataset_embedding: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """Applies the encoder layer with Pre-Norm structure and optional dataset cross-attention."""
        x = src

        # 1.  Transformer  (Self-Attention + FFN)
        x = x + self.dropout1(self._sa_block(self.norm1(x), rotary_pos_emb, src_key_padding_mask))
        x = x + self.dropout2(self._ff_block(self.norm2(x)))

        # 2.  Cross-Attention
        if dataset_embedding is not None:
            #  LayerNorm
            norm_x = self.norm3(x)
            # , shape: (B, 1, d_model)
            context_vec = self.dataset_attention(norm_x, dataset_embedding)
            # --- :  token ---
            #  (broadcasting)  context_vec  x
            x = x + self.dropout3(context_vec)

        return x


class RopeEncoder(nn.Module):
    """A stack of RoPE-enabled Transformer Encoder Layers."""

    def __init__(
        self,
        encoder_layer: RopeTransformerEncoderLayer,
        num_layers: int,
        norm: nn.LayerNorm | None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, rotary_pos_emb, src_key_padding_mask):
        output = src
        for mod in self.layers:
            output = mod(
                output, rotary_pos_emb, src_key_padding_mask=src_key_padding_mask
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class PositionalEncoding(nn.Module):
    """Default positional encoding."""

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


SPD_BACKENDS = [
    nn.attention.SDPBackend.FLASH_ATTENTION,
    nn.attention.SDPBackend.MATH,
    nn.attention.SDPBackend.EFFICIENT_ATTENTION,
]


class EncoderDecoder(nn.Module):
    """Encoder-Decoder model in PyTorch."""

    def __init__(
        self,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        encoder_pad_idx: int,
        max_encoder_len: int,
        max_decoder_len: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        numberic: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder_pad_idx = encoder_pad_idx
        self.src_tok_emb = nn.Embedding(encoder_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(decoder_vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.numberic = numberic

        # RoPE Encoder for best length generalization.
        self.rotary_pos_emb = RotaryPositionalEmbedding(
            d_model // nhead, max_len=max_encoder_len
        )
        encoder_layer = RopeTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = RopeEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # We use a standard positional encoding and decoder.
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

    def mean_pooling(
        self, memory: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
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
        masked_memory = memory * valid_mask.unsqueeze(
            -1
        )  # (batch_size, seq_len, d_model)

        #
        valid_lengths = valid_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

        #
        valid_lengths = torch.clamp(valid_lengths, min=1)

        #
        pooled = masked_memory.sum(dim=1) / valid_lengths  # (batch_size, d_model)

        return pooled

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        number_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        src_padding_mask = src == self.encoder_pad_idx
        tgt_causal_mask = self._generate_causal_mask(tgt_input.size(1)).to(src.device)

        if number_mask is not None:
            # : v = sign(n) * log(1 + |n|)
            transformed_mask = torch.sign(number_mask) * torch.log1p(
                torch.abs(number_mask)
            )

            #
            d_model = self.src_tok_emb.embedding_dim
            positions = torch.arange(src.size(1), device=src.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=src.device)
                * (-math.log(10000.0) / d_model)
            )

            pe = torch.zeros(1, src.size(1), d_model, device=src.device)
            pe[0, :, 0::2] = torch.sin(positions * div_term)
            pe[0, :, 1::2] = torch.cos(positions * div_term)

            #
            # transformed_maskpe
            value_pe = transformed_mask.unsqueeze(-1) * pe

            # number_mask
            value_mask = (number_mask != 0).unsqueeze(-1)

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            src_embeddings = self.src_tok_emb(src)
            if number_mask is not None:
                # mask
                src_embeddings = src_embeddings + value_pe * value_mask
            # Encode with RoPE encoder
            memory = self.encoder(
                self.emb_dropout(src_embeddings),
                self.rotary_pos_emb,
                src_key_padding_mask=src_padding_mask,
            )

            # Decode with standard decoder
            decoder_output = self.decoder(
                tgt=self.decoder_positional_encoding(self.tgt_tok_emb(tgt_input)),
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_padding_mask,
            )
        return self.generator(decoder_output)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the source sequence using the RoPE encoder."""
        src_padding_mask = src == self.encoder_pad_idx
        src_emb = self.emb_dropout(self.src_tok_emb(src))
        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            memory = self.encoder(
                src_emb, self.rotary_pos_emb, src_key_padding_mask=src_padding_mask
            )
        return memory, src_padding_mask

    def next_token_logits(
        self,
        current_tgt_seq: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes one step using the standard decoder."""
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
        """Decodes one step using the standard decoder."""
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


class CustomEncoderDecoder(nn.Module):
    """Custom Encoder-Decoder model that accepts any custom encoder in PyTorch."""

    def __init__(
        self,
        custom_encoder: nn.Module,
        decoder_vocab_size: int,
        encoder_pad_idx: int,
        max_decoder_len: int,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        encoder_vocab_size: int = None,
        use_embedding: bool = False,
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
            encoder_vocab_size:  embedding
            use_embedding:  embedding
            numberic:
        """
        super().__init__()
        self.d_model = d_model
        self.encoder_pad_idx = encoder_pad_idx
        self.custom_encoder = custom_encoder
        self.use_embedding = use_embedding
        self.numberic = numberic
        self.plot = plot

        #  embedding embedding
        if use_embedding:
            assert 0
            if encoder_vocab_size is None:
                raise ValueError(
                    "encoder_vocab_size must be provided when use_embedding=True"
                )
            self.src_tok_emb = nn.Embedding(encoder_vocab_size, d_model)
        else:
            self.src_tok_emb = None

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

    def mean_pooling(
        self, memory: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
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
        masked_memory = memory * valid_mask.unsqueeze(
            -1
        )  # (batch_size, seq_len, d_model)

        #
        valid_lengths = valid_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

        #
        valid_lengths = torch.clamp(valid_lengths, min=1)

        #
        pooled = masked_memory.sum(dim=1) / valid_lengths  # (batch_size, d_model)

        return pooled

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        number_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        #  padding mask embedding
        if self.use_embedding:
            src_padding_mask = src == self.encoder_pad_idx
        else:
            src_padding_mask = None

        tgt_causal_mask = self._generate_causal_mask(tgt_input.size(1)).to(src.device)

        #
        if number_mask is not None and self.numberic:
            # : v = sign(n) * log(1 + |n|)
            transformed_mask = torch.sign(number_mask) * torch.log1p(
                torch.abs(number_mask)
            )

            #
            d_model = self.d_model
            positions = torch.arange(src.size(1), device=src.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=src.device)
                * (-math.log(10000.0) / d_model)
            )

            pe = torch.zeros(1, src.size(1), d_model, device=src.device)
            pe[0, :, 0::2] = torch.sin(positions * div_term)
            pe[0, :, 1::2] = torch.cos(positions * div_term)

            #
            value_pe = transformed_mask.unsqueeze(-1) * pe
            value_mask = (number_mask != 0).unsqueeze(-1)

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            #  encoder
            if self.use_embedding:
                src_embeddings = self.src_tok_emb(src)
                if number_mask is not None and self.numberic:
                    # mask
                    src_embeddings = src_embeddings + value_pe * value_mask
                src_embeddings = self.emb_dropout(src_embeddings)
            else:
                #
                src_embeddings = src

            if not self.plot:
                memory = self.custom_encoder(src_embeddings)
            else:
                memory = src_embeddings.to(dtype=torch.float32)
                memory = memory.unsqueeze(1)

            #  memory  (batch_size, seq_len, d_model)
            if memory.dim() == 2:
                #  encoder  (batch_size, d_model) (batch_size, 1, d_model)
                memory = memory.unsqueeze(1)
            elif memory.dim() != 3:
                raise ValueError(
                    f"Custom encoder output should be 2D or 3D, got {memory.dim()}D"
                )

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
        if self.use_embedding:
            src_padding_mask = src == self.encoder_pad_idx
            src_emb = self.emb_dropout(self.src_tok_emb(src))
        else:
            src_padding_mask = None
            src_emb = src

        memory = self.custom_encoder(src_emb)

        #  memory
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        elif memory.dim() != 3:
            raise ValueError(
                f"Custom encoder output should be 2D or 3D, got {memory.dim()}D"
            )

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

class CustomEncoderDecoder_critic(nn.Module):
    """Custom Encoder-Decoder model that accepts any custom encoder in PyTorch."""

    def __init__(
        self,
        custom_encoder: nn.Module,
        decoder_vocab_size: int,
        encoder_pad_idx: int,
        max_decoder_len: int,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        encoder_vocab_size: int = None,
        use_embedding: bool = False,
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
            encoder_vocab_size:  embedding
            use_embedding:  embedding
            numberic:
        """
        super().__init__()
        self.d_model = d_model
        self.encoder_pad_idx = encoder_pad_idx
        self.custom_encoder = custom_encoder
        self.use_embedding = use_embedding
        self.numberic = numberic
        self.plot = plot

        #  embedding embedding
        if use_embedding:
            assert 0
            if encoder_vocab_size is None:
                raise ValueError(
                    "encoder_vocab_size must be provided when use_embedding=True"
                )
            self.src_tok_emb = nn.Embedding(encoder_vocab_size, d_model)
        else:
            self.src_tok_emb = None

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

        self.generator = nn.Linear(d_model, 1)
        self.init_weights()

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def mean_pooling(
        self, memory: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
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
        masked_memory = memory * valid_mask.unsqueeze(
            -1
        )  # (batch_size, seq_len, d_model)

        #
        valid_lengths = valid_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

        #
        valid_lengths = torch.clamp(valid_lengths, min=1)

        #
        pooled = masked_memory.sum(dim=1) / valid_lengths  # (batch_size, d_model)

        return pooled

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        number_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        #  padding mask embedding
        if self.use_embedding:
            src_padding_mask = src == self.encoder_pad_idx
        else:
            src_padding_mask = None

        tgt_causal_mask = self._generate_causal_mask(tgt_input.size(1)).to(src.device)

        #
        if number_mask is not None and self.numberic:
            # : v = sign(n) * log(1 + |n|)
            transformed_mask = torch.sign(number_mask) * torch.log1p(
                torch.abs(number_mask)
            )

            #
            d_model = self.d_model
            positions = torch.arange(src.size(1), device=src.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=src.device)
                * (-math.log(10000.0) / d_model)
            )

            pe = torch.zeros(1, src.size(1), d_model, device=src.device)
            pe[0, :, 0::2] = torch.sin(positions * div_term)
            pe[0, :, 1::2] = torch.cos(positions * div_term)

            #
            value_pe = transformed_mask.unsqueeze(-1) * pe
            value_mask = (number_mask != 0).unsqueeze(-1)

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            #  encoder
            if self.use_embedding:
                src_embeddings = self.src_tok_emb(src)
                if number_mask is not None and self.numberic:
                    # mask
                    src_embeddings = src_embeddings + value_pe * value_mask
                src_embeddings = self.emb_dropout(src_embeddings)
            else:
                #
                src_embeddings = src

            if not self.plot:
                memory = self.custom_encoder(src_embeddings)
            else:
                memory = src_embeddings.to(dtype=torch.float32)
                memory = memory.unsqueeze(1)

            #  memory  (batch_size, seq_len, d_model)
            if memory.dim() == 2:
                #  encoder  (batch_size, d_model) (batch_size, 1, d_model)
                memory = memory.unsqueeze(1)
            elif memory.dim() != 3:
                raise ValueError(
                    f"Custom encoder output should be 2D or 3D, got {memory.dim()}D"
                )

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
        if self.use_embedding:
            src_padding_mask = src == self.encoder_pad_idx
            src_emb = self.emb_dropout(self.src_tok_emb(src))
        else:
            src_padding_mask = None
            src_emb = src

        memory = self.custom_encoder(src_emb)

        #  memory
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        elif memory.dim() != 3:
            raise ValueError(
                f"Custom encoder output should be 2D or 3D, got {memory.dim()}D"
            )

        return memory, src_padding_mask

    def init_weights(self, init_encoder: bool = False):
        """
         Critic

        Args:
            init_encoder:  custom_encoder
                           encoder  Actor  False
                           encoder  Critic  True
        """
        # 1.  Embedding
        # Embedding std
        # nn.init.normal_(self.tgt_tok_emb.weight, mean=0, std=0.1)
        # if self.use_embedding and self.src_tok_emb is not None:
        #     nn.init.normal_(self.src_tok_emb.weight, mean=0, std=0.1)

        # 2.  Generator ()
        #  Critic Value 0
        # gain=0.01
        nn.init.orthogonal_(self.generator.weight, gain=0.01)
        nn.init.constant_(self.generator.bias, 0.0)

        # 3.  Decoder (Transformer )
        #  Transformer
        def _init_transformer_submodules(module):
            if isinstance(module, nn.Linear):
                # Transformer  ReLU  GELUgain  sqrt(2)
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm weight=1, bias=0
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

        #  decoder
        self.decoder.apply(_init_transformer_submodules)

        # 4.  Custom Encoder ()
        # if init_encoder:
        #     #  encoder  Linear/Conv/LayerNorm
        #     self.custom_encoder.apply(_init_transformer_submodules)
        #     print("Critic: Custom Encoder initialized.")
        # else:
        #     print("Critic: Custom Encoder skipped (assuming shared or pre-trained).")

        print("Critic: Decoder, Embeddings, and Generator initialized.")

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

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
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


class DatasetEncoderDecoder(nn.Module):
    """Custom Encoder-Decoder model that accepts any custom encoder in PyTorch."""

    def __init__(
        self,
        embedding_dim: int,
        decoder_vocab_size: int,
        encoder_pad_idx: int,
        max_encoder_len: int,
        max_decoder_len: int,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        encoder_vocab_size: int = None,
        **kwargs,
    ):
        """
        Args:
            dataset_encoder:  encoder MLPCNN  nn.Module
            decoder_vocab_size:
            encoder_pad_idx:
            max_decoder_len:
            d_model:
            nhead:
            num_decoder_layers:
            dim_feedforward:
            dropout: dropout
            encoder_vocab_size:  embedding
            use_embedding:  embedding
            numberic:
        """
        super().__init__()
        self.d_model = d_model
        self.encoder_pad_idx = encoder_pad_idx
        self.src_tok_emb = nn.Embedding(encoder_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(decoder_vocab_size, d_model)
        self.encoder = RopeTransformerEncoderLayer_with_dataset_embedding(
            d_model, embedding_dim, nhead, dim_feedforward, dropout
        )
        self.rotary_pos_emb = RotaryPositionalEmbedding(
            d_model // nhead, max_len=max_encoder_len
        )

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

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        dataset_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        src_padding_mask = src == self.encoder_pad_idx

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            memory = self.encoder(
                self.src_tok_emb(src),
                self.rotary_pos_emb,
                src_key_padding_mask=src_padding_mask,
                dataset_embedding=dataset_embedding,
            )
            decoder_output = self.decoder(
                tgt=self.decoder_positional_encoding(self.tgt_tok_emb(tgt_input)),
                memory=memory,
                tgt_mask=self._generate_causal_mask(tgt_input.size(1)).to(src.device),
                memory_key_padding_mask=src_padding_mask,
            )
        return self.generator(decoder_output)

    def encode(self, src: torch.Tensor, dataset_embedding: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the source sequence."""
        src_padding_mask = src == self.encoder_pad_idx
        src_emb = self.src_tok_emb(src)
        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            memory = self.encoder(
                self.src_tok_emb(src),
                self.rotary_pos_emb,
                src_key_padding_mask=src_padding_mask,
                dataset_embedding=dataset_embedding,
            )
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
