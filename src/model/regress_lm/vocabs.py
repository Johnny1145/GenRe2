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

"""Custom vocab classes for RegressLM."""

import abc
import pathlib
from typing import Generic, Sequence, TypeVar

import sentencepiece as spp
import sentencepiece as spt
import tokenizers as ht

from src.model.regress_lm import tokenizers

ObjectT = TypeVar("ObjectT")


class BaseVocab(abc.ABC, Generic[ObjectT]):
    """Base class for vocabularies."""

    @abc.abstractmethod
    def to_token_ids(self, obj: ObjectT, /) -> list[int]:
        """Converts object (e.g. text) to token ids."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the vocab size."""


class EncoderVocab(BaseVocab[ObjectT]):
    """Vocabulary class for encoders.

    Note we don't ever need to convert back to text.
    """

    @property
    @abc.abstractmethod
    def pad_id(self) -> int:
        """Returns the pad id."""


class DecoderVocab(BaseVocab[ObjectT]):
    """Vocabulary class for decoders.

    Supports single objective and multi-objective cases.

    For multi-objective, the output is simply the concatenation of tokens for each
    objective.
    """

    # TODO: Do we need multi-objective separator tokens (via BOS)?

    def __init__(self, tokenizer: tokenizers.DecoderTokenizer[ObjectT]):
        self.tokenizer = tokenizer

        self.itos = ["<pad>"] + sorted(self.tokenizer.all_tokens())
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def to_token_ids(self, obj: ObjectT | Sequence[ObjectT], /) -> list[int]:
        obj = obj if isinstance(obj, Sequence) else [obj]
        all_tokens = []
        for o in obj:
            all_tokens.extend(self.tokenizer.to_tokens(o))
        return [self.stoi[t] for t in all_tokens]

    def from_token_ids(self, token_ids: Sequence[int], /) -> list[ObjectT]:
        """Converts token ids to object."""
        # padding tokentoken
        # print(token_ids)
        token_strs = [self.itos[id] for id in token_ids if id != self.bos_pad_id and id < len(self.itos)]
        # print(token_strs)
        # # print(len(token_strs), self.num_tokens_per_obj)
        # assert 0

        # tokennum_tokens_per_obj
        if len(token_strs) % self.num_tokens_per_obj != 0:
            #
            num_complete_objs = len(token_strs) // self.num_tokens_per_obj
            #
            token_strs = token_strs[:num_complete_objs * self.num_tokens_per_obj]

            # token
            if not token_strs:
                return [0.0]  #

        decoded_objs = []
        for i in range(0, len(token_strs), self.num_tokens_per_obj):
            chunk = token_strs[i : i + self.num_tokens_per_obj]
            # try:
            decoded_objs.append(self.tokenizer.from_tokens(chunk))
            # except Exception as e:
            #     #
            #     print(f"Warning: Failed to decode tokens {chunk}: {e}, using default value 0.0")
            #     decoded_objs.append(0.0)

        return decoded_objs

    def token_ids_at_index(self, index: int) -> list[int]:
        """Returns the token ids for the given index."""
        index = index % self.num_tokens_per_obj
        return [self.stoi[t] for t in self.tokenizer.tokens_at_index(index)]

    @property
    def bos_pad_id(self) -> int:
        """Returns the BOS / PAD id for the decoder."""
        return self.stoi["<pad>"]

    @property
    def num_tokens_per_obj(self) -> int:
        """Returns the number of tokens used to represent each object."""
        return self.tokenizer.num_tokens_per_obj

    def __len__(self) -> int:
        """Returns the vocab size."""
        return len(self.stoi)


class BasicEnglishVocab(EncoderVocab[str]):
    """Basic English vocab for testing."""

    def __init__(self, words: list[str]):
        specials = ["<pad>", "<unk>"]
        # Build vocab dictionary ensuring special tokens have fixed IDs 0 and 1.
        vocab = {word: i + len(specials) for i, word in enumerate(words)}
        for i, token in enumerate(specials):
            vocab[token] = i

        # Instantiate a huggingface tokenizer with a WordLevel model
        self.tokenizer = ht.Tokenizer(
            ht.models.WordLevel(vocab=vocab, unk_token="<unk>")
        )
        self.tokenizer.normalizer = ht.normalizers.Lowercase()
        self.tokenizer.pre_tokenizer = ht.pre_tokenizers.Whitespace()

        pad_id_val = self.tokenizer.token_to_id("<pad>")
        if pad_id_val is None:
            raise ValueError("'<pad>' token not found in the vocabulary.")
        self._pad_id = pad_id_val

    def to_token_ids(self, obj: str) -> list[int]:
        return self.tokenizer.encode(obj).ids

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()


class StructuredTextVocab(EncoderVocab[str]):
    """For structured text, ideal for custom formats like JSON or DSLs.

    NOTE: Not working right now, pre_tokenizer is being completely ignored.
    """

    def __init__(self, tokens: list[str], split_regex: str = r"([\{\}\[\]:,])"):
        specials = ["<pad>", "<unk>"]

        self.vocab = {token: i + len(specials) for i, token in enumerate(tokens)}
        self.vocab.update({special: i for i, special in enumerate(specials)})

        self.tokenizer = ht.Tokenizer(
            ht.models.WordLevel(vocab=self.vocab, unk_token="<unk>")
        )
        pre_tokenizer = ht.pre_tokenizers.Split(
            pattern=split_regex, behavior="isolated"
        )
        self.tokenizer.pre_tokenizer = pre_tokenizer

    def to_token_ids(self, obj: str) -> list[int]:
        """Converts a structured string to a list of token IDs."""
        return self.tokenizer.encode(obj).ids

    @property
    def pad_id(self) -> int:
        """Returns the pad id."""
        return self.vocab["<pad>"]

    def __len__(self) -> int:
        """Returns the total vocabulary size."""
        return self.tokenizer.get_vocab_size()


class SentencePieceVocab(EncoderVocab[str]):
    """SentencePiece vocab."""

    T5_FILE = "./vocabs/sentencepiece.model"

    def __init__(self, file_path: str):
        """Initializes SentencePieceVocab by loading a pre-trained .model file."""
        self.sp_processor = spp.SentencePieceProcessor()

        if file_path.startswith("gs://"):  # Check Google Cloud Storage path.
            import os

            import gcsfs

            local_path = f"/tmp/{os.path.basename(file_path)}"
            gcsfs.GCSFileSystem(token="anon").get(file_path, local_path)
            file_path = local_path

        self.sp_processor.Load(file_path)

        if self.sp_processor.pad_id() == -1:
            raise ValueError(
                f"SentencePiece model '{file_path}' does not have a PAD token"
                " explicitly defined."
            )

    def to_token_ids(self, obj: str, /) -> list[int]:
        """Converts text to a list of token ids using the SentencePiece model."""
        return self.sp_processor.EncodeAsIds(obj)

    @property
    def pad_id(self) -> int:
        """Returns the pad id defined in the SentencePiece model."""
        return self.sp_processor.pad_id()

    def __len__(self) -> int:
        """Returns the total vocabulary size."""
        return self.sp_processor.GetPieceSize()

    @classmethod
    def from_t5(cls) -> "SentencePieceVocab":
        return cls(cls.T5_FILE)

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | pathlib.Path,
        vocab_size: int = 8192,
        model_prefix: str | pathlib.Path | None = None,
        sentencepiece_trainer_kwargs: dict[str, str] | None = None,
    ) -> "SentencePieceVocab":
        """Trains a SentencePiece vocab from the given corpus."""
        if model_prefix is None:
            model_prefix = pathlib.Path("/tmp/trained_sentencepiece")

        trainer_args = {
            "input": str(corpus_path),
            "model_prefix": str(model_prefix),
            "vocab_size": str(vocab_size),
            "model_type": "bpe",
            "pad_id": "0",
            "unk_id": "1",
            "pad_piece": "<pad>",
            "unk_piece": "<unk>",
            "bos_id": "-1",
            "eos_id": "-1",
            "hard_vocab_limit": "false",
            "byte_fallback": "true",
            "split_by_number": "false",
            "split_by_unicode_script": "false",
            "character_coverage": "1.0",
            "input_sentence_size": "0",
            "max_sentence_length": "500000",
            "shuffle_input_sentence": "false",
            "num_threads": "1",
        }
        if sentencepiece_trainer_kwargs:
            trainer_args.update(sentencepiece_trainer_kwargs)
        cmd = " ".join(f"--{k}={v}" for k, v in trainer_args.items())
        spt.SentencePieceTrainer.Train(cmd)
        return cls(str(model_prefix) + ".model")


class ExtendedSentencePieceVocab(SentencePieceVocab):
    """Extended SentencePiece vocab with additional <num> token."""

    def __init__(self, file_path: str):
        super().__init__(file_path)

        # <num> token
        self.num_token = "<num>"
        self.num_token_id = len(self.sp_processor)  # tokenID

    def to_token_ids(self, obj: str, /) -> list[int]:
        """Converts text to token ids, handling the <num> token specially."""
        # <num>
        if "<num>" in obj:
            tokens = []
            # <num>
            parts = obj.split("<num>")
            for i, part in enumerate(parts):
                if part:  #
                    tokens.extend(self.sp_processor.EncodeAsIds(part))
                if i < len(parts) - 1:  # <num> token id
                    tokens.append(self.num_token_id)
            return tokens
        else:
            return self.sp_processor.EncodeAsIds(obj)

    def __len__(self) -> int:
        """Returns the total vocabulary size including the additional token."""
        return self.sp_processor.GetPieceSize() + 1  # +1 for <num> token

    @classmethod
    def from_t5(cls) -> "ExtendedSentencePieceVocab":
        return cls(cls.T5_FILE)
