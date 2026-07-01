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

"""Custom tokenizers for RegressLM.

For full generality, note that the tokens are strings, as we assume the
vocabulary will assign integer IDs.
"""

import abc
import math
import re
from typing import Generic, TypeVar

import numpy as np
import ordered_set

DELIMITERS = ("<", ">")


def _to_token(s: str | int) -> str:
    left_d, right_d = DELIMITERS
    return f"{left_d}{s}{right_d}"


def _from_token(token: str) -> str | int:
    left_d, right_d = DELIMITERS
    pattern = f"{left_d}(.*?){right_d}"
    m = re.fullmatch(pattern, token)
    if not m:
        raise ValueError(f"Could not deserialize `{token}`.")
    return m.group(1)


ObjectT = TypeVar("ObjectT")


class DecoderTokenizer(abc.ABC, Generic[ObjectT]):
    """Abstract class for decoder tokenizers."""

    @property
    @abc.abstractmethod
    def num_tokens_per_obj(self) -> int:
        """Number of tokens used to represent each float."""

    def all_tokens(self) -> ordered_set.OrderedSet[str]:
        """Returns ordered set of all tokens used."""
        out = []
        for i in range(self.num_tokens_per_obj):
            out.extend(self.tokens_at_index(i))
        return ordered_set.OrderedSet(out)

    @abc.abstractmethod
    def tokens_at_index(self, index: int) -> ordered_set.OrderedSet[str]:
        """Returns ordered set of tokens possible at position `index`."""

    @abc.abstractmethod
    def to_tokens(self, obj: ObjectT, /) -> list[str]:
        """Converts an object to a string of tokens."""

    @abc.abstractmethod
    def from_tokens(self, tokens: list[str], /) -> ObjectT:
        """Converts a string of tokens to an object."""



class P10Tokenizer(DecoderTokenizer[float]):
    """Uses P10 tokenization from https://arxiv.org/abs/2112.01898.

    A float f can be represented as:

    `s * m * 10^e`

    where:
      s: Positive/Negative sign (+, -)
      m: Mantissa representing leading digits.
      e: Exponent.

    Attributes:
      num_digits: Number of digits in `m`. Each digit (even the leading) is
        between <0> and <9>.
      exponent_range: Controls number of exponent tokens, e.g. if 10, the exponent
        token range will be [<E-10>, <E10>], affecting the range of representable
        floats.
    """

    def __init__(self, num_digits: int = 4, exponent_range: int = 10):
        self.num_digits = num_digits
        self.exponent_range = exponent_range

    @property
    def num_tokens_per_obj(self) -> int:
        return 2 + self.num_digits

    def tokens_at_index(self, index: int) -> ordered_set.OrderedSet[str]:
        if index < 0 or index >= self.num_tokens_per_obj:
            raise ValueError(f"Index {index} out of bounds.")

        if index == 0:  # beginning
            tokens = [_to_token(s) for s in ["+", "-"]]
        elif index == self.num_tokens_per_obj - 1:  # end
            exps = [
                f"E{i}" for i in range(-self.exponent_range, self.exponent_range + 1)
            ]
            tokens = [_to_token(s) for s in exps]
        else:  # middle (digit)
            tokens = [_to_token(s) for s in range(0, 10)]
        return ordered_set.OrderedSet(tokens)

    @property
    def _max_abs_val(self) -> float:
        """Largest representable positive number."""
        return float(self.num_digits * "9") * (10.0**self.exponent_range)

    @property
    def _min_abs_val(self) -> float:
        """Smallest representable positive number."""
        min_mantissa = float("1" + (self.num_digits - 1) * "0")
        return min_mantissa * (10 ** (-self.exponent_range))

    def _round_float(self, f: float) -> float:
        """Rounds float to the closest in-range value."""
        abs_f = abs(f)
        abs_f = min(abs_f, self._max_abs_val)
        if abs_f < self._min_abs_val:
            # Decides whether to move to 0.0 or `min_abs_val`.
            zero_or_min = round(abs_f / self._min_abs_val)
            abs_f = self._min_abs_val * zero_or_min
        return abs_f if f >= 0 else -abs_f

    def to_tokens(self, f: float, /) -> list[str]:
        f = self._round_float(f)
        s = np.format_float_scientific(
            f,
            precision=self.num_digits - 1,
            min_digits=self.num_digits - 1,
            sign=True,
        )
        # We expect numpy to produce scientific notation of the form `+2.123e+4`.
        # It will round for us and ensure leading digit isn't zero, unless the
        # number is zero.
        m = re.fullmatch("([+-])([0-9.]*)e(.*)", s)
        if not m:
            raise RuntimeError(f"Unexpected numpy notation: {s}")
        sign: str = m.group(1)
        digits = list(m.group(2).replace(".", ""))
        exp = int(m.group(3)) - len(digits) + 1 if f else 0

        out = [sign] + digits + [f"E{exp}"]
        return [_to_token(s) for s in out]

    def from_tokens(self, tokens: list[str], /) -> float:
        primitives = [_from_token(t) for t in tokens]

        sign = -1 if primitives[0] == "-" else 1
        mantissa = int("".join(map(str, primitives[1:-1])))
        exp = int("".join(primitives[-1]).lstrip("E"))

        return float(sign * mantissa * 10**exp)

    def get_num_tokens(self) -> list[str]:
        num_tokens = []
        for i in range(10):
            num_tokens.append(_to_token(i))
        return num_tokens

    def token_to_number(self, token: str) -> float:
        return int(token[1:-1])

    # Added for NTL: expose exponent tokens and parsing for the last position
    def get_exponent_tokens(self) -> list[str]:
        exps = [
            f"E{i}" for i in range(-self.exponent_range, self.exponent_range + 1)
        ]
        return [_to_token(s) for s in exps]

    def token_to_exponent(self, token: str) -> int:
        # token format is like "<E-3>"
        inner = token[1:-1]
        if not inner.startswith("E"):
            raise ValueError(f"Not an exponent token: {token}")
        return int(inner[1:])


class IEEEFloatTokenizer(DecoderTokenizer[float]):
    """More official float tokenizer, minimizing the use of dedicated tokens.

    Follows IEEE-type standard.

    A float f = `s * b^e * m` can be represented as [s, e, m] from most to least
    important, where:
      s: Positive/Negative sign (+, -)
      b: Base
      e: Exponent (left-most is a sign, digits represented with base b)
      m: Mantissa (represented with base b)

    For example, 1.23456789e-222 can be represented as:

    <+><-><2><2><2><1><2><3><4>

    if b=10, num_exponent_digits=3, and num_mantissa_digits=4.
    """

    def __init__(
        self,
        base: int = 10,
        num_exponent_digits: int = 1,
        num_mantissa_digits: int = 4,
    ):
        self.base = base
        self.num_exponent_digits = num_exponent_digits
        self.num_mantissa_digits = num_mantissa_digits

    @property
    def num_tokens_per_obj(self) -> int:
        return 2 + self.num_exponent_digits + self.num_mantissa_digits

    def tokens_at_index(self, index: int) -> ordered_set.OrderedSet[str]:
        if index < 0 or index >= self.num_tokens_per_obj:
            raise ValueError(f"Index {index} out of bounds.")

        if index in [0, 1]:  # beginning
            tokens = [_to_token(s) for s in ["+", "-"]]
        else:  # middle (digit)
            tokens = [_to_token(s) for s in range(self.base)]
        return ordered_set.OrderedSet(tokens)

    def to_tokens(self, f: float, /) -> list[str]:
        sign = "+" if f >= 0 else "-"
        abs_f = abs(f)
        exponent = math.floor(np.log(abs_f) / np.log(self.base)) if abs_f > 0 else 0

        exponent_sign = "+" if exponent >= 0 else "-"
        abs_exponent = abs(exponent)

        e = np.base_repr(abs_exponent, base=self.base)
        if len(e) > self.num_exponent_digits and exponent_sign == "+":
            # TODO: Should we round or add 'inf' token?
            raise ValueError(f"Overflow: Exponent {abs_exponent} too large.")
        if len(e) > self.num_exponent_digits and exponent_sign == "-":
            # Underflow.
            all_zeros = ["0"] * (self.num_exponent_digits + self.num_mantissa_digits)
            out = [sign, "-"] + all_zeros
            return [_to_token(s) for s in out]
        e = e.zfill(self.num_exponent_digits)

        mantissa = np.base_repr(
            abs_f * self.base ** (self.num_mantissa_digits - 1 - exponent),
            base=self.base,
        )

        if len(mantissa) > self.num_mantissa_digits:
            mantissa = mantissa[: self.num_mantissa_digits]
        if len(mantissa) < self.num_mantissa_digits:  # Right-pad with zeros.
            mantissa += "0" * (self.num_mantissa_digits - len(mantissa))

        raw_str = sign + exponent_sign + e + mantissa
        return [_to_token(s) for s in raw_str]

    def from_tokens(self, tokens: list[str], /) -> float:
        primitives = [_from_token(t) for t in tokens]

        sign = -1 if primitives[0] == "-" else 1

        exponent_sign = -1 if primitives[1] == "-" else 1

        abs_exponent_str = "".join(
            map(str, primitives[2 : 2 + self.num_exponent_digits])
        )
        abs_exponent = int(abs_exponent_str, base=self.base)
        exponent = exponent_sign * abs_exponent

        mantissa_str = "".join(map(str, primitives[2 + self.num_exponent_digits :]))
        mantissa_unscaled = int(mantissa_str, base=self.base)
        mantissa = mantissa_unscaled / self.base ** (self.num_mantissa_digits - 1)

        return sign * (self.base**exponent) * mantissa


class ExponentFirstTokenizer(DecoderTokenizer[float]):
    """A tokenizer where the first token is sign, second is exponent, then mantissa.

    A float f can be represented as:

    `s * 10^e * m`

    where:
      s: Positive/Negative sign (+, -)
      e: Exponent (integer)
      m: Mantissa representing leading digits.

    Token order: [sign, exponent, digit1, digit2, ..., digitN]

    Attributes:
      num_digits: Number of digits in mantissa. Each digit is between <0> and <9>.
      exponent_range: Controls number of exponent tokens, e.g. if 10, the exponent
        token range will be [<-10>, <10>], affecting the range of representable
        floats.
    """

    def __init__(self, num_digits: int = 4, exponent_range: int = 10):
        self.num_digits = num_digits
        self.exponent_range = exponent_range

    @property
    def num_tokens_per_obj(self) -> int:
        return 2 + self.num_digits

    def tokens_at_index(self, index: int) -> ordered_set.OrderedSet[str]:
        if index < 0 or index >= self.num_tokens_per_obj:
            raise ValueError(f"Index {index} out of bounds.")

        if index == 0:  # sign
            tokens = [_to_token(s) for s in ["+", "-"]]
        elif index == 1:  # exponent
            exps = [
                f"E{i}" for i in range(-self.exponent_range, self.exponent_range + 1)
            ]
            tokens = [_to_token(s) for s in exps]
        else:  # mantissa digits
            tokens = [_to_token(s) for s in range(0, 10)]
        return ordered_set.OrderedSet(tokens)

    @property
    def _max_abs_val(self) -> float:
        """Largest representable positive number."""
        return float(self.num_digits * "9") * (10.0**self.exponent_range)

    @property
    def _min_abs_val(self) -> float:
        """Smallest representable positive number."""
        min_mantissa = float("1" + (self.num_digits - 1) * "0")
        return min_mantissa * (10 ** (-self.exponent_range))

    def _round_float(self, f: float) -> float:
        """Rounds float to the closest in-range value."""
        abs_f = abs(f)
        abs_f = min(abs_f, self._max_abs_val)
        if abs_f < self._min_abs_val:
            # Decides whether to move to 0.0 or `min_abs_val`.
            zero_or_min = round(abs_f / self._min_abs_val)
            abs_f = self._min_abs_val * zero_or_min
        return abs_f if f >= 0 else -abs_f

    def to_tokens(self, f: float, /) -> list[str]:
        f = self._round_float(f)

        if f == 0:
            sign = "+"
            exponent = 0
            mantissa_digits = ["0"] * self.num_digits
        else:
            sign = "+" if f >= 0 else "-"
            abs_f = abs(f)

            # Calculate exponent
            exponent = math.floor(math.log10(abs_f)) if abs_f > 0 else 0
            exponent = max(-self.exponent_range, min(self.exponent_range, exponent))

            # Calculate mantissa
            mantissa = abs_f / (10.0**exponent)
            mantissa_str = f"{mantissa:.{self.num_digits-1}f}".replace(".", "")
            mantissa_digits = list(mantissa_str[: self.num_digits])

            # Pad with zeros if needed
            while len(mantissa_digits) < self.num_digits:
                mantissa_digits.append("0")

        out = [sign, f"E{exponent}"] + mantissa_digits
        return [_to_token(s) for s in out]

    def from_tokens(self, tokens: list[str], /) -> float:
        primitives = [_from_token(t) for t in tokens]

        sign = -1 if primitives[0] == "-" else 1
        exponent_str = primitives[1].lstrip("E")
        exponent = int(exponent_str)

        # Parse mantissa as integer and scale it properly
        mantissa_digits = primitives[2:]
        mantissa = int("".join(mantissa_digits))

        # Scale mantissa to proper decimal value
        # The mantissa represents digits after the decimal point
        mantissa_scaled = mantissa / (10.0 ** (self.num_digits - 1))

        return sign * mantissa_scaled * (10.0**exponent)

    def get_num_tokens(self) -> list[str]:
        num_tokens = []
        for i in range(10):
            num_tokens.append(_to_token(i))
        return num_tokens

    def token_to_number(self, token: str) -> float:
        return int(token[1:-1])

class NormalizedBinaryTokenizer(DecoderTokenizer[float]):
    """Tokenizer for normalized binary fractions in [0, 1).

    Represents a float as a fixed-length binary fraction with `num_bits` bits.

    Encoding uses only two tokens per position: <0> and <1>. The value decoded
    from tokens [b1, b2, ..., bN] is sum(bi * 2^{-i}) for i in [1..N].

    Notes:
      - Values are rounded to the nearest representable fraction k / 2^{num_bits}.
      - The maximum representable value is 1 - 2^{-num_bits}. Any value >= 1.0
        will be clamped down to this maximum.
    """

    def __init__(self, num_bits: int = 8):
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")
        self.num_bits = num_bits

    @property
    def num_tokens_per_obj(self) -> int:
        return self.num_bits

    def tokens_at_index(self, index: int) -> ordered_set.OrderedSet[str]:
        if index < 0 or index >= self.num_tokens_per_obj:
            raise ValueError(f"Index {index} out of bounds.")
        return ordered_set.OrderedSet([_to_token(0), _to_token(1)])

    def _round_and_clip(self, f: float) -> int:
        """Rounds f in [0,1) to nearest integer of k where f ~= k/2^{num_bits}.

        Returns k as an integer in [0, 2^{num_bits}-1].
        """
        # Clamp to [0, 1]
        f_clamped = min(max(f, 0.0), 1.0)
        scale = 1 << self.num_bits  # 2^{num_bits}
        k = int(round(f_clamped * scale))
        # Handle the edge case when f ~= 1.0 -> clamp to max representable
        if k >= scale:
            k = scale - 1
        return k

    def to_tokens(self, f: float, /) -> list[str]:
        k = self._round_and_clip(f)
        # Convert k to a num_bits-length binary string
        bit_str = format(k, f"0{self.num_bits}b")
        return [_to_token(int(ch)) for ch in bit_str]

    def from_tokens(self, tokens: list[str], /) -> float:
        if len(tokens) != self.num_bits:
            raise ValueError(
                f"Expected {self.num_bits} tokens, got {len(tokens)} instead."
            )
        bits = []
        for t in tokens:
            primitive = _from_token(t)
            if primitive not in ("0", "1"):
                raise ValueError(f"Invalid bit token: {t}")
            bits.append(int(primitive))

        # Interpret bits as an integer k where tokens represent the binary of k
        k = 0
        for b in bits:
            k = (k << 1) | b

        # Convert back to fraction k / 2^{num_bits}
        return k / float(1 << self.num_bits)

    def get_num_tokens(self) -> list[str]:
        return [_to_token(0), _to_token(1)]

    def token_to_number(self, token: str) -> int:
        primitive = _from_token(token)
        if primitive not in ("0", "1"):
            raise ValueError(f"Not a binary token: {token}")
        return int(primitive)

class NormalizedTokenizer(DecoderTokenizer[float]):
    """ b in [2,10] [0,1)

    -  `num_digits`  base-b  sum(d_i * b^{-i})i in [1..N]
    -  b^{-num_digits} 1 - b^{-num_digits}
    -  token  <0>..<b-1>
    """

    def __init__(self, num_digits: int = 4, base: int = 6):
        if num_digits <= 0:
            raise ValueError("num_digits must be positive")
        if base < 2 or base > 10:
            raise ValueError("base must be between 2 and 10 (inclusive)")
        self.num_digits = num_digits
        self.base = base

    @property
    def num_tokens_per_obj(self) -> int:
        return self.num_digits

    def tokens_at_index(self, index: int) -> ordered_set.OrderedSet[str]:
        if index < 0 or index >= self.num_tokens_per_obj:
            raise ValueError(f"Index {index} out of bounds.")
        return ordered_set.OrderedSet([_to_token(i) for i in range(self.base)])

    def _round_and_clip(self, f: float) -> int:
        """ f in [0,1]  k/ b^{num_digits} k in [0, b^{num_digits}-1]"""
        f_clamped = min(max(f, 0.0), 1.0)
        scale = int(self.base ** self.num_digits)
        k = int(round(f_clamped * scale))
        if k >= scale:
            k = scale - 1
        return k

    def _int_to_base_str(self, value: int) -> str:
        # base<=10numpy.base_repr  0-9
        s = np.base_repr(value, base=self.base)
        return s

    def to_tokens(self, f: float, /) -> list[str]:
        k = self._round_and_clip(f)
        #  k  base  num_digits
        base_str = self._int_to_base_str(k)
        if len(base_str) < self.num_digits:
            base_str = base_str.zfill(self.num_digits)
        elif len(base_str) > self.num_digits:
            #  _round_and_clip
            base_str = base_str[-self.num_digits :]
        return [_to_token(int(ch)) for ch in base_str]

    def from_tokens(self, tokens: list[str], /) -> float:
        if len(tokens) != self.num_digits:
            raise ValueError(
                f"Expected {self.num_digits} tokens, got {len(tokens)} instead."
            )
        digits: list[int] = []
        for t in tokens:
            primitive = _from_token(t)
            if not primitive.isdigit():
                raise ValueError(f"Invalid digit token: {t}")
            d = int(primitive)
            if d < 0 or d >= self.base:
                raise ValueError(f"Digit {d} out of base-{self.base} range")
            digits.append(d)

        #  base  digits  k
        k = 0
        for d in digits:
            k = k * self.base + d
        return k / float(self.base ** self.num_digits)

    def get_num_tokens(self) -> list[str]:
        return [_to_token(i) for i in range(self.base)]

    def token_to_number(self, token: str) -> int:
        primitive = _from_token(token)
        if not primitive.isdigit():
            raise ValueError(f"Not a base-{self.base} digit token: {token}")
        d = int(primitive)
        if d < 0 or d >= self.base:
            raise ValueError(f"Digit {d} out of base-{self.base} range")
        return d

    def get_min_digit_token(self) -> str:
        """ token <0>"""
        return _to_token(0)

    def get_max_digit_token(self) -> str:
        """ token <base-1>"""
        return _to_token(self.base - 1)



class IntegerTokenizer(DecoderTokenizer[int]):
    """tokenizer-9+9

    1tokentoken<>
    -5<-5>3<3>0<0>
    """

    def __init__(self):
        # -9+919
        self.min_val = -9
        self.max_val = 9

    @property
    def num_tokens_per_obj(self) -> int:
        return 1  # 1token

    def tokens_at_index(self, index: int) -> ordered_set.OrderedSet[str]:
        if index != 0:
            raise ValueError(f"Index {index} out of bounds. Only index 0 is valid.")

        # token-9+9
        tokens = [_to_token(i) for i in range(self.min_val, self.max_val + 1)]
        return ordered_set.OrderedSet(tokens)

    def to_tokens(self, obj: int, /) -> list[str]:
        if not isinstance(obj, (int, float)):
            raise ValueError(f"Expected int or float, got {type(obj)}")

        #
        if isinstance(obj, float):
            obj = int(round(obj))

        #  [-9, 9]
        if obj < self.min_val:
            obj = self.min_val
        elif obj > self.max_val:
            obj = self.max_val

        return [_to_token(obj)]

    def from_tokens(self, tokens: list[str], /) -> int:
        if len(tokens) != 1:
            raise ValueError(f"Expected exactly 1 token, got {len(tokens)}")

        token = tokens[0]
        primitive = _from_token(token)

        try:
            value = int(primitive)
        except ValueError:
            raise ValueError(f"Could not convert token '{token}' to integer")

        if value < self.min_val or value > self.max_val:
            raise ValueError(f"Integer {value} out of range [{self.min_val}, {self.max_val}]")

        return value

    def get_num_tokens(self) -> list[str]:
        """token0-9"""
        num_tokens = []
        for i in range(10):
            num_tokens.append(_to_token(i))
        return num_tokens

    def token_to_number(self, token: str) -> int:
        """token"""
        return int(token[1:-1])
