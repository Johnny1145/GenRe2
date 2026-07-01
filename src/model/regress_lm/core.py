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

"""Core API for RegressLM."""

import dataclasses
from typing import Sequence

import torch


# High-level example.
@dataclasses.dataclass
class ExampleInput:
    x: str


@dataclasses.dataclass
class ExampleInputNumeric:
    x: float | Sequence[float]


@dataclasses.dataclass
class Example(ExampleInput):
    y: float | Sequence[float]


@dataclasses.dataclass
class ExampleNumeric(ExampleInputNumeric):
    y: float | Sequence[float]


@dataclasses.dataclass
class ExampleRL(Example):
    y_median: float | Sequence[float]
    y_mean: float | Sequence[float]
    y_std: float | Sequence[float]
    q1: float | Sequence[float]
    q3: float | Sequence[float]


@dataclasses.dataclass
class ExampleRLNumeric(ExampleNumeric):
    y_median: float | Sequence[float]
    y_mean: float | Sequence[float]
    y_std: float | Sequence[float]
    q1: float | Sequence[float]
    q3: float | Sequence[float]

@dataclasses.dataclass
class ExamplebyteRLNumeric(ExampleNumeric):
    y_max: float | Sequence[float]
    y_min: float | Sequence[float]
