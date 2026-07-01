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

"""Inference methods."""

import abc
from typing import Protocol, TypeVar

from regress_lm import core
from regress_lm.models import base as model_base

PredictionOutputT = TypeVar("PredictionOutputT")


class InferenceFn(Protocol[PredictionOutputT]):
    """Performs inference to collect some measurement.

    Made very general to allow different inference techniques (sampling,
    ranking, RAFT, etc.).
    """

    @abc.abstractmethod
    def __call__(
        self, model: model_base.Model, x: core.ExampleInput
    ) -> PredictionOutputT:
        """Performs inference on model to collect some measurement."""
