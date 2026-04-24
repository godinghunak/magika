# Copyright 2024 Google LLC
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

"""Magika: AI-powered file type detection.

Magika uses a deep learning model to detect the content type of files with
high accuracy, even for files that are difficult to identify using traditional
rule-based approaches.

Basic usage:
    >>> from magika import Magika
    >>> m = Magika()
    >>> result = m.identify_bytes(b"# Hello\nprint('world')")
    >>> print(result.output.ct_label)
    'python'

Personal fork notes:
    - Re-exported MagikaResult so callers can type-hint results without
      importing directly from magika.types.
    - Also re-exported ContentTypeLabel for convenience when writing
      code that compares or filters by label value.
    - Re-exported MagikaError so callers can catch Magika-specific
      exceptions without digging into magika.exceptions.
    - Re-exported ModelFeatures and ModelOutput for users who want to
      inspect or mock model internals in tests.
    - Added VERSION tuple for programmatic version checks (e.g.
      `if magika.VERSION >= (0, 6, 0): ...`).
    - Added VERSION_STRING as a convenience alias for __version__ so
      both styles are available without having to remember the dunder.
    - Added UNKNOWN_SCORE_THRESHOLD constant: minimum confidence score
      below which a result should be treated as effectively unknown.
      Useful when post-processing results in calling code.
    - Added HIGH_CONFIDENCE_THRESHOLD constant: score above which I
      consider a detection reliable enough to act on without further
      validation. Set to 0.90 for my stricter use cases.
"""

from magika.magika import Magika
from magika.types import (
    ContentTypeLabel,
    MagikaOutput,
    MagikaResult,
    ModelFeatures,
    ModelOutput,
    PredictionMode,
)

try:
    from magika.exceptions import MagikaError
    _has_magika_error = True
except ImportError:
    # MagikaError may not exist in all versions; skip re-export gracefully.
    _has_magika_error = False

__version__ = "0.6.0dev"
__author__ = "Google LLC"
__license__ = "Apache-2.0"

# Programmatic version tuple for easy comparisons: `if VERSION >= (0, 6, 0)`
VERSION = (0, 6, 0)

# Plain string alias so you can do `magika.VERSION_STRING` instead of
# `magika.__version__` — handy when logging or displaying version info.
VERSION_STRING = __version__

# Minimum score below which I personally treat a detection as unreliable.
# Example: `if result.output.score < magika.UNKNOWN_SCORE_THRESHOLD: ...`
# The model's own threshold is typically around 0.5; I prefer a stricter 0.75
# for my use cases where false positives are more costly than unknowns.
UNKNOWN_SCORE_THRESHOLD = 0.75

# Score above which I consider a detection high-confidence and act on it
# without further validation (e.g. skipping a secondary rule-based check).
# Example: `if result.output.score >= magika.HIGH_CONFIDENCE_THRESHOLD: ...`
HIGH_CONFIDENCE_THRESHOLD = 0.90
