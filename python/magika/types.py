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

"""Type definitions and data classes for Magika."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class MagikaStatus(str, Enum):
    """Status codes for Magika prediction results."""

    OK = "ok"
    EMPTY_FILE = "empty_file"
    TOO_SHORT = "too_short"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ContentTypeInfo:
    """Metadata about a detected content type."""

    label: str
    mime_type: str
    group: str
    description: str
    extensions: list[str] = field(default_factory=list)
    is_text: bool = False

    def __repr__(self) -> str:
        return (
            f"ContentTypeInfo(label={self.label!r}, "
            f"mime_type={self.mime_type!r}, "
            f"group={self.group!r})"
        )


@dataclass
class MagikaResult:
    """Result of a Magika file type prediction.

    Attributes:
        path: The path of the file that was analyzed.
        dl: The deep-learning-based prediction result.
        output: The final output prediction (may differ from dl if overrides apply).
        score: Confidence score in the range [0.0, 1.0].
        status: Status code indicating success or type of failure.
    """

    path: Path
    dl: ContentTypeInfo
    output: ContentTypeInfo
    score: float
    status: MagikaStatus

    @property
    def ok(self) -> bool:
        """Return True if the prediction completed successfully."""
        return self.status == MagikaStatus.OK

    @property
    def label(self) -> str:
        """Shortcut to the output content type label."""
        return self.output.label

    @property
    def mime_type(self) -> str:
        """Shortcut to the output MIME type string."""
        return self.output.mime_type

    @property
    def is_high_confidence(self) -> bool:
        """Return True if the score meets a high-confidence threshold (>= 0.90)."""
        return self.score >= 0.90

    def __repr__(self) -> str:
        return (
            f"MagikaResult(path={str(self.path)!r}, "
            f"label={self.label!r}, "
            f"score={self.score:.4f}, "
            f"status={self.status.value!r})"
        )


@dataclass
class ModelFeatures:
    """Raw features extracted from a file for model inference.

    Attributes:
        beg: Bytes from the beginning of the file.
        mid: Bytes from the middle of the file.
        end: Bytes from the end of the file.
    """

    beg: bytes
    mid: bytes
    end: bytes

    @property
    def total_bytes(self) -> int:
        """Total number of bytes across all feature segment