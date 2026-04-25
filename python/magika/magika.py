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

"""Core Magika class for AI-powered file type detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from magika.types import (
    ContentTypeInfo,
    MagikaResult,
    MagikaStatus,
)


# Default thresholds for model confidence
DEFAULT_PREDICTION_MODE = "medium-confidence"
HIGH_CONFIDENCE_THRESHOLD = 0.95
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.40

# Number of bytes to read from the beginning and end of a file
BEGINNING_BYTES = 512
END_BYTES = 512


class Magika:
    """Magika: AI-powered file type detection.

    This class provides the main interface for detecting the content type
    of files using a trained deep learning model.

    Example usage:
        magika = Magika()
        result = magika.identify_path(Path("/path/to/file"))
        print(result.output.ct_label)
    """

    def __init__(
        self,
        prediction_mode: str = DEFAULT_PREDICTION_MODE,
        no_dereference: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize Magika.

        Args:
            prediction_mode: Confidence mode - 'best-guess', 'medium-confidence',
                             or 'high-confidence'.
            no_dereference: If True, do not follow symbolic links.
            verbose: Enable verbose output.
            debug: Enable debug output.
        """
        self._prediction_mode = prediction_mode
        self._no_dereference = no_dereference
        self._verbose = verbose
        self._debug = debug
        self._model = None  # Lazy-loaded on first use

    def identify_path(self, path: Path) -> MagikaResult:
        """Identify the content type of a single file path.

        Args:
            path: Path to the file to identify.

        Returns:
            MagikaResult with detection results.
        """
        path = Path(path)

        if not path.exists():
            return MagikaResult(
                path=path,
                dl=ContentTypeInfo(ct_label="unknown", mime_type="application/octet-stream", group="unknown", description="Unknown", extensions=[], is_text=False),
                output=ContentTypeInfo(ct_label="unknown", mime_type="application/octet-stream", group="unknown", description="Unknown", extensions=[], is_text=False),
                score=0.0,
                status=MagikaStatus.FILE_NOT_FOUND_ERROR,
            )

        if path.is_symlink() and self._no_dereference:
            return MagikaResult(
                path=path,
                dl=ContentTypeInfo(ct_label="symlink", mime_type="inode/symlink", group="symlink", description="Symbolic link", extensions=[], is_text=False),
                output=ContentTypeInfo(ct_label="symlink", mime_type="inode/symlink", group="symlink", description="Symbolic link", extensions=[], is_text=False),
                score=1.0,
                status=MagikaStatus.OK,
            )

        if path.is_dir():
            return MagikaResult(
                path=path,
                dl=ContentTypeInfo(ct_label="directory", mime_type="inode/directory", group="directory", description="Directory", extensions=[], is_text=False),
                output=ContentTypeInfo(ct_label="directory", mime_type="inode/directory", group="directory", description="Directory", extensions=[], is_text=False),
                score=1.0,
                status=MagikaStatus.OK,
            )

        try:
            content = self._read_file_content(path)
            return self.identify_bytes(content, path=path)
        except PermissionError:
            return MagikaResult(
                path=path,
                dl=ContentTypeInfo(ct_label="unknown", mime_type="application/octet-stream", group="unknown", description="Unknown", extensions=[], is_text=False),
                output=ContentTypeInfo(ct_label="unknown", mime_type="application/octet-stream", group="unknown", description="Unknown", extensions=[], is_text=False),
                score=0.0,
                status=MagikaStatus.PERMISSION_ERROR,
            )

    def identify_paths(self, paths: List[Path]) -> List[MagikaResult]:
        """Identify content types for multiple file paths.

        Args:
            paths: List of paths to identify.

        Returns:
            List of MagikaResult objects, one per input path.
        """
        return [self.identify_path(p) for p in paths]

    def identify_bytes(self, content: bytes, path: Optional[Path] = None) -> MagikaResult:
        """Identify the content type from raw bytes.

        Args:
            content: Raw bytes to analyze.
            path: Optional path for result metadata.

        Returns:
            MagikaResult with detection results.
        """
        # TODO: Integrate with the ONNX model for real inference
        # For now, perform basic heuristic detection
        ct_label, mime_type, score = self._basic_heuristic(content)

        ct_info = ContentTypeInfo(
            ct_label=ct_label,
            mime_type=mime_type,
            group=self._get_group(ct_label),
            description=ct_label.upper(),
            extensions=self._get_extensions(ct_label),
            is_text=self._is_text_type(ct_label),
        )

        return MagikaResult(
            path=path or Path("-"),
            dl=ct_info,
            output=ct_info,
            score=score,
            status=MagikaStatus.OK,
        )

    def _read_file_content(self, path: Path) -> bytes:
        """Read relevant bytes from a file for analysis."""
        file_size = path.stat().st_size
        if file_size == 0:
            return b""

        with open(path, "rb") as f:
            if file_size <= BEGINNING_BYTES + END_BYTES:
                return f.read()

            beginning = f.read(BEGINNING_BYTES)
            f.seek(-END_BYTES, os.SEEK_END)
            end = f.read(END_BYTES)
            return beginning + end

    def _basic_heuristic(self, content: bytes) -> tuple:
        """Basic heuristic file type detection based on magic bytes."""
        if not content:
            return "empty", "application/x-empty", 1.0

        # Check common magic bytes
        if content[:4] == b"\x89PNG":
            return "png", "image/png", 0.99
        if content[:3] == b"GIF":
            return "gif", "image/gif", 0.99
        if content[:2] == b"\xff\xd8":
            return "jpeg", "image/jpeg", 0.99
        if content[:4] in (b"\x1f\x8b\x08", b"PK\x03\x04"):
            return "zip", "application/zip", 0.99
        if content[:4] == b"%PDF":
            return "pdf", "application/pdf", 0.99
        if content[:4] == b"\x7fELF":
            return "elf", "application/x-elf", 0.99
        if content[:2] in (b"MZ", b"ZM"):
            return "pe", "application/x-dosexec", 0.99

        # Try to detect text content
        try:
            content[:512].decode("utf-8")
            if content.lstrip().startswith(b"<"):
                if b"<html" in content[:512].lower() or b"<!doctype html" in content[:512].lower():
                    return "html", "text/html", 0.85
                return "xml", "application/xml", 0.80
            if content.lstrip().startswith(b"{") or content.lstrip().startswith(b"["):
                return "json", "application/json", 0.80
            return "txt", "text/plain", 0.70
        except UnicodeDecodeError:
            pass

        return "unknown", "application/octet-stream", 0.50

    def _get_group(self, ct_label: str) -> str:
        """Get the content type group for a label."""
        image_types = {"png", "jpeg", "gif", "bmp", "webp", "tiff", "ico"}
        text_types = {"txt", "html", "xml", "json", "csv", "markdown", "rst"}
        binary_types = {"elf", "pe", "macho", "zip", "tar", "gz", "pdf"}

        if ct_label in image_types:
            return "image"
        if ct_label in text_types:
            return "text"
        if ct_label in binary_types:
            return "binary"
        return "unknown"

    def _get_extensions(self, ct_label: str) -> List[str]:
        """Get common file extensions for a content type label."""
        extensions_map = {
            "png": ["png"],
            "jpeg": ["jpg", "jpeg"],
            "gif": ["gif"],
            "txt": ["txt"],
            "html": ["html", "htm"],
            "xml": ["xml"],
            "json": ["json"],
            "pdf": ["pdf"],
            "zip": ["zip"],
            "elf": [],
            "pe": ["exe", "dll"],
        }
        return extensions_map.get(ct_label, [])

    def _is_text_type(self, ct_label: str) -> bool:
        """Determine if a content type label represents text content."""
        text_types = {"txt", "html", "xml", "json", "csv", "markdown", "rst",
                      "python", "javascript", "typescript", "c", "cpp", "java",
                      "go", "rust", "shell", "yaml", "toml", "ini", "conf"}
        return ct_label in text_types
