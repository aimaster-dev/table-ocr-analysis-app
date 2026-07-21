"""Domain models for extracted tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class CellContent:
    row: int
    col: int
    text: str
    confidence: float = 1.0


@dataclass(slots=True)
class ExtractedTable:
    """A single table matrix extracted from an image."""

    rows: list[list[str]]
    confidences: list[list[float]] = field(default_factory=list)
    source_image: Path | None = None
    # Zero-based inclusive (row_start, col_start, row_end, col_end).
    merged_ranges: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_rectangular(self) -> list[list[str]]:
        """Pad ragged rows so every row has the same column count."""
        width = self.col_count
        return [list(row) + [""] * (width - len(row)) for row in self.rows]


@dataclass(slots=True)
class ConversionResult:
    image_path: Path
    status: JobStatus
    tables: list[ExtractedTable] = field(default_factory=list)
    output_path: Path | None = None
    output_paths: list[Path] = field(default_factory=list)
    message: str = ""
    elapsed_seconds: float = 0.0
