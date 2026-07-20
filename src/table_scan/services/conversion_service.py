"""Application services coordinating batch conversion."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from table_scan.config.settings import AppSettings, SUPPORTED_IMAGE_EXTENSIONS
from table_scan.core.pipeline import TableExtractionPipeline
from table_scan.models.table_result import ConversionResult, JobStatus

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, ConversionResult], None]


class ConversionService:
    """Batch-convert a directory (or explicit file list) of table photos."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._pipeline = TableExtractionPipeline(settings)

    def warm_up(self) -> None:
        self._pipeline.warm_up()

    @staticmethod
    def discover_images(directory: Path) -> list[Path]:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))
        files = [
            p
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        return files

    def convert_many(
        self,
        images: Iterable[Path],
        output_dir: Path,
        *,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[ConversionResult]:
        image_list = list(images)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[ConversionResult] = []
        total = len(image_list)

        for index, image_path in enumerate(image_list, start=1):
            if should_cancel and should_cancel():
                remaining = image_list[index - 1 :]
                for skipped in remaining:
                    result = ConversionResult(
                        image_path=skipped,
                        status=JobStatus.SKIPPED,
                        message="Cancelled by user",
                    )
                    results.append(result)
                    if on_progress:
                        on_progress(index, total, result)
                break

            logger.info("Converting (%s/%s): %s", index, total, image_path.name)
            result = self._pipeline.convert_image(image_path, output_dir)
            results.append(result)
            if on_progress:
                on_progress(index, total, result)

        return results
