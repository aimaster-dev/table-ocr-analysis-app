"""Detect ruled-table structure from horizontal / vertical line positions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field

import cv2
import numpy as np

from table_scan.core.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TableGrid:
    """Bounding boxes for each cell: cells[row][col] = (x, y, w, h)."""

    cells: list[list[tuple[int, int, int, int]]]
    is_fallback: bool = False
    # Zero-based inclusive (row_start, col_start, row_end, col_end).
    merged_ranges: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.cells)

    @property
    def cols(self) -> int:
        return max((len(r) for r in self.cells), default=0)


class TableDetector:
    """Build cell matrices from local H/V ruling-line geometry."""

    def __init__(
        self,
        *,
        min_cell_area: int = 200,
        max_tables: int = 5,
        min_rows: int = 2,
        min_cols: int = 2,
        line_merge_gap: int = 0,
    ) -> None:
        self.min_cell_area = min_cell_area
        self.max_tables = max_tables
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.line_merge_gap = line_merge_gap
        self._used_hough = False

    def detect(self, image_bgr: np.ndarray) -> list[TableGrid]:
        self._used_hough = False
        binary = ImagePreprocessor.to_binary(image_bgr)
        horizontal = self._extract_lines(binary, horizontal=True)
        vertical = self._extract_lines(binary, horizontal=False)

        grids = self._grids_from_masks(
            horizontal,
            vertical,
            image_bgr.shape[:2],
            filter_intersections=False,
        )
        if not grids:
            # A second, geometry-based pass recovers long but shaky, slanted,
            # or locally broken hand-drawn rules that rectangular morphology
            # cannot preserve.
            hough_horizontal, hough_vertical = self._extract_hough_lines(binary)
            horizontal = cv2.bitwise_or(horizontal, hough_horizontal)
            vertical = cv2.bitwise_or(vertical, hough_vertical)
            grids = self._grids_from_masks(
                horizontal,
                vertical,
                image_bgr.shape[:2],
                filter_intersections=True,
            )
            self._used_hough = bool(grids)

        if grids:
            grids.sort(key=self._grid_sort_key)
            return grids[: self.max_tables]

        grid_mask = cv2.add(horizontal, vertical)
        contours, _ = cv2.findContours(grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cell_boxes = self._collect_cell_boxes(contours, image_bgr.shape[:2])
        if len(cell_boxes) >= self.min_rows * self.min_cols:
            return self._cluster_into_tables(cell_boxes)[: self.max_tables]

        logger.debug("No reliable grid found; falling back to full-page OCR layout")
        return [self._fallback_grid(image_bgr)]

    def _grids_from_masks(
        self,
        horizontal: np.ndarray,
        vertical: np.ndarray,
        shape: tuple[int, int],
        *,
        filter_intersections: bool,
    ) -> list[TableGrid]:
        grids: list[TableGrid] = []
        for x, y, w, h in self._candidate_regions(
            horizontal, vertical, shape
        ):
            h_crop = horizontal[y : y + h, x : x + w]
            v_crop = vertical[y : y + h, x : x + w]
            merge_gap = self.line_merge_gap or max(3, round(min(w, h) * 0.006))
            local_y_lines = self._line_positions(
                h_crop, axis=0, merge_gap=merge_gap
            )
            local_x_lines = self._line_positions(
                v_crop, axis=1, merge_gap=merge_gap
            )
            if filter_intersections:
                local_x_lines, local_y_lines = self._filter_by_intersections(
                    local_x_lines, local_y_lines, h_crop, v_crop
                )
            local_x_lines, local_y_lines = self._complete_open_boundaries(
                local_x_lines,
                local_y_lines,
                h_crop,
                v_crop,
            )
            y_lines = [y + value for value in local_y_lines]
            x_lines = [x + value for value in local_x_lines]
            logger.info(
                "Table region (%s,%s,%s,%s): %s horizontal, %s vertical rules",
                x,
                y,
                w,
                h,
                len(y_lines),
                len(x_lines),
            )
            if len(y_lines) < self.min_rows + 1 or len(x_lines) < self.min_cols + 1:
                continue
            grid = self._grid_from_lines(x_lines, y_lines, vertical_mask=vertical)
            if grid.rows >= self.min_rows and grid.cols >= self.min_cols:
                grids.append(grid)
        return grids

    def remove_ruling_lines(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an OCR view with long table rules inpainted.

        Coordinates remain unchanged, allowing recognized boxes to be mapped
        back to the detected cell grid.  Morphological opening only selects
        strokes that are much longer than ordinary CJK glyph strokes.
        """
        binary = ImagePreprocessor.to_binary(image_bgr)
        horizontal = self._extract_lines(binary, horizontal=True)
        vertical = self._extract_lines(binary, horizontal=False)
        if self._used_hough:
            hough_horizontal, hough_vertical = self._extract_hough_lines(binary)
            horizontal = cv2.bitwise_or(horizontal, hough_horizontal)
            vertical = cv2.bitwise_or(vertical, hough_vertical)
        mask = cv2.bitwise_or(horizontal, vertical)
        if not np.any(mask):
            return image_bgr.copy()
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        return cv2.inpaint(image_bgr, mask, 2, cv2.INPAINT_TELEA)

    def _extract_lines(self, binary: np.ndarray, *, horizontal: bool) -> np.ndarray:
        height, width = binary.shape
        if horizontal:
            length = max(width // 30, 24)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        else:
            length = max(height // 35, 20)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))

        dilated = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        bridge_length = max(3, length // 8)
        bridge = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (bridge_length, 1) if horizontal else (1, bridge_length),
        )
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, bridge, iterations=1)
        thicken = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 3) if horizontal else (3, 1)
        )
        return cv2.dilate(dilated, thicken, iterations=1)

    @staticmethod
    def _extract_hough_lines(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize near-horizontal/vertical Hough segments for drawn grids."""
        height, width = binary.shape
        horizontal = np.zeros_like(binary)
        vertical = np.zeros_like(binary)
        minimum = min(height, width)
        lines = cv2.HoughLinesP(
            binary,
            1,
            np.pi / 360.0,
            threshold=max(20, minimum // 14),
            minLineLength=max(30, minimum // 9),
            maxLineGap=max(10, minimum // 30),
        )
        if lines is None:
            return horizontal, vertical
        try:
            segments = np.asarray(lines).reshape(-1, 4)
        except ValueError:
            return horizontal, vertical

        for x1, y1, x2, y2 in segments:
            dx, dy = float(x2 - x1), float(y2 - y1)
            angle = abs(float(np.degrees(np.arctan2(dy, dx))))
            angle = min(angle, 180.0 - angle)
            if angle <= 10.0:
                center_y = int(round((int(y1) + int(y2)) / 2.0))
                cv2.line(horizontal, (int(x1), center_y), (int(x2), center_y), 255, 3)
            elif angle >= 80.0:
                center_x = int(round((int(x1) + int(x2)) / 2.0))
                cv2.line(vertical, (center_x, int(y1)), (center_x, int(y2)), 255, 3)

        horizontal = cv2.morphologyEx(
            horizontal,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
            iterations=1,
        )
        vertical = cv2.morphologyEx(
            vertical,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)),
            iterations=1,
        )
        return horizontal, vertical

    def _line_positions(
        self,
        line_img: np.ndarray,
        *,
        axis: int,
        merge_gap: int | None = None,
    ) -> list[int]:
        projection = np.sum(line_img > 0, axis=1 - axis).astype(np.float32)
        if projection.size == 0 or float(projection.max()) <= 0:
            return []

        ortho = line_img.shape[1 - axis]
        # Local regions allow partial rules from merged headers and mildly
        # broken photographed grids.  The line mask has already suppressed
        # ordinary text, so a lower span threshold is safe here.
        # Opening has already rejected ordinary glyph strokes.  Requiring a
        # third of the strongest rule was too aggressive for screenshots and
        # photographed forms: pale or interrupted internal dividers often have
        # only 18-25% of the support of the page edge.  A lower dual floor
        # retains those real rules while still rejecting short text strokes.
        threshold = max(ortho * 0.10, float(projection.max()) * 0.18)
        mask = projection >= threshold
        if not np.any(mask):
            return []

        positions: list[int] = []
        in_run = False
        start = 0
        for i, flag in enumerate(mask.tolist()):
            if flag and not in_run:
                in_run = True
                start = i
            elif not flag and in_run:
                in_run = False
                positions.append((start + i - 1) // 2)
        if in_run:
            positions.append((start + len(mask) - 1) // 2)

        gap = merge_gap if merge_gap is not None else self.line_merge_gap
        return self._merge_close(positions, min_gap=max(2, gap))

    @classmethod
    def _complete_open_boundaries(
        cls,
        x_lines: list[int],
        y_lines: list[int],
        horizontal: np.ndarray,
        vertical: np.ndarray,
    ) -> tuple[list[int], list[int]]:
        """Recover a table boundary clipped exactly by an image edge.

        Phone crops and screenshots commonly omit the final right/bottom rule
        even though every perpendicular rule continues to the edge.  Treating
        the open cell as absent drops a whole column or row.  We add an edge
        only when its gap is comparable to neighboring cells *and* several
        perpendicular rules visibly span the gap, so ordinary page whitespace
        is not converted into a synthetic cell.
        """
        x_lines = cls._complete_axis_edges(
            x_lines,
            extent=horizontal.shape[1],
            cross_lines=y_lines,
            cross_mask=horizontal,
            horizontal_axis=True,
        )
        y_lines = cls._complete_axis_edges(
            y_lines,
            extent=vertical.shape[0],
            cross_lines=x_lines,
            cross_mask=vertical,
            horizontal_axis=False,
        )
        return x_lines, y_lines

    @staticmethod
    def _complete_axis_edges(
        lines: list[int],
        *,
        extent: int,
        cross_lines: list[int],
        cross_mask: np.ndarray,
        horizontal_axis: bool,
    ) -> list[int]:
        if len(lines) < 2 or extent < 2 or len(cross_lines) < 3:
            return lines

        ordered = sorted(set(lines))
        gaps = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
        if not gaps:
            return ordered
        typical = float(np.median(gaps))
        minimum = max(6.0, typical * 0.45)
        maximum = max(minimum, typical * 2.5)
        edge = extent - 1

        def supported(start: int, stop: int) -> bool:
            if stop - start < minimum:
                return False
            hits = 0
            for position in cross_lines:
                if horizontal_axis:
                    sample = cross_mask[
                        max(0, position - 2) : min(cross_mask.shape[0], position + 3),
                        max(0, start) : min(cross_mask.shape[1], stop + 1),
                    ]
                else:
                    sample = cross_mask[
                        max(0, start) : min(cross_mask.shape[0], stop + 1),
                        max(0, position - 2) : min(cross_mask.shape[1], position + 3),
                    ]
                coverage = (
                    float(np.count_nonzero(sample)) / float(sample.size)
                    if sample.size
                    else 0.0
                )
                if coverage >= 0.16:
                    hits += 1
            return hits >= max(3, int(np.ceil(len(cross_lines) * 0.45)))

        leading = ordered[0]
        if minimum <= leading <= maximum and supported(0, leading):
            ordered.insert(0, 0)

        trailing = edge - ordered[-1]
        if minimum <= trailing <= maximum and supported(ordered[-1], edge):
            ordered.append(edge)
        return ordered

    @staticmethod
    def _candidate_regions(
        horizontal: np.ndarray,
        vertical: np.ndarray,
        shape: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        """Find independent connected grid regions before line projection."""
        height, width = shape
        image_area = height * width
        grid_mask = cv2.bitwise_or(horizontal, vertical)
        connected = cv2.morphologyEx(
            grid_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        connected = cv2.dilate(
            connected,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        regions: list[tuple[int, int, int, int]] = []
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (connected > 0).astype(np.uint8), connectivity=8
        )
        for label in range(1, component_count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if w < 40 or h < 30 or w * h < image_area * 0.008:
                continue
            component = (labels == label).astype(np.uint8)
            junctions = cv2.bitwise_and(
                (horizontal > 0).astype(np.uint8),
                (vertical > 0).astype(np.uint8),
            )
            junctions = cv2.bitwise_and(junctions, component)
            junctions = cv2.dilate(
                junctions,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1,
            )
            junction_count, _ = cv2.connectedComponents(junctions, connectivity=8)
            if junction_count - 1 < 6:
                # A plain page/photo frame has four corners but no cell grid.
                continue
            # The padding restores the center of rules widened by morphology.
            pad = 3
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
            regions.append((x1, y1, x2 - x1, y2 - y1))

        regions.sort(key=lambda box: (box[1], box[0]))
        return TableDetector._dedupe_regions(regions)

    @staticmethod
    def _dedupe_regions(
        regions: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        kept: list[tuple[int, int, int, int]] = []
        for region in sorted(regions, key=lambda b: b[2] * b[3], reverse=True):
            if any(TableDetector._iou(region, other) > 0.85 for other in kept):
                continue
            kept.append(region)
        return sorted(kept, key=lambda box: (box[1], box[0]))

    @staticmethod
    def _grid_sort_key(grid: TableGrid) -> tuple[int, int]:
        if not grid.cells or not grid.cells[0]:
            return (0, 0)
        x, y, _, _ = grid.cells[0][0]
        return (y, x)

    def _filter_by_intersections(
        self,
        x_lines: list[int],
        y_lines: list[int],
        horizontal: np.ndarray,
        vertical: np.ndarray,
    ) -> tuple[list[int], list[int]]:
        """Remove unrelated page frames while retaining merged header rules."""
        if not x_lines or not y_lines:
            return x_lines, y_lines

        def crosses(x: int, y: int) -> bool:
            radius = 3
            y1, y2 = max(0, y - radius), min(horizontal.shape[0], y + radius + 1)
            x1, x2 = max(0, x - radius), min(horizontal.shape[1], x + radius + 1)
            return bool(
                np.any(horizontal[y1:y2, x1:x2])
                and np.any(vertical[y1:y2, x1:x2])
            )

        # A page/document frame intersects only its own two rules.  Real table
        # columns intersect several row rules even when a merged header omits
        # some internal dividers.
        min_column_crossings = max(3, self.min_rows + 1)
        strong_x = [
            x for x in x_lines if sum(crosses(x, y) for y in y_lines) >= min_column_crossings
        ]
        if len(strong_x) < self.min_cols + 1:
            return x_lines, y_lines

        strong_y = [y for y in y_lines if sum(crosses(x, y) for x in strong_x) >= 2]
        if len(strong_y) < self.min_rows + 1:
            return x_lines, y_lines

        strong_x = [
            x
            for x in strong_x
            if sum(crosses(x, y) for y in strong_y) >= min_column_crossings
        ]
        if len(strong_x) < self.min_cols + 1:
            return x_lines, y_lines
        return strong_x, strong_y

    @staticmethod
    def _merge_close(values: list[int], *, min_gap: int) -> list[int]:
        if not values:
            return []
        merged = [values[0]]
        for value in values[1:]:
            if value - merged[-1] < min_gap:
                merged[-1] = (merged[-1] + value) // 2
            else:
                merged.append(value)
        return merged

    def _grid_from_lines(
        self,
        x_lines: list[int],
        y_lines: list[int],
        *,
        vertical_mask: np.ndarray | None = None,
    ) -> TableGrid:
        cells: list[list[tuple[int, int, int, int]]] = []
        for r in range(len(y_lines) - 1):
            y1, y2 = y_lines[r], y_lines[r + 1]
            if y2 - y1 < 6:
                continue
            row: list[tuple[int, int, int, int]] = []
            for c in range(len(x_lines) - 1):
                x1, x2 = x_lines[c], x_lines[c + 1]
                if x2 - x1 < 6:
                    continue
                row.append((x1, y1, x2 - x1, y2 - y1))
            if row:
                cells.append(row)
        merged_ranges = (
            self._horizontal_merges(x_lines, y_lines, vertical_mask)
            if vertical_mask is not None
            else []
        )
        return TableGrid(cells=cells, merged_ranges=merged_ranges)

    @staticmethod
    def _horizontal_merges(
        x_lines: list[int],
        y_lines: list[int],
        vertical_mask: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Infer colspans where an internal divider is absent for one row."""
        merges: list[tuple[int, int, int, int]] = []
        column_count = len(x_lines) - 1
        row_count = len(y_lines) - 1
        supports = np.zeros((row_count, max(0, column_count - 1)), dtype=np.float32)
        for row in range(row_count):
            y1, y2 = y_lines[row], y_lines[row + 1]
            for boundary in range(1, column_count):
                x = x_lines[boundary]
                sample = vertical_mask[
                    max(0, y1 + 3) : min(vertical_mask.shape[0], y2 - 2),
                    max(0, x - 2) : min(vertical_mask.shape[1], x + 3),
                ]
                supports[row, boundary - 1] = (
                    float(np.count_nonzero(sample)) / float(sample.size)
                    if sample.size
                    else 0.0
                )

        for row in range(len(y_lines) - 1):
            y1, y2 = y_lines[row], y_lines[row + 1]
            if y2 - y1 < 8:
                continue
            absent: list[int] = []
            for boundary in range(1, column_count):
                support = float(supports[row, boundary - 1])
                neighbors: list[float] = []
                if row > 0:
                    neighbors.append(float(supports[row - 1, boundary - 1]))
                if row + 1 < row_count:
                    neighbors.append(float(supports[row + 1, boundary - 1]))
                # A real merged band is normally bounded by rows where the
                # divider is visible.  Requiring those neighbors prevents a
                # faint divider missing across several photographed rows from
                # collapsing whole records into one merged cell.
                bounded_absence = bool(neighbors) and all(
                    value >= 0.12 for value in neighbors
                )
                if support < 0.12 and bounded_absence:
                    absent.append(boundary)

            start: int | None = None
            end: int | None = None
            for boundary in range(1, column_count):
                if boundary in absent:
                    if start is None:
                        start = boundary - 1
                    end = boundary
                elif start is not None and end is not None:
                    merges.append((row, start, row, end))
                    start = end = None
            if start is not None and end is not None:
                merges.append((row, start, row, end))
        return merges

    def _collect_cell_boxes(
        self,
        contours: list[np.ndarray],
        shape: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        height, width = shape
        image_area = height * width
        boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_cell_area:
                continue
            if area > image_area * 0.45:
                continue
            if w > width * 0.92 and h > height * 0.35:
                continue
            aspect = w / max(h, 1)
            if aspect > 25 or aspect < 0.04:
                continue
            boxes.append((x, y, w, h))

        boxes = self._dedupe_boxes(boxes)
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    @staticmethod
    def _dedupe_boxes(
        boxes: list[tuple[int, int, int, int]],
        iou_threshold: float = 0.7,
    ) -> list[tuple[int, int, int, int]]:
        kept: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if any(TableDetector._iou(box, other) > iou_threshold for other in kept):
                continue
            kept.append(box)
        return kept

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def _cluster_into_tables(
        self,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[TableGrid]:
        if not boxes:
            return []

        rows = self._group_by_axis(boxes, axis="y")
        matrix = [sorted(row_boxes, key=lambda b: b[0]) for row_boxes in rows]
        return [TableGrid(cells=matrix)]

    def _group_by_axis(
        self,
        boxes: list[tuple[int, int, int, int]],
        *,
        axis: str,
    ) -> list[list[tuple[int, int, int, int]]]:
        if not boxes:
            return []

        idx = 1 if axis == "y" else 0
        size_idx = 3 if axis == "y" else 2
        sorted_boxes = sorted(boxes, key=lambda b: b[idx])
        median_size = float(np.median([b[size_idx] for b in sorted_boxes]))
        threshold = max(median_size * 0.6, 12.0)

        groups: list[list[tuple[int, int, int, int]]] = [[sorted_boxes[0]]]
        for box in sorted_boxes[1:]:
            prev = groups[-1][-1]
            prev_center = prev[idx] + prev[size_idx] / 2
            cur_center = box[idx] + box[size_idx] / 2
            if abs(cur_center - prev_center) <= threshold:
                groups[-1].append(box)
            else:
                groups.append([box])
        return groups

    def _fallback_grid(self, image_bgr: np.ndarray) -> TableGrid:
        h, w = image_bgr.shape[:2]
        return TableGrid(cells=[[(0, 0, w, h)]], is_fallback=True)
