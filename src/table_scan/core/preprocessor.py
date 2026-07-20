"""Image preprocessing for more reliable photographed-document OCR."""

from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """Rectify, deskew, and contrast-enhance scanned / photographed tables."""

    def __init__(
        self,
        *,
        deskew: bool = True,
        enhance_contrast: bool = True,
        rectify_perspective: bool = True,
    ) -> None:
        self.deskew = deskew
        self.enhance_contrast = enhance_contrast
        self.rectify_perspective = rectify_perspective

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image provided for preprocessing")

        working = image_bgr.copy()
        if self.rectify_perspective:
            working = self._rectify_perspective(working)
        if self.deskew:
            working = self._deskew(working)
        if self.enhance_contrast:
            working = self._enhance(working)
        return working

    @staticmethod
    def _rectify_perspective(image_bgr: np.ndarray) -> np.ndarray:
        """Rectify a strong document/table quadrilateral, otherwise do nothing."""
        height, width = image_bgr.shape[:2]
        image_area = float(height * width)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=2,
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.22 or area > image_area * 0.995:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            points = polygon.reshape(4, 2).astype(np.float32)
            ordered = ImagePreprocessor._order_quad(points)
            if not ImagePreprocessor._reasonable_quad(ordered):
                continue

            x, y, box_w, box_h = cv2.boundingRect(polygon)
            bbox_corners = np.asarray(
                [[x, y], [x + box_w, y], [x + box_w, y + box_h], [x, y + box_h]],
                dtype=np.float32,
            )
            corner_error = float(np.mean(np.linalg.norm(ordered - bbox_corners, axis=1)))
            if corner_error < np.hypot(width, height) * 0.012:
                return image_bgr

            top = np.linalg.norm(ordered[1] - ordered[0])
            bottom = np.linalg.norm(ordered[2] - ordered[3])
            left = np.linalg.norm(ordered[3] - ordered[0])
            right = np.linalg.norm(ordered[2] - ordered[1])
            target_w = int(round(max(top, bottom)))
            target_h = int(round(max(left, right)))
            if target_w < 120 or target_h < 120 or max(target_w, target_h) / min(target_w, target_h) > 8:
                continue
            destination = np.asarray(
                [
                    [0, 0],
                    [target_w - 1, 0],
                    [target_w - 1, target_h - 1],
                    [0, target_h - 1],
                ],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(ordered, destination)
            return cv2.warpPerspective(
                image_bgr,
                matrix,
                (target_w, target_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return image_bgr

    @staticmethod
    def _order_quad(points: np.ndarray) -> np.ndarray:
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(differences)]
        ordered[3] = points[np.argmax(differences)]
        return ordered

    @staticmethod
    def _reasonable_quad(points: np.ndarray) -> bool:
        for index in range(4):
            previous = points[(index - 1) % 4] - points[index]
            following = points[(index + 1) % 4] - points[index]
            denominator = np.linalg.norm(previous) * np.linalg.norm(following)
            if denominator <= 1e-6:
                return False
            cosine = abs(float(np.dot(previous, following) / denominator))
            if cosine > 0.58:  # reject angles sharper than about 55 degrees
                return False
        return True

    def _enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """Improve local contrast without blurring small CJK strokes."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _deskew(self, image_bgr: np.ndarray) -> np.ndarray:
        """Estimate rotation from long rules, falling back to dark foreground."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        angle = self._rule_angle(gray)
        if angle is None:
            # The old implementation inverted the grayscale image directly,
            # which treated off-white paper and shadows as foreground.  A real
            # binary foreground mask is required for a meaningful rectangle.
            _, foreground = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            coords = np.column_stack(np.where(foreground > 0))
            if len(coords) < 20:
                return image_bgr
            rect_angle = float(cv2.minAreaRect(coords)[-1])
            angle = -(90.0 + rect_angle) if rect_angle < -45.0 else -rect_angle

        if abs(angle) < 0.3 or abs(angle) > 15:
            return image_bgr

        height, width = image_bgr.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            image_bgr,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _rule_angle(gray: np.ndarray) -> float | None:
        """Return the weighted median angle of plausible horizontal rules."""
        height, width = gray.shape
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 1800.0,
            threshold=max(25, width // 12),
            minLineLength=max(40, width // 5),
            maxLineGap=max(8, width // 50),
        )
        if lines is None:
            return None

        candidates: list[tuple[float, float]] = []
        try:
            line_rows = np.asarray(lines).reshape(-1, 4)
        except ValueError:
            return None
        for x1, y1, x2, y2 in line_rows:
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dx) < 1.0:
                continue
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if abs(angle) <= 15.0:
                candidates.append((angle, float(np.hypot(dx, dy))))
        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        half_weight = sum(weight for _, weight in candidates) / 2.0
        accumulated = 0.0
        for angle, weight in candidates:
            accumulated += weight
            if accumulated >= half_weight:
                # Positive line angle means the image must rotate clockwise.
                return angle
        return candidates[-1][0]

    @staticmethod
    def to_binary(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # A 3x3 blur suppresses sensor noise without erasing fine CJK strokes.
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        block_size = max(15, (min(gray.shape) // 35) | 1)
        block_size = min(block_size, 51)
        return cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            9,
        )
