from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

ORIENTATION_ANGLES = (0, 90, 180, 270)


@dataclass
class PreprocessConfig:
    proxy_max_side: int = 1400
    min_page_area_ratio: float = 0.25
    min_quad_score: float = 0.45
    deskew_limit_degrees: float = 10.0
    border_px: int = 12
    orientation: str = "auto"  # auto, none, or 0/90/180/270
    output_size: tuple[int, int] | None = None


@dataclass
class PreprocessResult:
    source: str
    stages: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _resize_proxy(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def load_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as pil_image:
        exif_orientation = pil_image.getexif().get(274)
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        rgb = np.asarray(pil_image)
        metadata = {
            "width": pil_image.width,
            "height": pil_image.height,
            "mode": pil_image.mode,
            "exif_orientation": exif_orientation,
        }
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), metadata


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype("float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype="float32",
    )


def _quad_score(
    quad: np.ndarray, shape: tuple[int, int], contour_area: float
) -> float:
    height, width = shape
    area_ratio = contour_area / (height * width)
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    rectangularity = min(sides) / max(sides)
    return float(
        0.65 * min(1.0, area_ratio / 0.8) + 0.35 * rectangularity
    )


def detect_page(
    image: np.ndarray, config: PreprocessConfig
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    proxy, scale = _resize_proxy(image, config.proxy_max_side)
    gray = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    best: tuple[float, np.ndarray] | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contour)
        if area / (proxy.shape[0] * proxy.shape[1]) < config.min_page_area_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = _order_points(approx)
        score = _quad_score(quad, proxy.shape[:2], area)
        if best is None or score > best[0]:
            best = (score, quad)

    warnings: list[str] = []
    if best is None or best[0] < config.min_quad_score:
        warnings.append(
            "No confident page quadrilateral found; using the full image."
        )
        return image.copy(), {
            "page_detected": False,
            "page_score": None,
            "corners": None,
            "edge_proxy": closed,
        }, warnings

    score, proxy_quad = best
    quad = proxy_quad / scale
    top_left, top_right, bottom_right, bottom_left = quad
    width = int(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    height = int(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if config.output_size:
        width, height = config.output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(quad, destination)
    warped = cv2.warpPerspective(
        image, matrix, (width, height), borderValue=(255, 255, 255)
    )
    return warped, {
        "page_detected": True,
        "page_score": score,
        "corners": quad.tolist(),
        "edge_proxy": closed,
    }, warnings


def rotate_counterclockwise(image: np.ndarray, angle: int) -> np.ndarray:
    if angle not in ORIENTATION_ANGLES:
        raise ValueError(f"angle must be one of {ORIENTATION_ANGLES}")
    return {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
    }[angle]


def estimate_orientation(
    image: np.ndarray, requested: str = "auto"
) -> tuple[int, float, str, tuple[int, ...]]:
    if requested == "none":
        return 0, 1.0, "disabled", (0,)
    if requested != "auto":
        angle = int(requested) % 360
        if angle not in ORIENTATION_ANGLES:
            raise ValueError(
                f"orientation must be auto, none, or one of {ORIENTATION_ANGLES}"
            )
        return angle, 1.0, "manual", (angle,)

    try:
        from paddleocr import DocImgOrientationClassification

        classifier = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori", device="cpu"
        )
        result = next(iter(classifier.predict(image, batch_size=1)))
        payload = result.json if isinstance(result.json, dict) else result.json()
        response = payload.get("res", payload)
        angle = int(
            response.get(
                "label_names", [str(int(response.get("angle", 0)))]
            )[0]
        )
        score = float(np.asarray(response.get("scores", [1.0])).ravel()[0])
        angle %= 360
        if angle not in ORIENTATION_ANGLES:
            raise ValueError(f"Unsupported PaddleOCR orientation: {angle}")
        return angle, score, "paddleocr", (angle,)
    # Model loading and inference are optional; any provider/runtime failure
    # deliberately falls back to deterministic OpenCV processing.
    except Exception as exc:  # noqa: BLE001
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scores: dict[int, float] = {}
        for angle in (0, 90):
            rotated = rotate_counterclockwise(gray, angle)
            binary = cv2.threshold(
                rotated,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1]
            horizontal_rules = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (max(30, rotated.shape[1] // 8), 1)
                ),
            )
            vertical_rules = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (1, max(30, rotated.shape[0] // 8))
                ),
            )
            text_only = cv2.subtract(
                binary, cv2.bitwise_or(horizontal_rules, vertical_rules)
            )
            horizontal_text = cv2.morphologyEx(
                text_only,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (max(15, rotated.shape[1] // 25), 1)
                ),
            )
            scores[angle] = float(horizontal_text.mean())
        axis_angle = max(scores, key=scores.get)
        confidence = float(
            (scores[axis_angle] + 1e-6) / (scores[90 - axis_angle] + 1e-6)
        )
        candidates = (axis_angle, (axis_angle + 180) % 360)
        method = f"text-axis-fallback-ambiguous ({type(exc).__name__})"
        return axis_angle, confidence, method, candidates


def deskew(image: np.ndarray, limit: float) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(20, image.shape[1] // 30), 1)
        ),
    )
    points = cv2.findNonZero(horizontal)
    if points is None or len(points) < 30:
        return image, 0.0
    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) > limit:
        return image, 0.0
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    straightened = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return straightened, float(angle)


def enhance_for_ocr(
    image: np.ndarray, border_px: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(normalized)
    normalized = cv2.medianBlur(normalized, 3)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    normalized = cv2.copyMakeBorder(
        normalized,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    binary = cv2.copyMakeBorder(
        binary,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    return normalized, binary, background


def preprocess_document(
    path: Path, config: PreprocessConfig | None = None
) -> PreprocessResult:
    config = config or PreprocessConfig()
    started = time.perf_counter()
    original, metadata = load_image(path)
    page, page_metadata, warnings = detect_page(original, config)
    angle, orientation_score, orientation_method, orientation_candidates = (
        estimate_orientation(page, config.orientation)
    )
    oriented = rotate_counterclockwise(page, angle)
    straightened, skew_angle = deskew(
        oriented, config.deskew_limit_degrees
    )
    normalized, binary, background = enhance_for_ocr(
        straightened, config.border_px
    )
    metadata.update(
        {
            "orientation_angle": angle,
            "orientation_candidates": list(orientation_candidates),
            "orientation_score": orientation_score,
            "orientation_method": orientation_method,
            "deskew_angle": skew_angle,
            "elapsed_seconds": time.perf_counter() - started,
            "page_detected": page_metadata["page_detected"],
            "page_score": page_metadata["page_score"],
            "corners": page_metadata["corners"],
        }
    )
    stages = {
        "original": original,
        "page": page,
        "oriented": oriented,
        "deskewed": straightened,
        "normalized_gray": normalized,
        "adaptive_binary": binary,
        "background_estimate": background,
        "edge_proxy": page_metadata["edge_proxy"],
    }
    return PreprocessResult(str(path), stages, metadata, warnings)


def save_result(result: PreprocessResult, root: Path) -> Path:
    output_dir = root / Path(result.source).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_dir / "normalized_gray.png"),
        result.stages["normalized_gray"],
    )
    cv2.imwrite(
        str(output_dir / "adaptive_binary.png"),
        result.stages["adaptive_binary"],
    )
    payload = {
        "source": result.source,
        "metadata": result.metadata,
        "warnings": result.warnings,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return output_dir
