"""
VaLLM Specialist Model - Tesseract OCR Service.

Author: Joel Otepa Wembo
https://joelwembo.com

Self-contained, reusable OCR module using Tesseract. Use as fallback when
primary text extraction (e.g. fine-tuned pipeline / pdfplumber / pypdf) fails
or returns empty or low-quality text for resumes, job descriptions, or scans.

Supports: PDF (rendered to images), PNG, JPEG, TIFF, BMP. Returns plain text
suitable for ResumeParserModel.predict({"text": ...}) or downstream parsing.

System dependency: Tesseract must be installed (e.g. tesseract-ocr on Linux,
  Windows installer from GitHub, brew install tesseract on macOS).
  Optional: Tesseract language packs (e.g. fra, deu) for non-English docs.

Not wired anywhere by default; import and call when primary extraction fails.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional dependencies (all required for full functionality)
# -----------------------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    Image = None
    PYTESSERACT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
SUPPORTED_PDF_EXTENSION = ".pdf"
DEFAULT_DPI = 200
MAX_PAGES = 50
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_LANG = "eng"
MIN_TEXT_LENGTH_TO_CONSIDER_VALID = 50


class OCRResult:
    """Structured result of an OCR run."""

    __slots__ = ("text", "confidence", "page_count", "warnings", "success", "source", "processing_time_sec")

    def __init__(
        self,
        text: str,
        confidence: float = 0.0,
        page_count: int = 0,
        warnings: Optional[List[str]] = None,
        success: bool = True,
        source: str = "",
        processing_time_sec: float = 0.0,
    ):
        self.text = (text or "").strip()
        self.confidence = max(0.0, min(1.0, confidence))
        self.page_count = max(0, page_count)
        self.warnings = list(warnings or [])
        self.success = success
        self.source = source or ""
        self.processing_time_sec = max(0.0, processing_time_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "page_count": self.page_count,
            "warnings": self.warnings,
            "success": self.success,
            "source": self.source,
            "processing_time_sec": self.processing_time_sec,
        }

    @property
    def is_usable(self) -> bool:
        """True if result has enough text to be used as fallback."""
        return self.success and len(self.text) >= MIN_TEXT_LENGTH_TO_CONSIDER_VALID


def _check_tesseract() -> Optional[str]:
    """Verify Tesseract is installed and return version or error message."""
    if not PYTESSERACT_AVAILABLE:
        return "pytesseract or Pillow not installed"
    try:
        version = pytesseract.get_tesseract_version()
        return None if version else "Tesseract returned no version"
    except pytesseract.TesseractNotFoundError:
        return "Tesseract not found in PATH"
    except Exception as e:
        return str(e)


def _pil_image_from_path(path: Union[str, Path]) -> "Image.Image":
    """Load PIL Image from file path."""
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("Pillow is not installed; install with: pip install Pillow")
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def _pil_image_from_bytes(data: bytes) -> "Image.Image":
    """Load PIL Image from bytes."""
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("Pillow is not installed; install with: pip install Pillow")
    from io import BytesIO
    img = Image.open(BytesIO(data))
    return img.convert("RGB")


def _pdf_pages_to_images(path: Union[str, Path], dpi: int = DEFAULT_DPI, max_pages: int = MAX_PAGES) -> List["Image.Image"]:
    """Render PDF pages to PIL images using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF is not installed; install with: pip install PyMuPDF")
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("Pillow is not installed; install with: pip install Pillow")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    images: List[Image.Image] = []
    doc = fitz.open(path)
    try:
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def _run_tesseract_on_image(
    image: "Image.Image",
    lang: str = DEFAULT_LANG,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, float]:
    """Run Tesseract on a single PIL image. Returns (text, mean_confidence)."""
    if not PYTESSERACT_AVAILABLE or pytesseract is None:
        raise RuntimeError("pytesseract is not installed; install with: pip install pytesseract")
    config = "--psm 3"  # Fully automatic page segmentation
    data = pytesseract.image_to_data(image, lang=lang, config=config, timeout=timeout)
    # Parse TSV for confidence; fallback to image_to_string if parsing fails
    lines = data.strip().split("\n")
    if len(lines) <= 1:
        text = pytesseract.image_to_string(image, lang=lang, config=config, timeout=timeout)
        return (text or "", 0.0)
    headers = lines[0].split("\t")
    try:
        conf_idx = headers.index("conf")
    except ValueError:
        conf_idx = -1
    confidences: List[float] = []
    text_parts: List[str] = []
    for line in lines[1:]:
        parts = line.split("\t", maxsplit=len(headers) - 1)
        if len(parts) <= conf_idx:
            continue
        try:
            c = int(parts[conf_idx])
            if c >= 0:
                confidences.append(c / 100.0)
        except (ValueError, IndexError):
            pass
        if len(parts) > 11:
            text_parts.append(parts[11] if len(parts) > 11 else "")
    if not confidences:
        full_text = pytesseract.image_to_string(image, lang=lang, config=config, timeout=timeout)
        return (full_text or "", 0.0)
    mean_conf = sum(confidences) / len(confidences)
    text = pytesseract.image_to_string(image, lang=lang, config=config, timeout=timeout)
    return (text or "", mean_conf)


def extract_text(
    source: Union[str, Path, bytes],
    *,
    lang: str = DEFAULT_LANG,
    dpi: int = DEFAULT_DPI,
    max_pages: int = MAX_PAGES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    preprocess: bool = True,
) -> OCRResult:
    """
    Extract text using Tesseract OCR (synchronous).

    Args:
        source: File path (str or Path) or raw bytes (image or PDF).
        lang: Tesseract language code (e.g. 'eng', 'eng+fra').
        dpi: DPI for PDF rendering (higher = better quality, slower).
        max_pages: Max PDF pages to process.
        timeout_seconds: Tesseract timeout per page.
        preprocess: Apply basic image preprocessing for scans (contrast, deskew).

    Returns:
        OCRResult with .text, .confidence, .page_count, .warnings, .success, .is_usable.
    """
    import time
    start = time.perf_counter()
    warnings: List[str] = []

    err = _check_tesseract()
    if err:
        logger.warning("Tesseract unavailable: %s", err)
        return OCRResult(
            text="",
            confidence=0.0,
            page_count=0,
            warnings=[f"Tesseract unavailable: {err}"],
            success=False,
            source=_source_repr(source),
            processing_time_sec=time.perf_counter() - start,
        )

    try:
        if isinstance(source, bytes):
            return _extract_from_bytes(source, lang, dpi, max_pages, timeout_seconds, preprocess, warnings, start)
        path = Path(source)
        if not path.is_file():
            return OCRResult(
                text="",
                confidence=0.0,
                page_count=0,
                warnings=[f"File not found: {path}"],
                success=False,
                source=str(path),
                processing_time_sec=time.perf_counter() - start,
            )
        suffix = path.suffix.lower()
        if suffix == SUPPORTED_PDF_EXTENSION:
            return _extract_from_pdf(path, lang, dpi, max_pages, timeout_seconds, preprocess, warnings, start)
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            return _extract_from_image_path(path, lang, timeout_seconds, preprocess, warnings, start)
        return OCRResult(
            text="",
            confidence=0.0,
            page_count=0,
            warnings=[f"Unsupported format: {suffix}. Use PDF or image (png, jpg, tiff, bmp, gif)."],
            success=False,
            source=str(path),
            processing_time_sec=time.perf_counter() - start,
        )
    except Exception as e:
        logger.exception("OCR failed: %s", e)
        return OCRResult(
            text="",
            confidence=0.0,
            page_count=0,
            warnings=[str(e)],
            success=False,
            source=_source_repr(source),
            processing_time_sec=time.perf_counter() - start,
        )


def _source_repr(source: Union[str, Path, bytes]) -> str:
    if isinstance(source, bytes):
        return "<bytes>"
    return str(source)


def _preprocess_image(img: "Image.Image") -> "Image.Image":
    """Basic preprocessing for scanned documents: grayscale, contrast, optional deskew."""
    if not PYTESSERACT_AVAILABLE:
        return img
    try:
        from PIL import ImageEnhance, ImageOps
        if img.mode != "L":
            img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=2)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        return img.convert("RGB")
    except Exception:
        return img.convert("RGB")


def _extract_from_image_path(
    path: Path,
    lang: str,
    timeout: int,
    preprocess: bool,
    warnings: List[str],
    start: float,
) -> OCRResult:
    import time
    try:
        img = _pil_image_from_path(path)
        if preprocess:
            img = _preprocess_image(img)
        text, conf = _run_tesseract_on_image(img, lang=lang, timeout=timeout)
        return OCRResult(
            text=text,
            confidence=conf,
            page_count=1,
            warnings=warnings,
            success=True,
            source=str(path),
            processing_time_sec=time.perf_counter() - start,
        )
    except Exception as e:
        warnings.append(str(e))
        return OCRResult(
            text="",
            confidence=0.0,
            page_count=1,
            warnings=warnings,
            success=False,
            source=str(path),
            processing_time_sec=time.perf_counter() - start,
        )


def _extract_from_bytes(
    data: bytes,
    lang: str,
    dpi: int,
    max_pages: int,
    timeout: int,
    preprocess: bool,
    warnings: List[str],
    start: float,
) -> OCRResult:
    import time
    # Heuristic: PDF starts with %PDF
    if data[:4] == b"%PDF":
        if not PYMUPDF_AVAILABLE:
            warnings.append("PyMuPDF not installed; cannot render PDF from bytes. Install: pip install PyMuPDF")
            return OCRResult(
                text="",
                confidence=0.0,
                page_count=0,
                warnings=warnings,
                success=False,
                source="<bytes>",
                processing_time_sec=time.perf_counter() - start,
            )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            try:
                tmp.write(data)
                tmp.flush()
                return _extract_from_pdf(Path(tmp.name), lang, dpi, max_pages, timeout, preprocess, warnings, start)
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
    # Treat as image
    try:
        img = _pil_image_from_bytes(data)
        if preprocess:
            img = _preprocess_image(img)
        text, conf = _run_tesseract_on_image(img, lang=lang, timeout=timeout)
        return OCRResult(
            text=text,
            confidence=conf,
            page_count=1,
            warnings=warnings,
            success=True,
            source="<bytes>",
            processing_time_sec=time.perf_counter() - start,
        )
    except Exception as e:
        warnings.append(str(e))
        return OCRResult(
            text="",
            confidence=0.0,
            page_count=1,
            warnings=warnings,
            success=False,
            source="<bytes>",
            processing_time_sec=time.perf_counter() - start,
        )


def _extract_from_pdf(
    path: Path,
    lang: str,
    dpi: int,
    max_pages: int,
    timeout: int,
    preprocess: bool,
    warnings: List[str],
    start: float,
) -> OCRResult:
    import time
    try:
        images = _pdf_pages_to_images(path, dpi=dpi, max_pages=max_pages)
    except Exception as e:
        warnings.append(str(e))
        return OCRResult(
            text="",
            confidence=0.0,
            page_count=0,
            warnings=warnings,
            success=False,
            source=str(path),
            processing_time_sec=time.perf_counter() - start,
        )
    if not images:
        return OCRResult(
            text="",
            confidence=0.0,
            page_count=0,
            warnings=warnings,
            success=False,
            source=str(path),
            processing_time_sec=time.perf_counter() - start,
        )
    texts: List[str] = []
    confidences: List[float] = []
    for i, img in enumerate(images):
        if preprocess:
            img = _preprocess_image(img)
        text, conf = _run_tesseract_on_image(img, lang=lang, timeout=timeout)
        texts.append(text or "")
        confidences.append(conf)
    combined = "\n\n".join(texts).strip()
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(
        text=combined,
        confidence=mean_conf,
        page_count=len(images),
        warnings=warnings,
        success=True,
        source=str(path),
        processing_time_sec=time.perf_counter() - start,
    )


async def extract_text_async(
    source: Union[str, Path, bytes],
    *,
    lang: str = DEFAULT_LANG,
    dpi: int = DEFAULT_DPI,
    max_pages: int = MAX_PAGES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    preprocess: bool = True,
) -> OCRResult:
    """
    Async wrapper around extract_text (runs in executor). Use when primary
    extraction fails and you need non-blocking OCR fallback.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: extract_text(
            source,
            lang=lang,
            dpi=dpi,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
            preprocess=preprocess,
        ),
    )


def is_available() -> bool:
    """Return True if Tesseract OCR is available and usable."""
    return _check_tesseract() is None


def get_supported_extensions() -> set:
    """Return set of supported file extensions (e.g. for validation)."""
    s = set(SUPPORTED_IMAGE_EXTENSIONS)
    if PYMUPDF_AVAILABLE:
        s.add(SUPPORTED_PDF_EXTENSION)
    return s
