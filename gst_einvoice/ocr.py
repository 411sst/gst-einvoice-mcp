"""OCR for the pages build 1 routed to ``ocr_required``.

Tesseract is driven through pytesseract's word-level TSV output, so every word
keeps its own confidence. ``OcrPage.text`` is assembled *from those words*, which
is what makes ``OcrWord.start``/``end`` index exactly into it: a downstream stage
that matched a span of the text can ask ``confidence_for_span`` how well that
exact span was read. A single page-level number would hide the case this module
exists to surface -- one field read badly on an otherwise clean page.
"""

import io
import os
import shutil
from dataclasses import dataclass

import numpy as np
import pymupdf
import pytesseract
from PIL import Image, ImageOps

TESSERACT_FALLBACK = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_TESSERACT_CMD: str | None = None


def _resolve_tesseract() -> str:
    """Locate the Tesseract binary once, lazily: PATH first, then the Windows default.

    Boundary validation of an external dependency: if it is missing, every later
    call would fail with pytesseract's own opaque error, so fail here instead and
    say where we looked.
    """
    global _TESSERACT_CMD
    if _TESSERACT_CMD is not None:
        return _TESSERACT_CMD
    found = shutil.which("tesseract")
    if found is None and os.path.exists(TESSERACT_FALLBACK):
        found = TESSERACT_FALLBACK
    if found is None:
        raise RuntimeError(
            "Tesseract OCR is needed to read scanned pages and no binary was found. "
            'Looked for "tesseract" on PATH (shutil.which) and for the default install '
            f"at {TESSERACT_FALLBACK}. Install it with "
            '"winget install UB-Mannheim.TesseractOCR", or put the tesseract '
            "executable on PATH, then re-run."
        )
    _TESSERACT_CMD = found
    return found


def upscale_low_dpi(img: Image.Image, min_width: int = 1700) -> tuple[Image.Image, str | None]:
    """Enlarge a low-resolution capture (a phone photo of A4) toward ``min_width``.

    Returns ``(image, note)``; ``note`` is None when the image was left alone, so
    the caller records the step only when it actually did something. Never
    downscales, and the factor is capped at 4.0 -- past that there is no detail
    left to recover and the interpolation only invents edges.
    """
    if img.width >= min_width:
        return img, None
    factor = min(min_width / img.width, 4.0)
    # Two decimals: at one, a 1653px scan enlarged to 1700px records "upscaled 1.0x",
    # which reads as "not scaled at all". This note is the only record of what the
    # image looked like when the confidences were measured, so it must not round a
    # real resize away. Where two decimals would still round to 1.00x -- 1692-1699px,
    # a hair under min_width -- the resize is skipped instead of recorded as a
    # non-event: the image and the record then agree, and the handful of pixels
    # bought by resampling at 1.0006x was never worth anything to Tesseract.
    note = f"upscaled {factor:.2f}x"
    if note == "upscaled 1.00x":
        return img, None
    size = (round(img.width * factor), round(img.height * factor))
    return img.resize(size, Image.Resampling.LANCZOS), note


def deskew(
    img: Image.Image, max_angle: float = 5.0, step: float = 0.25
) -> tuple[Image.Image, str | None]:
    """Straighten the page by the rotation maximising horizontal projection variance.

    Flat text lines make the per-row ink counts alternate between dense lines and
    empty gaps, which is the maximum-variance state; a tilted page smears them
    together. Candidate angles are scored on a constant-size canvas (``expand=False``)
    so their variances are comparable, and only the winner is applied for real with
    ``expand=True`` and white fill. A winning angle smaller than one ``step`` is
    within the grid's own resolution, so it is treated as no skew and the image is
    returned unchanged. Returns ``(image, note)``.

    The search runs at the page's own resolution. Scoring a downscaled proxy is
    faster but answers a different question: on a straight page the resampling
    aliases the text rows and the proxy reports a skew that is not there, which
    would be recorded in ``preprocessing`` as something that was done to the image.
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.int16)
    threshold = (int(arr.min()) + int(arr.max())) // 2
    mask = arr <= threshold
    if not mask.any() or mask.all():
        # A blank or uniformly dark page has no text lines to align against.
        return img, None
    ink = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")

    steps = int(round(max_angle / step))
    best_angle = 0.0
    best_score = -1.0
    for i in range(-steps, steps + 1):
        angle = i * step
        rotated = ink.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
        profile = np.asarray(rotated, dtype=np.uint8).sum(axis=1, dtype=np.int64)
        score = float(profile.var())
        if score > best_score or (score == best_score and abs(angle) < abs(best_angle)):
            best_angle, best_score = angle, score

    if abs(best_angle) < step:
        return img, None
    straightened = img.rotate(
        best_angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white"
    )
    return straightened, f"deskewed {best_angle:.2f}deg"


def normalize_contrast(img: Image.Image) -> tuple[Image.Image, str | None]:
    """Grayscale, then stretch the histogram to the full range. Returns ``(image, note)``."""
    out = ImageOps.autocontrast(img.convert("L"))
    if img.mode == "L" and out.tobytes() == img.tobytes():
        return img, None
    return out, "contrast normalised"


@dataclass(frozen=True)
class OcrWord:
    """One word Tesseract read. ``start``/``end`` index into ``OcrPage.text``."""

    text: str
    confidence: float
    start: int
    end: int
    line: int


@dataclass(frozen=True)
class OcrPage:
    """One OCR'd page: the text, the words behind it, and how the image was prepared."""

    page: int
    text: str
    words: tuple[OcrWord, ...]
    mean_confidence: float | None
    preprocessing: tuple[str, ...]


def ocr_image(image: Image.Image, *, page: int = 1) -> OcrPage:
    """Preprocess and OCR one page image, keeping per-word confidences and offsets.

    ``page`` is the 1-based page number this image came from, carried through so a
    span can be traced back to its page.
    """
    pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract()

    prepared = image
    applied: list[str] = []
    for stage in (upscale_low_dpi, deskew, normalize_contrast):
        prepared, note = stage(prepared)
        if note is not None:
            applied.append(note)

    data = pytesseract.image_to_data(prepared, output_type=pytesseract.Output.DICT)

    words: list[OcrWord] = []
    chunks: list[str] = []
    cursor = 0
    line_key: tuple[int, int, int] | None = None
    line_index = -1
    for i, raw in enumerate(data["text"]):
        word = str(raw).strip()
        confidence = float(data["conf"][i])
        # conf == -1 marks a layout row (block/paragraph/line), not a word.
        if not word or confidence < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key != line_key:
            if line_key is not None:
                chunks.append("\n")
                cursor += 1
            line_key = key
            line_index += 1
        else:
            chunks.append(" ")
            cursor += 1
        chunks.append(word)
        words.append(
            OcrWord(
                text=word,
                confidence=confidence / 100.0,
                start=cursor,
                end=cursor + len(word),
                line=line_index,
            )
        )
        cursor += len(word)

    mean = sum(w.confidence for w in words) / len(words) if words else None
    return OcrPage(
        page=page,
        text="".join(chunks),
        words=tuple(words),
        mean_confidence=mean,
        preprocessing=tuple(applied),
    )


def ocr_pdf_page(path: str | os.PathLike[str], page_number: int, *, dpi: int = 300) -> OcrPage:
    """Rasterise one 1-based page of a PDF (or single-page image file) and OCR it."""
    with pymupdf.open(path) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise IndexError(
                f"page {page_number} is outside {os.fspath(path)!r}, which has "
                f"{doc.page_count} page(s); page numbers are 1-based"
            )
        pixmap = doc[page_number - 1].get_pixmap(dpi=dpi)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image.load()
    return ocr_image(image, page=page_number)


def confidence_for_span(page: OcrPage, start: int, end: int) -> float | None:
    """Lowest confidence among the words overlapping ``[start, end)``; None if none do.

    The minimum, not the mean: a field is only as trustworthy as its worst-read
    word, and averaging is exactly how a shaky digit disappears into a clean-looking
    number. The span is half-open, so an empty one overlaps nothing and reports None
    rather than the word it happens to sit inside.
    """
    if end <= start:
        return None
    overlapping = [w.confidence for w in page.words if w.start < end and w.end > start]
    return min(overlapping) if overlapping else None
