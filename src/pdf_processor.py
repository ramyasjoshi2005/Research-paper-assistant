"""
PDF Processor Module
Extracts, cleans, and validates text from PDF documents page by page.
Normalizes Unicode typographic ligatures, eliminates control symbols,
and caches extracted pages to avoid redundant PDF parsing.
Detects image-only / scanned documents requiring OCR.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src.config import EXTRACTED_TEXT_CACHE_DIR, PDF_DIR

logger = logging.getLogger(__name__)


def clean_page_text(text: str) -> str:
    """
    Cleans extracted PDF page text rigorously:
    - Normalizes Unicode (NFKC) to expand typographic ligatures (fi, fl, ffi, ffl) into searchable letters
    - Removes unprintable ASCII/Unicode control characters without corrupting scientific notation
    - Reconnects hyphenated words broken across line wraps (e.g. 'experi-\nment' -> 'experiment')
    - Preserves mathematical negative numbers (e.g. '-1000', '-0.5')
    - Collapses excessive blank lines and whitespace
    """
    if not text:
        return ""

    # 1. Unicode Normalization Form KC (expands typographic ligatures like \ufb01 -> fi, \ufb02 -> fl)
    cleaned = unicodedata.normalize("NFKC", text)

    # 2. Strip unprintable/control characters (leaving standard newlines and tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)

    # 3. Clean floating caron or modifier diacritics attached strangely
    cleaned = cleaned.replace("\u02c7", "")

    # 4. Fix hyphenated word breaks across line boundaries: word-\nword -> wordword
    # Preserves negative numbers like -100 or -0.5
    cleaned = re.sub(r"(\b[a-zA-Z]{2,})-\n([a-zA-Z]{2,}\b)", r"\1\2", cleaned)

    # 5. Collapse excessive whitespace and blank lines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def extract_text_from_pdf(
    pdf_path: Path,
    use_cache: bool = True,
    force_reclean: bool = False,
) -> Dict[str, Any]:
    """
    Extracts text page-by-page from a PDF file using PyMuPDF (fitz) or pypdf.
    Preserves exact 1-based page numbers and source provenance.
    
    Returns:
        {
            "pdf_filename": str,
            "pdf_path": str,
            "total_pages": int,
            "pages": [{"page_number": int, "text": str, "char_count": int}],
            "total_characters": int,
            "is_scanned": bool,
            "error": Optional[str],
            "status": "success" | "warning" | "error"
        }
    """
    path = Path(pdf_path)
    if not path.exists():
        return {
            "pdf_filename": path.name,
            "pdf_path": str(path),
            "total_pages": 0,
            "pages": [],
            "total_characters": 0,
            "is_scanned": False,
            "error": f"File does not exist: {path}",
            "status": "error",
        }

    cache_file = EXTRACTED_TEXT_CACHE_DIR / f"{path.stem}.json"
    if use_cache and not force_reclean and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                # Verify that cache is not containing old uncleaned ligatures
                sample_text = " ".join(p.get("text", "") for p in cached_data.get("pages", [])[:2])
                if not any(c in sample_text for c in ("\ufb00", "\ufb01", "\ufb02", "\ufb03", "\ufb04", "\x15", "\x14")):
                    return cached_data
                logger.info(f"Detected unnormalized ligatures in cache for {path.stem}. Re-cleaning...")
        except Exception as e:
            logger.warning(f"Could not read cached PDF extraction for {path.stem}: {e}")

    result: Dict[str, Any] = {
        "pdf_filename": path.name,
        "pdf_path": str(path),
        "total_pages": 0,
        "pages": [],
        "total_characters": 0,
        "is_scanned": False,
        "error": None,
        "status": "success",
    }

    # Attempt extraction via PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        result["total_pages"] = len(doc)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text("text") or ""
            cleaned = clean_page_text(raw_text)

            if cleaned:
                result["pages"].append({
                    "page_number": page_idx + 1,  # 1-indexed
                    "text": cleaned,
                    "char_count": len(cleaned),
                })
        doc.close()

    except Exception as e_fitz:
        logger.warning(f"PyMuPDF failed on {path.name}: {e_fitz}. Trying fallback extraction...")
        # Fallback to pypdf if available
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            result["total_pages"] = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                cleaned = clean_page_text(raw_text)
                if cleaned:
                    result["pages"].append({
                        "page_number": page_idx + 1,
                        "text": cleaned,
                        "char_count": len(cleaned),
                    })
        except Exception as e_pypdf:
            result["error"] = f"Extraction failed with both engines. Details: {e_fitz}; {e_pypdf}"
            result["status"] = "error"
            return result

    # Calculate total extracted characters
    total_chars = sum(p["char_count"] for p in result["pages"])
    result["total_characters"] = total_chars

    # Detect scanned or image-only PDFs
    if result["total_pages"] > 0 and total_chars < 100:
        result["is_scanned"] = True
        result["status"] = "warning"
        result["error"] = (
            "Document contains little or no extractable digital text. "
            "It appears to be a scanned or image-only PDF, which requires OCR to extract text."
        )
    elif len(result["pages"]) == 0:
        result["is_scanned"] = True
        result["status"] = "warning"
        result["error"] = "No text could be extracted from pages. File may be image-only."

    # Cache successful extraction
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not cache extraction to {cache_file}: {e}")

    return result
