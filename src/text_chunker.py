"""
Text Chunker Module
Splits document text into overlapping chunks while strictly preserving all source metadata
and provenance details for verifiable academic citations.
Breaks at sentence or paragraph boundaries wherever possible.
"""

import re
from typing import List, Dict, Any, Optional
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, PDF_DIR


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Splits text into overlapping chunks, attempting to break cleanly at sentence
    boundaries or whitespace.
    """
    if not text:
        return []

    # If text is already shorter than chunk_size, return it directly
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Look for sentence boundary (.!?) within the last 25% of the slice
        slice_zone = text[start:end]
        lookback_window = int(chunk_size * 0.25)
        search_start = max(0, len(slice_zone) - lookback_window)
        search_zone = slice_zone[search_start:]

        # Find best break point (paragraph > sentence end > space)
        split_idx = -1

        para_match = [m.start() for m in re.finditer(r"\n\n", search_zone)]
        if para_match:
            split_idx = search_start + para_match[-1] + 2
        else:
            sent_match = [m.start() for m in re.finditer(r"[.!?]\s+", search_zone)]
            if sent_match:
                split_idx = search_start + sent_match[-1] + 1
            else:
                space_match = [m.start() for m in re.finditer(r"\s+", search_zone)]
                if space_match:
                    split_idx = search_start + space_match[-1]

        if split_idx != -1 and split_idx > 0:
            actual_end = start + split_idx
        else:
            actual_end = end

        chunk = text[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start with overlap
        start = max(start + 1, actual_end - chunk_overlap)

    return chunks


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    paper_metadata: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunks extracted pages from a PDF while preserving all source provenance metadata
    for verifiable RAG citations.
    
    Args:
        pages: List of {"page_number": int, "text": str}
        paper_metadata: Dict containing paper info (title, id, year, authors, urls, etc.)
    """
    arxiv_id = paper_metadata.get("id", "unknown")
    title = paper_metadata.get("title", "Untitled Paper")
    authors = paper_metadata.get("authors", "Not available")
    pdf_filename = paper_metadata.get("pdf_filename", f"{arxiv_id}.pdf")
    year = paper_metadata.get("year")
    categories = paper_metadata.get("categories", [])
    abs_url = paper_metadata.get("abs_url", f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id != "unknown" else "")
    pdf_url = paper_metadata.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id != "unknown" else "")
    local_pdf_path = paper_metadata.get("local_pdf_path", str(PDF_DIR / pdf_filename))

    all_chunks: List[Dict[str, Any]] = []

    for page in pages:
        page_num = page.get("page_number", 1)
        raw_text = page.get("text", "")
        if not raw_text.strip():
            continue

        text_pieces = split_text_into_chunks(
            raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_idx, piece in enumerate(text_pieces):
            chunk_id = f"{arxiv_id}_p{page_num}_c{chunk_idx+1}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": piece,
                "page_number": page_num,
                "chunk_index": chunk_idx + 1,
                "arxiv_id": arxiv_id,
                "paper_title": title,
                "authors": authors,
                "pdf_filename": pdf_filename,
                "local_pdf_path": local_pdf_path,
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "publication_date": str(year) if year else "Not available",
                "year": year,
                "categories": categories,
                "char_length": len(piece),
            })

    return all_chunks
