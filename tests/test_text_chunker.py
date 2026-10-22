import pytest
from src.text_chunker import split_text_into_chunks, chunk_document_pages


def test_split_text_into_chunks_basic():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunks = split_text_into_chunks(text, chunk_size=35, chunk_overlap=10)
    assert len(chunks) > 1
    # Ensure chunks are non-empty
    for c in chunks:
        assert len(c.strip()) > 0


def test_chunk_document_pages_metadata_preservation():
    pages = [
        {"page_number": 1, "text": "This is page one text discussing the novel methodology."},
        {"page_number": 2, "text": "This is page two text outlining experimental evaluation results."},
    ]
    meta = {
        "id": "2301.00001",
        "title": "A Great Machine Learning Paper",
        "pdf_filename": "2301.00001.pdf",
        "year": 2023,
        "categories": ["cs.LG", "cs.AI"],
    }

    chunks = chunk_document_pages(pages, meta, chunk_size=40, chunk_overlap=10)
    assert len(chunks) >= 2

    first_chunk = chunks[0]
    assert first_chunk["arxiv_id"] == "2301.00001"
    assert first_chunk["paper_title"] == "A Great Machine Learning Paper"
    assert first_chunk["page_number"] in [1, 2]
    assert "chunk_id" in first_chunk
    assert first_chunk["chunk_id"].startswith("2301.00001_p")

