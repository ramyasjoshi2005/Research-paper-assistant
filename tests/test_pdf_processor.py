from pathlib import Path
import pytest
from src.pdf_processor import clean_page_text, extract_text_from_pdf


def test_clean_page_text():
    raw = "The trans-\nformer architecture replaces recur-\nrent neural networks.\n\n\n\nResult:"
    cleaned = clean_page_text(raw)
    assert "transformer" in cleaned
    assert "recurrent" in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_nonexistent_pdf(tmp_path):
    fake_path = tmp_path / "nonexistent.pdf"
    result = extract_text_from_pdf(fake_path, use_cache=False)
    assert result["status"] == "error"
    assert "does not exist" in result["error"]
    assert result["total_pages"] == 0


def test_extract_real_pdf_or_mock(tmp_path):
    # Test creating a minimal valid PDF using PyMuPDF if available
    try:
        import fitz
        sample_pdf = tmp_path / "test_sample.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            "Retrieval-Augmented Generation (RAG) combines dense semantic retrieval with generative neural networks "
            "to provide verifiable citations for complex academic literature search and question answering."
        )
        doc.save(str(sample_pdf))
        doc.close()

        result = extract_text_from_pdf(sample_pdf, use_cache=False)
        assert result["status"] == "success"
        assert result["total_pages"] == 1
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_number"] == 1
        assert "Retrieval-Augmented" in result["pages"][0]["text"]
        assert result["is_scanned"] is False
    except ImportError:
        pytest.skip("PyMuPDF not installed")
