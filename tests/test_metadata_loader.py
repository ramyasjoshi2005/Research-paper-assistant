import json
import pytest
from src.metadata_loader import (
    sanitize_arxiv_id,
    extract_year,
    clean_text,
    parse_metadata_line,
)


def test_sanitize_arxiv_id():
    assert sanitize_arxiv_id("2103.12345v1") == "2103.12345"
    assert sanitize_arxiv_id("2103.12345v2") == "2103.12345"
    assert sanitize_arxiv_id("math.GT/0309136v1") == "math.GT/0309136"
    assert sanitize_arxiv_id("  1706.03762  ") == "1706.03762"


def test_extract_year():
    # From update_date
    assert extract_year({"update_date": "2023-04-12"}) == 2023
    # From versions created
    assert extract_year({"versions": [{"created": "Mon, 2 Apr 2018 19:18:42 GMT"}]}) == 2018
    # From arXiv id
    assert extract_year({"id": "2106.00001"}) == 2021
    assert extract_year({"id": "9901.00001"}) == 1999


def test_clean_text():
    assert clean_text("  Attention\n  Is All\n You Need \n\n ") == "Attention Is All You Need"
    assert clean_text(None) == ""


def test_parse_metadata_line_valid():
    sample_json = json.dumps({
        "id": "1706.03762v5",
        "title": "Attention Is All You Need\n",
        "authors": "Ashish Vaswani, Noam Shazeer",
        "categories": "cs.CL cs.AI",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "update_date": "2017-06-12",
    })
    record = parse_metadata_line(sample_json)
    assert record is not None
    assert record["id"] == "1706.03762"
    assert record["title"] == "Attention Is All You Need"
    assert record["year"] == 2017
    assert "cs.CL" in record["categories"]
    assert record["pdf_url"] == "https://arxiv.org/pdf/1706.03762.pdf"


def test_parse_metadata_line_malformed():
    assert parse_metadata_line("") is None
    assert parse_metadata_line("   ") is None
    assert parse_metadata_line("this is not json { [") is None
    assert parse_metadata_line('{"title": "No ID paper"}') is None

