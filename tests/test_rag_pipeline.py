import pytest
from pathlib import Path
from src.vector_store import PersistentVectorStore
from src.llm_provider import generate_extractive_fallback_answer, LLMClient


def test_vector_store_crud_and_deduplication(tmp_path):
    store = PersistentVectorStore(db_dir=tmp_path)

    paper_meta = {
        "id": "2106.12345",
        "title": "Quantum Deep Learning",
        "pdf_filename": "2106.12345.pdf",
        "year": 2021,
        "categories": ["cs.AI", "quant-ph"],
    }
    chunks = [
        {
            "chunk_id": "2106.12345_p1_c1",
            "text": "Quantum circuits can accelerate optimization in deep neural networks.",
            "page_number": 1,
            "chunk_index": 1,
        },
        {
            "chunk_id": "2106.12345_p2_c1",
            "text": "Experimental benchmark on quantum simulators showed 15% speedup.",
            "page_number": 2,
            "chunk_index": 1,
        },
    ]

    # Ingest
    ingested = store.add_paper_chunks(paper_meta, chunks)
    assert ingested == 2
    assert store.is_paper_indexed("2106.12345") is True

    # Duplicate ingestion test: should replace without multiplying count
    ingested_again = store.add_paper_chunks(paper_meta, chunks)
    assert ingested_again == 2
    stats = store.get_stats()
    assert stats["total_papers_indexed"] == 1
    assert stats["total_chunks_indexed"] == 2

    # Retrieval test
    results = store.similarity_search("quantum optimization circuits", top_k=2)
    assert len(results) > 0
    assert results[0]["arxiv_id"] == "2106.12345"
    assert results[0]["page_number"] in [1, 2]
    assert "text" in results[0]


def test_extractive_fallback_grounded_citations():
    mock_chunks = [
        {
            "chunk_id": "c1",
            "paper_title": "Attention Is All You Need",
            "page_number": 3,
            "arxiv_id": "1706.03762",
            "similarity_score": 0.89,
            "text": "The Transformer allows for significantly more parallelization and can reach a new state of the art.",
        }
    ]

    res = generate_extractive_fallback_answer("What is the benefit of the Transformer?", mock_chunks)
    assert res["is_generative"] is False
    assert res["mode"] == "extractive_fallback"
    assert len(res["citations"]) == 1
    assert res["citations"][0]["paper_title"] == "Attention Is All You Need"
    assert res["citations"][0]["page_number"] == 3
    assert "Transformer allows for significantly more parallelization" in res["answer"]


def test_missing_api_key_graceful_handling():
    client = LLMClient(provider="gemini", gemini_api_key="")
    assert client.has_api_key() is False

    mock_chunks = [
        {
            "paper_title": "Paper A",
            "page_number": 1,
            "arxiv_id": "0001",
            "similarity_score": 0.75,
            "text": "Some verified research text.",
        }
    ]
    # Should not throw exception, should return extractive answer
    response = client.generate_rag_response("What does Paper A do?", mock_chunks)
    assert response["is_generative"] is False
    assert "Paper A" in response["answer"]


def test_single_paper_filter_enforcement_and_aspect_sections(tmp_path):
    store = PersistentVectorStore(db_dir=tmp_path)

    # Ingest Paper A
    meta_a = {"id": "0704.0047", "title": "Acoustic Locator Part I", "authors": "T. Kosel", "year": 2007}
    chunks_a = [
        {"chunk_id": "0704.0047_p4", "text": "Experimental findings showed location error ranges from 1.3 mm to 60 mm with average 20 mm.", "page_number": 4},
        {"chunk_id": "0704.0047_p5", "text": "Estimation is subject to systematic error caused by smoothing. Compromise must be found.", "page_number": 5},
    ]
    store.add_paper_chunks(meta_a, chunks_a)

    # Ingest Paper B
    meta_b = {"id": "0706.0022", "title": "Semantic Network", "authors": "M. Rodriguez", "year": 2007}
    chunks_b = [
        {"chunk_id": "0706.0022_p2", "text": "Semantic network algorithm evaluation across vertices and graph algorithms.", "page_number": 2},
    ]
    store.add_paper_chunks(meta_b, chunks_b)

    # Query strictly targeting Paper A
    retrieved = store.similarity_search("What are the experimental findings and limitations?", top_k=4, paper_id_filter="0704.0047")
    
    # Assertions:
    # 1. Chunks from Paper B must NEVER appear
    for c in retrieved:
        assert c["arxiv_id"] == "0704.0047"
        assert c["arxiv_id"] != "0706.0022"

    # 2. Extractive answer structure check
    ans_data = generate_extractive_fallback_answer("What are the main experimental findings, and what limitations do the authors report?", retrieved)
    ans_text = ans_data["answer"]
    assert "Main Experimental Findings" in ans_text
    assert "Author-Reported Limitations" in ans_text
    assert "Source Papers Used in This Answer" in ans_text
    assert "0704.0047" in ans_text
    assert "0706.0022" not in ans_text

