"""
RAG Pipeline Orchestrator
Coordinates document text extraction, metadata-preserved chunking, embedding generation,
persistent vector storage, hybrid similarity retrieval, and grounded answer synthesis.
Enforces strict single-paper filtering and intent-specific aspect retrieval.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import DEFAULT_TOP_K, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, PDF_DIR
from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import chunk_document_pages
from src.vector_store import get_vector_store
from src.llm_provider import get_llm_client, detect_query_intent
from src.metadata_loader import get_paper_by_id, sanitize_arxiv_id

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline for research papers."""

    def __init__(self):
        self.vector_store = get_vector_store()

    def ingest_paper_pdf(
        self,
        pdf_path: Path,
        paper_metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes a local PDF file into indexed vector store chunks.
        Preserves complete provenance, author attribution, and page mapping.
        """
        path = Path(pdf_path)
        if not path.exists():
            return {
                "success": False,
                "message": f"PDF file not found at: {path}",
                "arxiv_id": "",
                "total_pages": 0,
                "total_chunks": 0,
                "is_scanned": False,
            }

        # Determine clean paper ID from filename if not in metadata
        arxiv_id = sanitize_arxiv_id(path.stem.replace("_", "/"))

        # Check if already indexed
        if not force_reindex and self.vector_store.is_paper_indexed(arxiv_id):
            return {
                "success": True,
                "message": f"Paper '{arxiv_id}' is already indexed in the vector database.",
                "arxiv_id": arxiv_id,
                "total_pages": 0,
                "total_chunks": 0,
                "is_scanned": False,
                "already_indexed": True,
            }

        # Obtain complete metadata
        if not paper_metadata:
            meta = get_paper_by_id(arxiv_id)
            if meta:
                paper_metadata = meta
            else:
                paper_metadata = {
                    "id": arxiv_id,
                    "title": path.stem,
                    "authors": "Not available",
                    "pdf_filename": path.name,
                    "local_pdf_path": str(path),
                    "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "year": None,
                    "categories": [],
                }
        else:
            paper_metadata["pdf_filename"] = path.name
            paper_metadata["local_pdf_path"] = str(path)
            if not paper_metadata.get("abs_url"):
                paper_metadata["abs_url"] = f"https://arxiv.org/abs/{arxiv_id}"
            if not paper_metadata.get("pdf_url"):
                paper_metadata["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Extract text page by page with ligature cleanup
        extraction = extract_text_from_pdf(path)
        if extraction["status"] == "error":
            return {
                "success": False,
                "message": f"PDF extraction failed: {extraction.get('error')}",
                "arxiv_id": arxiv_id,
                "total_pages": extraction.get("total_pages", 0),
                "total_chunks": 0,
                "is_scanned": extraction.get("is_scanned", False),
            }

        if extraction["is_scanned"] or not extraction["pages"]:
            return {
                "success": False,
                "message": (
                    "Document appears to be scanned or image-only with no extractable text. "
                    "OCR would be required to extract text from this document."
                ),
                "arxiv_id": arxiv_id,
                "total_pages": extraction.get("total_pages", 0),
                "total_chunks": 0,
                "is_scanned": True,
            }

        # Chunk pages with full provenance
        chunks = chunk_document_pages(
            extraction["pages"],
            paper_metadata=paper_metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            return {
                "success": False,
                "message": "No valid text chunks could be produced from the document.",
                "arxiv_id": arxiv_id,
                "total_pages": extraction.get("total_pages", 0),
                "total_chunks": 0,
                "is_scanned": False,
            }

        # Ingest into vector store
        num_ingested = self.vector_store.add_paper_chunks(paper_metadata, chunks)

        return {
            "success": True,
            "message": f"Successfully indexed {num_ingested} chunks from {extraction['total_pages']} pages.",
            "arxiv_id": arxiv_id,
            "total_pages": extraction["total_pages"],
            "total_chunks": num_ingested,
            "is_scanned": False,
            "already_indexed": False,
        }

    def query(
        self,
        question: str,
        paper_id_filter: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Executes an intent-aware grounded RAG query:
        1. Enforces strict single-paper filtering when selected.
        2. Detects question intent and targets specific aspects (e.g. findings, limitations).
        3. Retrieves evidence via hybrid ranking (dense semantic + lexical keyword).
        4. Synthesizes answer using configured LLM or extractive fallback with page citations.
        """
        # Strict paper filter: normalize "All" or empty string to None
        target_filter = paper_id_filter.strip() if (paper_id_filter and paper_id_filter != "All") else None

        intents = detect_query_intent(question)

        # Multi-aspect retrieval for questions targeting both findings and limitations
        if intents.get("is_dual_findings_limitations"):
            # 1. Target experimental findings and results
            chunks_findings = self.vector_store.similarity_search(
                f"{question} experimental results empirical findings error accuracy measurements performance test",
                top_k=max(2, top_k // 2),
                paper_id_filter=target_filter,
                min_relevance_threshold=0.04,
            )
            # 2. Target author-reported limitations and constraints
            chunks_limitations = self.vector_store.similarity_search(
                f"{question} limitations drawbacks systematic error smoothing compromise constraints complexity failure",
                top_k=max(2, top_k // 2),
                paper_id_filter=target_filter,
                min_relevance_threshold=0.04,
            )
            
            # Combine and deduplicate
            combined_chunks = []
            seen_ids = set()
            for c in chunks_findings:
                if c["chunk_id"] not in seen_ids:
                    c["aspect_tag"] = "finding"
                    seen_ids.add(c["chunk_id"])
                    combined_chunks.append(c)
            for c in chunks_limitations:
                if c["chunk_id"] not in seen_ids:
                    c["aspect_tag"] = "limitation"
                    seen_ids.add(c["chunk_id"])
                    combined_chunks.append(c)
            retrieved_chunks = combined_chunks

        elif intents.get("limitations"):
            retrieved_chunks = self.vector_store.similarity_search(
                f"{question} limitations drawbacks systematic error smoothing compromise constraints complexity",
                top_k=top_k,
                paper_id_filter=target_filter,
                min_relevance_threshold=0.04,
            )
        elif intents.get("findings"):
            retrieved_chunks = self.vector_store.similarity_search(
                f"{question} experimental findings results empirical evaluation accuracy performance measurements",
                top_k=top_k,
                paper_id_filter=target_filter,
                min_relevance_threshold=0.04,
            )
        else:
            retrieved_chunks = self.vector_store.similarity_search(
                query=question,
                top_k=top_k,
                paper_id_filter=target_filter,
                min_relevance_threshold=0.04,
            )

        # STRICT GUARANTEE: If a single paper filter is active, filter out any chunk from other papers
        if target_filter:
            retrieved_chunks = [c for c in retrieved_chunks if c.get("arxiv_id") == target_filter]

        # Generate grounded response
        llm = get_llm_client()
        result = llm.generate_rag_response(
            query=question,
            retrieved_chunks=retrieved_chunks,
        )

        result["retrieved_chunks"] = retrieved_chunks
        result["query"] = question
        result["filter_paper_id"] = target_filter
        return result


# Global singleton instance
_rag_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Returns singleton instance of RAGPipeline."""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance
