"""
Paper Summarizer Module
Generates structured, academic summaries grounded strictly in the paper's indexed text.
Follows the 8-point research paper evaluation structure.
"""

import logging
from typing import Dict, Any, Optional, List

from src.vector_store import get_vector_store
from src.llm_provider import get_llm_client

logger = logging.getLogger(__name__)

SUMMARY_SECTIONS = [
    ("Research Problem", "What specific challenge, gap, or question does this research investigate?"),
    ("Motivation", "Why is this problem important and what motivated the authors?"),
    ("Proposed Method", "What model, algorithm, framework, or theoretical methodology is introduced?"),
    ("Dataset & Experimental Setup", "What datasets, benchmarks, baselines, or hardware setup are used?"),
    ("Main Findings", "What were the key results, improvements, or empirical discoveries?"),
    ("Limitations", "What shortcomings, failure cases, or assumptions do the authors disclose?"),
    ("Future Research Directions", "What next steps or open questions do the authors propose?"),
    ("Important Terminology", "Key domain-specific concepts, acronyms, and formulas defined in the paper."),
]


def summarize_paper(
    arxiv_id: str,
    paper_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Produces an 8-section structured summary of an indexed paper.
    If LLM API is available, synthesizes an executive summary grounded in evidence.
    Otherwise, builds an extractive structured summary based on targeted query retrieval.
    """
    vector_store = get_vector_store()
    llm = get_llm_client()

    # Verify paper is indexed
    if not vector_store.is_paper_indexed(arxiv_id):
        return {
            "success": False,
            "error": f"Paper '{arxiv_id}' has not been indexed yet. Please index it first from My Library.",
            "sections": {},
        }

    # Retrieve broad evidence chunks from introductory, methodological, and concluding sections
    query_topics = [
        "problem motivation proposed method architecture",
        "experiments dataset results evaluation findings",
        "limitations future work conclusion",
    ]
    
    collected_chunks: List[Dict[str, Any]] = []
    seen_chunk_ids = set()

    for q in query_topics:
        chunks = vector_store.similarity_search(query=q, top_k=3, paper_id_filter=arxiv_id)
        for c in chunks:
            if c["chunk_id"] not in seen_chunk_ids:
                seen_chunk_ids.add(c["chunk_id"])
                collected_chunks.append(c)

    if not collected_chunks:
        return {
            "success": False,
            "error": "No indexed chunks found for this paper.",
            "sections": {},
        }

    paper_title = collected_chunks[0].get("paper_title", f"Paper {arxiv_id}")

    # If LLM API key is present, generate fluent academic summary
    if llm.has_api_key():
        context_text = "\n\n".join([
            f"[Page {c['page_number']}]: {c['text']}" for c in collected_chunks
        ])

        prompt = f"""You are a senior academic researcher. Provide a comprehensive, structured summary of the following research paper.

PAPER TITLE: {paper_title} (arXiv: {arxiv_id})

RULES:
1. Ground every statement strictly in the provided paper excerpts.
2. If any aspect is not discussed in the excerpts, write "Not reported in the indexed text".
3. Use the exact 8 section headers below.

PAPER EXCERPTS:
{context_text}

Provide your summary in this exact format:

### 1. Research Problem
<Answer or 'Not reported in the indexed text'>

### 2. Motivation
<Answer or 'Not reported in the indexed text'>

### 3. Proposed Method
<Answer or 'Not reported in the indexed text'>

### 4. Dataset and Experimental Setup
<Answer or 'Not reported in the indexed text'>

### 5. Main Findings
<Answer or 'Not reported in the indexed text'>

### 6. Limitations
<Answer or 'Not reported in the indexed text'>

### 7. Future Research Directions
<Answer or 'Not reported in the indexed text'>

### 8. Important Terminology
<Answer or 'Not reported in the indexed text'>
"""
        response = llm.generate_rag_response(
            query=f"Summarize the paper '{paper_title}'",
            retrieved_chunks=collected_chunks,
        )
        return {
            "success": True,
            "paper_title": paper_title,
            "arxiv_id": arxiv_id,
            "mode": response.get("mode", "generative_rag"),
            "provider": response.get("provider", "LLM"),
            "summary_text": response.get("answer", ""),
            "citations": response.get("citations", []),
            "total_evidence_chunks": len(collected_chunks),
        }

    # Extractive structured fallback
    structured_sections = {}
    for header, desc in SUMMARY_SECTIONS:
        # Retrieve most relevant chunk for each section
        top_chunk = vector_store.similarity_search(query=f"{header} {desc}", top_k=1, paper_id_filter=arxiv_id)
        if top_chunk and top_chunk[0].get("similarity_score", 0) > 0.05:
            tc = top_chunk[0]
            structured_sections[header] = {
                "content": tc["text"],
                "page": tc["page_number"],
                "score": tc["similarity_score"],
            }
        else:
            structured_sections[header] = {
                "content": "Not reported in the indexed text passages.",
                "page": None,
                "score": 0.0,
            }

    # Build formatted markdown
    from src.metadata_loader import get_paper_by_id
    meta = get_paper_by_id(arxiv_id) or {}
    authors = meta.get("authors", "Not available")
    year = meta.get("year", "Not available")
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    md_lines = [
        f"## Structured Research Summary: {paper_title}\n",
        f"*Authors:* {authors} ({year}) &nbsp;|&nbsp; *arXiv:* [`{arxiv_id}`]({abs_url}) &nbsp;|&nbsp; [📄 Direct PDF]({pdf_url})\n",
        "*Generated via Grounded Extractive Retrieval (Zero API Key Mode)*\n",
    ]
    for idx, (header, _) in enumerate(SUMMARY_SECTIONS, 1):
        sec = structured_sections[header]
        page_ref = f" *(Source: Page {sec['page']})*" if sec["page"] else ""
        md_lines.append(f"### {idx}. {header}{page_ref}")
        md_lines.append(f"{sec['content']}\n")

    md_lines.append("\n---\n### 📚 Paper Provenance")
    md_lines.append(f"- **Full Title:** {paper_title}")
    md_lines.append(f"- **Authors:** {authors}")
    md_lines.append(f"- **Publication Year:** {year}")
    md_lines.append(f"- **Canonical References:** [arXiv Abstract Page]({abs_url}) · [Original PDF Document]({pdf_url})")

    return {
        "success": True,
        "paper_title": paper_title,
        "arxiv_id": arxiv_id,
        "mode": "extractive_fallback",
        "provider": "Local Extractive Engine",
        "summary_text": "\n".join(md_lines),
        "citations": [
            {
                "citation_id": f"[{i+1}]",
                "paper_title": paper_title,
                "page_number": c["page_number"],
                "arxiv_id": arxiv_id,
                "similarity_score": c["similarity_score"],
                "snippet": c["text"][:180] + "...",
            }
            for i, c in enumerate(collected_chunks)
        ],
        "total_evidence_chunks": len(collected_chunks),
    }

