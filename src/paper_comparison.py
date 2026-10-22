"""
Paper Comparison Module
Performs comparative analysis between two indexed research papers.
Produces a 10-dimension comparison matrix grounded strictly in retrieved evidence.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from src.vector_store import get_vector_store
from src.llm_provider import get_llm_client

logger = logging.getLogger(__name__)

COMPARISON_CRITERIA = [
    ("Research Objective", "core objective and primary goal of the research"),
    ("Problem Addressed", "specific problem, challenge, or bottleneck being solved"),
    ("Proposed Methodology", "algorithms, model architecture, or theoretical framework"),
    ("Datasets Used", "benchmark datasets, training data, or experimental corpora"),
    ("Evaluation Metrics", "quantitative or qualitative evaluation metrics (accuracy, BLEU, F1, latency)"),
    ("Main Findings", "empirical performance results and key discoveries"),
    ("Limitations", "known drawbacks, failure modes, assumptions, or scalability constraints"),
    ("Key Similarities", "shared approaches, assumptions, or problem spaces"),
    ("Key Differences", "contrasting paradigms, trade-offs, or structural variations"),
    ("Potential Research Gaps", "unexplored intersections, combined opportunities, or open questions"),
]


def compare_two_papers(
    paper_id_a: str,
    paper_id_b: str,
) -> Dict[str, Any]:
    """
    Compares two indexed papers across 10 evaluation criteria.
    Returns comparison matrix, pandas DataFrame, and source citations.
    """
    vector_store = get_vector_store()
    llm = get_llm_client()

    if not vector_store.is_paper_indexed(paper_id_a):
        return {"success": False, "error": f"Paper '{paper_id_a}' is not yet indexed."}
    if not vector_store.is_paper_indexed(paper_id_b):
        return {"success": False, "error": f"Paper '{paper_id_b}' is not yet indexed."}

    # Retrieve representative chunks for both papers
    chunks_a = vector_store.similarity_search("method problem results dataset limitations", top_k=4, paper_id_filter=paper_id_a)
    chunks_b = vector_store.similarity_search("method problem results dataset limitations", top_k=4, paper_id_filter=paper_id_b)

    title_a = chunks_a[0]["paper_title"] if chunks_a else paper_id_a
    title_b = chunks_b[0]["paper_title"] if chunks_b else paper_id_b

    # Generative comparison if LLM available
    if llm.has_api_key():
        text_a = "\n".join([f"[Page {c['page_number']}]: {c['text']}" for c in chunks_a])
        text_b = "\n".join([f"[Page {c['page_number']}]: {c['text']}" for c in chunks_b])

        prompt = f"""You are an expert AI comparative literature reviewer.
Compare these two research papers rigorously using only the provided text excerpts.

PAPER A: {title_a} (arXiv: {paper_id_a})
EXCERPTS A:
{text_a}

PAPER B: {title_b} (arXiv: {paper_id_b})
EXCERPTS B:
{text_b}

CRITERIA TO COMPARE:
1. Research Objective
2. Problem Addressed
3. Proposed Methodology
4. Datasets
5. Evaluation Metrics
6. Main Findings
7. Limitations
8. Key Similarities
9. Key Differences
10. Potential Research Gaps

INSTRUCTIONS:
- For each criterion, provide a concise comparative evaluation for Paper A and Paper B.
- If information is not in the excerpts, explicitly state "Not reported in the indexed text".
- Do not fabricate metrics, datasets, or claims.
- Return your answer as a clear Markdown Table with columns:
  | Evaluation Dimension | Paper A: {title_a[:30]} | Paper B: {title_b[:30]} | Comparative Analysis |
"""
        response = llm.generate_rag_response(
            query=f"Compare {paper_id_a} and {paper_id_b}",
            retrieved_chunks=chunks_a + chunks_b,
        )
        return {
            "success": True,
            "title_a": title_a,
            "title_b": title_b,
            "paper_id_a": paper_id_a,
            "paper_id_b": paper_id_b,
            "mode": response.get("mode", "generative_rag"),
            "provider": response.get("provider", "LLM"),
            "markdown_table": response.get("answer", ""),
            "citations_a": chunks_a,
            "citations_b": chunks_b,
        }

    # Extractive structured fallback
    comparison_rows = []
    for dimension, query_desc in COMPARISON_CRITERIA:
        res_a = vector_store.similarity_search(query_desc, top_k=1, paper_id_filter=paper_id_a)
        res_b = vector_store.similarity_search(query_desc, top_k=1, paper_id_filter=paper_id_b)

        val_a = (
            f"{res_a[0]['text'][:160]}... (p. {res_a[0]['page_number']})"
            if res_a and res_a[0].get("similarity_score", 0) > 0.05
            else "Not reported in indexed text"
        )
        val_b = (
            f"{res_b[0]['text'][:160]}... (p. {res_b[0]['page_number']})"
            if res_b and res_b[0].get("similarity_score", 0) > 0.05
            else "Not reported in indexed text"
        )

        comparison_rows.append({
            "Dimension": dimension,
            f"Paper A ({paper_id_a})": val_a,
            f"Paper B ({paper_id_b})": val_b,
        })

    df = pd.DataFrame(comparison_rows)

    # Format Markdown
    md_lines = [
        f"## Comparative Matrix: {title_a} vs. {title_b}\n",
        "*Generated via Extractive Retrieval (Zero API Key Mode)*\n",
        df.to_markdown(index=False),
    ]

    return {
        "success": True,
        "title_a": title_a,
        "title_b": title_b,
        "paper_id_a": paper_id_a,
        "paper_id_b": paper_id_b,
        "mode": "extractive_fallback",
        "provider": "Local Extractive Engine",
        "markdown_table": "\n".join(md_lines),
        "dataframe": df,
        "citations_a": chunks_a,
        "citations_b": chunks_b,
    }

