"""
LLM Provider Module
Handles answer generation through configurable LLMs (Gemini, OpenAI)
and includes an intelligent, transparent Extractive Fallback for offline/zero-API-key usage.
Detects question intent (findings, limitations, methodology, etc.), structures responses accordingly,
and enforces comprehensive source attribution with page numbers and canonical paper links.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from src.config import (
    LLM_PROVIDER,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    GEMINI_MODEL,
    OPENAI_MODEL,
    get_config_val,
)

logger = logging.getLogger(__name__)


def detect_query_intent(query: str) -> Dict[str, bool]:
    """
    Detects question intent to provide targeted, structured academic answers.
    Detects: findings, limitations, methodology, problem, datasets, simple explanation.
    """
    q = query.lower()
    
    has_findings = any(w in q for w in [
        "finding", "findings", "result", "results", "discover", "discovered",
        "performance", "accuracy", "observe", "observed", "demonstrate", "empirical", "conclusion"
    ])
    has_limitations = any(w in q for w in [
        "limit", "limits", "limitation", "limitations", "drawback", "drawbacks",
        "shortcoming", "weakness", "error", "failure", "assumption", "constraints", "future work"
    ])
    has_methodology = any(w in q for w in [
        "method", "methodology", "algorithm", "architecture", "model", "approach",
        "framework", "technique", "pipeline", "formulation", "how does"
    ])
    has_problem = any(w in q for w in [
        "problem", "solve", "challenge", "motivation", "address", "why", "goal", "objective"
    ])
    has_datasets = any(w in q for w in [
        "dataset", "datasets", "benchmark", "metrics", "data", "corpus", "evaluation", "baseline"
    ])
    has_simple_explanation = any(w in q for w in [
        "simple", "explain", "beginner", "student", "overview", "summary", "what is"
    ])

    return {
        "findings": has_findings,
        "limitations": has_limitations,
        "methodology": has_methodology,
        "problem": has_problem,
        "datasets": has_datasets,
        "simple": has_simple_explanation,
        "is_dual_findings_limitations": has_findings and has_limitations,
    }


def format_source_papers_section(chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs a deduplicated 'Source Papers Used in This Answer' section
    for verifiable academic citation and provenance.
    Only lists papers that actually contributed retrieved evidence.
    """
    if not chunks:
        return ""

    seen_papers = {}
    for c in chunks:
        pid = c.get("arxiv_id", "unknown")
        if pid not in seen_papers:
            seen_papers[pid] = {
                "title": c.get("paper_title", f"arXiv:{pid}"),
                "authors": c.get("authors", "Not available"),
                "year": c.get("publication_date", "Not available"),
                "abs_url": c.get("abs_url", f"https://arxiv.org/abs/{pid}" if pid != "unknown" else ""),
                "pdf_url": c.get("pdf_url", f"https://arxiv.org/pdf/{pid}.pdf" if pid != "unknown" else ""),
                "local_pdf": c.get("pdf_filename", f"{pid}.pdf"),
            }

    lines = [
        "\n\n---\n### 📚 Source Papers Used in This Answer\n",
    ]
    for pid, p in seen_papers.items():
        links = []
        if p["abs_url"]:
            links.append(f"[arXiv Abstract]({p['abs_url']})")
        if p["pdf_url"]:
            links.append(f"[Direct PDF]({p['pdf_url']})")
        links_str = f" | {' · '.join(links)}" if links else ""

        lines.append(
            f"- **{p['title']}**\n"
            f"  *Authors:* {p['authors']} &nbsp;|&nbsp; *Year:* {p['year']} &nbsp;|&nbsp; *arXiv:* `{pid}`{links_str}\n"
        )

    return "\n".join(lines)


def generate_extractive_fallback_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Builds a structured, transparent answer directly from retrieved passages
    when no generative LLM API key is configured or when the API is unreachable.
    Does NOT pretend to be an LLM; formats evidence with verifiable page citations.
    """
    if not retrieved_chunks:
        return {
            "answer": (
                "**No Relevant Passages Found**\n\n"
                "The indexed papers in your library do not contain sufficient evidence matching your query. "
                "Try searching for alternative keywords or indexing additional papers on this topic."
            ),
            "mode": "extractive_fallback",
            "provider": "Local Extractive Engine",
            "is_generative": False,
            "citations": [],
        }

    intents = detect_query_intent(query)
    citations = []

    # Format structured citations
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        title = chunk.get("paper_title", "Unknown Paper")
        authors = chunk.get("authors", "Not available")
        page = chunk.get("page_number", "?")
        score = chunk.get("similarity_score", 0.0)
        arxiv_id = chunk.get("arxiv_id", "")
        text = chunk.get("text", "").strip()
        abs_url = chunk.get("abs_url", f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
        pdf_url = chunk.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "")
        year = chunk.get("publication_date", "Not available")

        # Format first author surname
        first_author = authors.split(",")[0].split(" and ")[0].strip()
        author_ref = f"{first_author} et al." if "," in authors or " and " in authors else first_author

        citation_marker = f"[{idx}] {title} — {author_ref}, {year}, PDF p. {page}"
        citations.append({
            "citation_id": f"[{idx}]",
            "citation_label": citation_marker,
            "paper_title": title,
            "authors": authors,
            "page_number": page,
            "arxiv_id": arxiv_id,
            "abs_url": abs_url,
            "pdf_url": pdf_url,
            "similarity_score": score,
            "snippet": text[:200] + ("..." if len(text) > 200 else ""),
            "full_text": text,
        })

    # Group passages by intent if dual question
    body_sections = []
    
    if intents["is_dual_findings_limitations"]:
        finding_chunks = [c for c in citations if any(w in c["full_text"].lower() for w in ["result", "finding", "test", "rmse", "perform", "error", "show", "observed", "conclu"])]
        limitation_chunks = [c for c in citations if any(w in c["full_text"].lower() for w in ["limit", "error", "compromise", "systematic", "difficulty", "drawback", "smoothing", "trial"])]

        body_sections.append("#### 📊 Main Experimental Findings")
        if finding_chunks:
            for c in finding_chunks[:3]:
                body_sections.append(
                    f"- **{c['citation_label']}** (Confidence: {c['similarity_score']*100:.1f}%)\n"
                    f"  > \"{c['full_text']}\"\n"
                    f"  [arXiv Page]({c['abs_url']}) | [Direct PDF (p. {c['page_number']})]({c['pdf_url']})\n"
                )
        else:
            body_sections.append("*Specific quantitative findings were not established in the available indexed text passages for this paper.*\n")

        body_sections.append("#### ⚠️ Author-Reported Limitations")
        if limitation_chunks:
            for c in limitation_chunks[:3]:
                body_sections.append(
                    f"- **{c['citation_label']}** (Confidence: {c['similarity_score']*100:.1f}%)\n"
                    f"  > \"{c['full_text']}\"\n"
                    f"  [arXiv Page]({c['abs_url']}) | [Direct PDF (p. {c['page_number']})]({c['pdf_url']})\n"
                )
        else:
            body_sections.append("*The available indexed text does not establish any author-reported limitations. (Note: Not all papers explicitly disclose limitations in dedicated sections).*\n")

    elif intents["limitations"]:
        body_sections.append("#### ⚠️ Disclosed Limitations & Assumptions")
        for c in citations:
            body_sections.append(
                f"- **{c['citation_label']}** (Match: {c['similarity_score']*100:.1f}%)\n"
                f"  > \"{c['full_text']}\"\n"
                f"  [arXiv Page]({c['abs_url']}) | [Direct PDF (p. {c['page_number']})]({c['pdf_url']})\n"
            )

    elif intents["findings"]:
        body_sections.append("#### 📊 Main Findings & Discoveries")
        for c in citations:
            body_sections.append(
                f"- **{c['citation_label']}** (Match: {c['similarity_score']*100:.1f}%)\n"
                f"  > \"{c['full_text']}\"\n"
                f"  [arXiv Page]({c['abs_url']}) | [Direct PDF (p. {c['page_number']})]({c['pdf_url']})\n"
            )

    elif intents["methodology"]:
        body_sections.append("#### ⚙️ Proposed Methodology & Technical Approach")
        for c in citations:
            body_sections.append(
                f"- **{c['citation_label']}** (Match: {c['similarity_score']*100:.1f}%)\n"
                f"  > \"{c['full_text']}\"\n"
                f"  [arXiv Page]({c['abs_url']}) | [Direct PDF (p. {c['page_number']})]({c['pdf_url']})\n"
            )

    else:
        body_sections.append("#### 📋 Retrieved Evidence Passages")
        for c in citations:
            body_sections.append(
                f"- **{c['citation_label']}** (Match: {c['similarity_score']*100:.1f}%)\n"
                f"  > \"{c['full_text']}\"\n"
                f"  [arXiv Page]({c['abs_url']}) | [Direct PDF (p. {c['page_number']})]({c['pdf_url']})\n"
            )

    # Append Source Papers section
    source_papers_md = format_source_papers_section(retrieved_chunks)

    answer_text = (
        "### ⚠️ Local Extractive Mode (Grounded Evidence Passages)\n\n"
        "*Notice: This response presents verbatim evidence passages retrieved directly from your selected paper(s) with page-level citations. "
        "It is strictly extractive rather than a generative AI summary. To enable fluent generative synthesis, configure a free Gemini API key in Settings.*\n\n"
        f"**Query:** *\"{query}\"*\n\n"
        + "\n".join(body_sections)
        + source_papers_md
    )

    return {
        "answer": answer_text,
        "mode": "extractive_fallback",
        "provider": "Local Extractive Engine",
        "is_generative": False,
        "citations": citations,
    }


def call_gemini(
    prompt: str,
    api_key: str,
    model_name: str = GEMINI_MODEL,
) -> str:
    """Calls Google Gemini API safely with automatic model alias resolution."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    # Candidate model aliases for maximum compatibility across SDK versions
    clean_name = model_name.replace("models/", "")
    candidate_models = [
        clean_name,
        f"{clean_name}-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-pro",
    ]

    last_error = None
    for model_candidate in candidate_models:
        try:
            model = genai.GenerativeModel(model_candidate)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1600,
                },
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            continue

    raise last_error or ValueError("Gemini returned an empty response.")


def call_openai(
    prompt: str,
    api_key: str,
    model_name: str = OPENAI_MODEL,
) -> str:
    """Calls OpenAI API."""
    import urllib.request
    import json

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a rigorous academic AI Research Assistant. "
                    "Answer questions strictly grounded in the provided retrieved passages. "
                    "Cite sources using [Paper Title, Page X] and never invent claims or statistics."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        return res_data["choices"][0]["message"]["content"].strip()


class LLMClient:
    """Configurable client managing LLM API interactions with automatic extractive fallback."""

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        gemini_api_key: str = GEMINI_API_KEY,
        openai_api_key: str = OPENAI_API_KEY,
    ):
        self.provider = provider
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key

    def has_api_key(self) -> bool:
        """Checks if a valid API key is present for the active provider."""
        if self.provider == "gemini" and self.gemini_api_key:
            return True
        if self.provider == "openai" and self.openai_api_key:
            return True
        return False

    def generate_rag_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Coordinates generating a grounded RAG response.
        If API key is missing or call fails, falls back gracefully to extractive mode.
        """
        if not retrieved_chunks:
            return {
                "answer": (
                    "**No Relevant Information Found**\n\n"
                    "The indexed papers do not contain sufficient evidence to answer this question. "
                    "Please select or download additional relevant papers."
                ),
                "mode": "no_context",
                "provider": "none",
                "is_generative": False,
                "citations": [],
            }

        # Build citations list
        citations = []
        for i, c in enumerate(retrieved_chunks, start=1):
            title = c.get("paper_title", "Unknown")
            authors = c.get("authors", "Not available")
            page = c.get("page_number", "?")
            year = c.get("publication_date", "Not available")
            first_author = authors.split(",")[0].split(" and ")[0].strip()
            author_ref = f"{first_author} et al." if "," in authors or " and " in authors else first_author

            citations.append({
                "citation_id": f"[{i}]",
                "citation_label": f"[{i}] {title} — {author_ref}, {year}, PDF p. {page}",
                "paper_title": title,
                "authors": authors,
                "page_number": page,
                "arxiv_id": c.get("arxiv_id", ""),
                "abs_url": c.get("abs_url", ""),
                "pdf_url": c.get("pdf_url", ""),
                "similarity_score": c.get("similarity_score", 0.0),
                "snippet": c.get("text", "")[:180] + ("..." if len(c.get("text", "")) > 180 else ""),
            })

        # If no API key configured, use extractive synthesis
        if not self.has_api_key():
            return generate_extractive_fallback_answer(query, retrieved_chunks)

        # Build grounded academic prompt with strict citation requirements
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"--- EVIDENCE SOURCE [{i}] ---\n"
                f"Paper Title: {chunk.get('paper_title')}\n"
                f"Authors: {chunk.get('authors', 'Not available')} | Year: {chunk.get('publication_date', 'Not available')}\n"
                f"arXiv ID: {chunk.get('arxiv_id')} | PDF Page Number: {chunk.get('page_number')}\n"
                f"Text Content:\n{chunk.get('text')}\n"
            )

        context_str = "\n".join(context_blocks)
        source_papers_md = format_source_papers_section(retrieved_chunks)

        prompt = f"""You are a senior academic research assistant.
Answer the user's question strictly grounded in the research paper evidence provided below.

RULES:
1. Ground every claim directly in the provided evidence. Cite sources using:
   [{chunk.get('paper_title', 'Paper')}, Page {chunk.get('page_number', '?')}] or inline numbers like [1], [2].
2. If the user asks about specific aspects (e.g. findings, limitations, methodology), structure your answer with clear markdown headings for each aspect.
3. If the provided evidence does not contain information on any requested aspect (e.g. limitations), explicitly state:
   "Not reported in the indexed text passages for this paper." Never fabricate findings or limitations.
4. Clearly distinguish between direct empirical evidence from the paper and model inference.
5. Never invent author names, benchmark numbers, equations, or page citations.

RETRIEVED RESEARCH EVIDENCE:
{context_str}

USER QUESTION:
{query}

STRUCTURED ACADEMIC ANSWER:"""

        try:
            if self.provider == "gemini":
                raw_answer = call_gemini(prompt, self.gemini_api_key)
                provider_name = f"Google Gemini ({GEMINI_MODEL})"
            elif self.provider == "openai":
                raw_answer = call_openai(prompt, self.openai_api_key)
                provider_name = f"OpenAI ({OPENAI_MODEL})"
            else:
                return generate_extractive_fallback_answer(query, retrieved_chunks)

            # Append source papers section to ensure complete attribution
            full_answer = raw_answer + source_papers_md

            return {
                "answer": full_answer,
                "mode": "generative_rag",
                "provider": provider_name,
                "is_generative": True,
                "citations": citations,
            }

        except Exception as e:
            logger.error(f"LLM API call failed ({self.provider}): {e}. Falling back to extractive engine.")
            fallback = generate_extractive_fallback_answer(query, retrieved_chunks)
            fallback["warning"] = f"LLM API error ({type(e).__name__}: {str(e)}). Switched to extractive fallback."
            return fallback


# Global singleton instance
_llm_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Returns singleton instance of LLMClient, refreshing keys safely from environment or st.secrets."""
    global _llm_client_instance
    _llm_client_instance = LLMClient(
        provider=get_config_val("LLM_PROVIDER", "gemini").lower(),
        gemini_api_key=get_config_val("GEMINI_API_KEY", ""),
        openai_api_key=get_config_val("OPENAI_API_KEY", ""),
    )
    return _llm_client_instance
