"""
Paper Search Module
Provides filtering, keyword scoring, relevance ranking, and multi-field search
over arXiv research paper metadata.
"""

from typing import List, Dict, Any, Optional
import re


def search_papers(
    papers: List[Dict[str, Any]],
    query: str = "",
    category: Optional[str] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    sort_by: str = "relevance",  # "relevance", "date_desc", "date_asc"
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Filters and ranks papers from a list of metadata records.
    
    Args:
        papers: List of paper metadata dictionaries
        query: Search keywords or phrases
        category: arXiv category filter (e.g. 'cs.AI')
        min_year: Minimum publication/update year
        max_year: Maximum publication/update year
        sort_by: Sorting strategy ('relevance', 'date_desc', 'date_asc')
        max_results: Maximum papers to return
    """
    if not papers:
        return []

    query_terms = [t.lower() for t in re.findall(r"\w+", query)] if query else []

    filtered: List[Dict[str, Any]] = []

    for paper in papers:
        # Category filter
        if category and category != "All":
            paper_cats = paper.get("categories", [])
            if not any(cat.startswith(category) or cat == category for cat in paper_cats):
                continue

        # Year filter
        paper_year = paper.get("year")
        if min_year and paper_year and paper_year < min_year:
            continue
        if max_year and paper_year and paper_year > max_year:
            continue

        # Text search scoring
        score = 0.0
        if query_terms:
            title_lower = paper.get("title", "").lower()
            abstract_lower = paper.get("abstract", "").lower()
            authors_lower = paper.get("authors", "").lower()
            id_lower = paper.get("id", "").lower()

            matched_all_terms = True
            for term in query_terms:
                term_score = 0.0
                if term in id_lower:
                    term_score += 15.0
                if term in title_lower:
                    # Title matches are weighted significantly higher
                    term_score += 10.0
                if term in authors_lower:
                    term_score += 4.0
                if term in abstract_lower:
                    term_score += 2.0

                if term_score == 0:
                    matched_all_terms = False
                    break
                score += term_score

            if not matched_all_terms:
                continue
        else:
            # Default score if no query
            score = 1.0

        item = dict(paper)
        item["search_score"] = score
        filtered.append(item)

    # Sorting
    if sort_by == "relevance":
        filtered.sort(key=lambda p: (p.get("search_score", 0), p.get("year") or 0), reverse=True)
    elif sort_by == "date_desc":
        filtered.sort(key=lambda p: (p.get("year") or 0, p.get("update_date") or ""), reverse=True)
    elif sort_by == "date_asc":
        filtered.sort(key=lambda p: (p.get("year") or 9999, p.get("update_date") or "9999"))

    return filtered[:max_results]


def get_available_categories(papers: List[Dict[str, Any]]) -> List[str]:
    """Extracts unique sorted categories from papers list."""
    categories = set()
    for p in papers:
        for cat in p.get("categories", []):
            categories.add(cat)
    return sorted(list(categories))

