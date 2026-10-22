"""
Metadata Loader Module
Streams and parses arXiv metadata JSON Lines file without loading the entire 3.8GB dataset into memory.
Caches and indexes samples for high-performance discovery.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Any
from datetime import datetime

from src.config import ARXIV_METADATA_PATH, METADATA_SAMPLE_CACHE_PATH

logger = logging.getLogger(__name__)

DEFAULT_TARGET_CATEGORIES = (
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "cs.IR", "stat.ML"
)


def clean_text(text: Optional[str]) -> str:
    """Removes excess whitespace, newlines, and normalizes text."""
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").split())


def extract_year(paper: Dict[str, Any]) -> Optional[int]:
    """Extracts publication or update year from arXiv metadata record."""
    # Check update_date first (e.g. "2021-05-24")
    update_date = paper.get("update_date")
    if update_date and isinstance(update_date, str):
        try:
            return int(update_date.split("-")[0])
        except (ValueError, IndexError):
            pass

    # Check versions (e.g. created dates)
    versions = paper.get("versions", [])
    if versions and isinstance(versions, list):
        created = versions[0].get("created", "")
        if created:
            # Format: 'Mon, 2 Apr 2007 19:18:42 GMT'
            parts = created.split()
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    try:
                        return int(part)
                    except ValueError:
                        pass

    # Extract from arxiv id if possible (e.g. "2103.12345" -> 2021, "0704.1234" -> 2007)
    arxiv_id = paper.get("id", "")
    if arxiv_id and "." in arxiv_id:
        prefix = arxiv_id.split(".")[0]
        if len(prefix) == 4 and prefix.isdigit():
            yy = int(prefix[:2])
            year = 2000 + yy if yy < 90 else 1900 + yy
            return year

    return None


def sanitize_arxiv_id(raw_id: str) -> str:
    """Strips version suffix (e.g., 'v1', 'v2') and whitespace from arXiv ID."""
    clean_id = raw_id.strip()
    # If ID ends with v followed by digits, remove version
    if "v" in clean_id:
        prefix, sep, suffix = clean_id.rpartition("v")
        if suffix.isdigit():
            clean_id = prefix
    return clean_id


def parse_metadata_line(line: str) -> Optional[Dict[str, Any]]:
    """Safely decodes and normalizes a single JSON line."""
    line = line.strip()
    if not line:
        return None
    try:
        raw = json.loads(line)
        arxiv_id = sanitize_arxiv_id(str(raw.get("id", "")))
        if not arxiv_id:
            return None

        title = clean_text(raw.get("title", "Untitled"))
        abstract = clean_text(raw.get("abstract", ""))
        authors = clean_text(raw.get("authors", "Unknown"))
        categories = raw.get("categories", "").split()
        year = extract_year(raw)

        return {
            "id": arxiv_id,
            "raw_id": raw.get("id", ""),
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "categories": categories,
            "categories_str": " ".join(categories),
            "journal_ref": clean_text(raw.get("journal-ref", "")),
            "doi": raw.get("doi", ""),
            "update_date": raw.get("update_date", ""),
            "year": year,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        }
    except Exception as e:
        logger.debug(f"Error parsing metadata line: {e}")
        return None


def stream_metadata(
    file_path: Optional[Path] = None,
    limit: Optional[int] = None,
    target_categories: Optional[tuple] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Streams records from arxiv-metadata.json incrementally.
    Avoids loading the large file into RAM.
    """
    path = file_path or ARXIV_METADATA_PATH
    if not path.exists():
        logger.warning(f"Metadata file not found: {path}")
        return

    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            record = parse_metadata_line(line)
            if not record:
                continue

            if target_categories:
                if not any(
                    any(cat.startswith(target) for target in target_categories)
                    for cat in record["categories"]
                ):
                    continue

            yield record
            count += 1
            if limit and count >= limit:
                break


def build_metadata_sample(
    sample_size: int = 2500,
    force_rebuild: bool = False,
    target_categories: tuple = DEFAULT_TARGET_CATEGORIES,
) -> List[Dict[str, Any]]:
    """
    Builds and caches a representative subset of papers to make UI searching
    extremely fast and responsive without scanning 3.8GB on every keystroke.
    """
    if METADATA_SAMPLE_CACHE_PATH.exists() and not force_rebuild:
        try:
            with open(METADATA_SAMPLE_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            logger.warning(f"Could not load cached sample: {e}. Rebuilding...")

    papers: List[Dict[str, Any]] = []
    seen_ids = set()

    for paper in stream_metadata(limit=sample_size * 2, target_categories=target_categories):
        pid = paper["id"]
        if pid not in seen_ids:
            seen_ids.add(pid)
            papers.append(paper)
            if len(papers) >= sample_size:
                break

    # If dataset has fewer records than requested, save what we found
    if papers:
        try:
            with open(METADATA_SAMPLE_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=2)
            logger.info(f"Cached {len(papers)} papers to {METADATA_SAMPLE_CACHE_PATH}")
        except Exception as e:
            logger.error(f"Failed to cache metadata sample: {e}")

    return papers


def get_paper_by_id(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Looks up a paper by arXiv ID from the cached sample or streams dataset."""
    clean_id = sanitize_arxiv_id(arxiv_id)

    # 1. Search cached sample
    if METADATA_SAMPLE_CACHE_PATH.exists():
        try:
            with open(METADATA_SAMPLE_CACHE_PATH, "r", encoding="utf-8") as f:
                sample = json.load(f)
                for paper in sample:
                    if paper["id"] == clean_id:
                        return paper
        except Exception:
            pass

    # 2. Search dataset streaming up to 20,000 papers
    for paper in stream_metadata(limit=25000):
        if paper["id"] == clean_id:
            return paper

    return None

