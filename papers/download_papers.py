"""
Paper Downloader Script
Downloads research paper PDFs from arXiv using valid identifiers.
Adheres to arXiv access rate limits, validates PDF magic headers, and skips existing files.
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import requests
from src.config import (
    PDF_DIR,
    ARXIV_PDF_BASE_URL,
    USER_AGENT,
    REQUEST_TIMEOUT,
    REQUEST_DELAY_SECONDS,
    DEFAULT_DOWNLOAD_LIMIT,
)
from src.metadata_loader import sanitize_arxiv_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_valid_pdf_content(content: bytes) -> bool:
    """Verifies that downloaded bytes start with the %PDF magic header and have non-trivial size."""
    if len(content) < 1000:
        return False
    return content.startswith(b"%PDF-")


def get_pdf_path(arxiv_id: str) -> Path:
    """Returns local destination path for a given arXiv ID."""
    clean_id = sanitize_arxiv_id(arxiv_id).replace("/", "_")
    return PDF_DIR / f"{clean_id}.pdf"


def download_single_paper(
    arxiv_id: str,
    target_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Tuple[bool, str, Optional[Path]]:
    """
    Downloads a single paper PDF from arXiv.
    
    Returns:
        (success: bool, message: str, file_path: Optional[Path])
    """
    clean_id = sanitize_arxiv_id(arxiv_id)
    if not clean_id:
        return False, "Invalid or empty arXiv ID provided.", None

    save_dir = target_dir or PDF_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"{clean_id.replace('/', '_')}.pdf"
    file_path = save_dir / pdf_filename

    # Skip if already exists and not overwriting
    if file_path.exists() and not overwrite:
        if file_path.stat().st_size > 1000:
            logger.info(f"Skipping already existing PDF: {file_path.name}")
            return True, f"File already exists: {file_path.name}", file_path

    url = f"{ARXIV_PDF_BASE_URL}{clean_id}.pdf"
    headers = {"User-Agent": USER_AGENT}

    logger.info(f"Downloading from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        
        # Check HTTP status
        if response.status_code == 404:
            return False, f"Paper {clean_id} not found on arXiv (HTTP 404).", None
        elif response.status_code == 403 or response.status_code == 429:
            return False, f"arXiv rate limit or access restriction reached (HTTP {response.status_code}). Please wait before retrying.", None
        elif response.status_code != 200:
            return False, f"Download failed with HTTP status {response.status_code}.", None

        content = response.content
        if not is_valid_pdf_content(content):
            # Often arXiv returns an HTML block page or CAPTCHA if rate-limited
            if b"<html" in content.lower():
                return False, f"Received HTML error page instead of PDF for {clean_id}. arXiv may require verification.", None
            return False, f"Downloaded file for {clean_id} failed PDF validation header checks.", None

        # Write to disk
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Successfully downloaded {clean_id} -> {file_path.name} ({len(content) / 1024:.1f} KB)")
        return True, f"Successfully downloaded {file_path.name}", file_path

    except requests.exceptions.Timeout:
        return False, f"Connection timed out while fetching {clean_id}.", None
    except requests.exceptions.RequestException as e:
        return False, f"Network error downloading {clean_id}: {str(e)}", None
    except Exception as e:
        return False, f"Unexpected error saving PDF for {clean_id}: {str(e)}", None


def download_papers(
    paper_ids: List[str],
    limit: int = DEFAULT_DOWNLOAD_LIMIT,
    delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> Dict[str, Any]:
    """
    Downloads a list of selected papers with rate limiting and safeguards.
    """
    results = {
        "successful": [],
        "skipped": [],
        "failed": [],
        "total_requested": len(paper_ids),
    }

    selected = paper_ids[:limit]
    logger.info(f"Starting controlled download of {len(selected)} papers (limit: {limit})...")

    for idx, pid in enumerate(selected):
        success, msg, path = download_single_paper(pid)
        if success:
            if "already exists" in msg:
                results["skipped"].append({"id": pid, "message": msg, "path": str(path)})
            else:
                results["successful"].append({"id": pid, "message": msg, "path": str(path)})
        else:
            results["failed"].append({"id": pid, "message": msg})

        # Polite interval between requests
        if idx < len(selected) - 1:
            time.sleep(delay_seconds)

    logger.info(
        f"Download batch finished: {len(results['successful'])} downloaded, "
        f"{len(results['skipped'])} existing, {len(results['failed'])} failed."
    )
    return results


def main():
    """CLI Entrypoint for downloading papers."""
    import argparse
    from src.metadata_loader import stream_metadata

    parser = argparse.ArgumentParser(description="Download research paper PDFs from arXiv")
    parser.add_argument("--ids", nargs="+", help="Specific arXiv IDs to download")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of papers to download (default: 5)")
    parser.add_argument("--category", type=str, default="cs.AI", help="Target arXiv category if sampling from dataset")
    args = parser.parse_args()

    ids_to_download = []
    if args.ids:
        ids_to_download = args.ids
    else:
        logger.info(f"No explicit IDs provided. Sampling top {args.limit} papers from dataset category {args.category}...")
        for paper in stream_metadata(limit=args.limit * 3, target_categories=(args.category,)):
            ids_to_download.append(paper["id"])
            if len(ids_to_download) >= args.limit:
                break

    if not ids_to_download:
        print("No papers found to download.")
        return

    print(f"Selected arXiv IDs: {ids_to_download}")
    results = download_papers(ids_to_download, limit=args.limit)
    print("\nSummary:")
    print(f"  Successfully downloaded: {len(results['successful'])}")
    print(f"  Skipped (already cached): {len(results['skipped'])}")
    print(f"  Failed: {len(results['failed'])}")


if __name__ == "__main__":
    main()

