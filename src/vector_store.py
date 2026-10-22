"""
Vector Store Module
Provides persistent local storage and similarity retrieval for research paper chunks.
Prevents duplicate ingestion, retains complete citation and provenance metadata,
and implements hybrid semantic+lexical retrieval with intent-aware ranking.
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.config import VECTOR_DB_DIR, PDF_DIR
from src.embeddings import get_embedding_engine

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
}


class PersistentVectorStore:
    """
    Persistent, local vector database with SQLite storage.
    Supports hybrid (dense + lexical) retrieval, intent detection,
    and complete provenance tracking for verifiable academic citations.
    """

    def __init__(self, db_dir: Path = VECTOR_DB_DIR):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "research_papers.sqlite"
        self.embedding_engine = get_embedding_engine()
        self._init_database()
        self._migrate_database()

    def _init_database(self):
        """Creates tables for chunks and paper catalog if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Papers catalog
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indexed_papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    categories TEXT,
                    abstract TEXT,
                    total_pages INTEGER,
                    total_chunks INTEGER,
                    pdf_filename TEXT,
                    local_pdf_path TEXT,
                    abs_url TEXT,
                    pdf_url TEXT,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    arxiv_id TEXT,
                    page_number INTEGER,
                    chunk_index INTEGER,
                    paper_title TEXT,
                    authors TEXT,
                    pdf_filename TEXT,
                    local_pdf_path TEXT,
                    abs_url TEXT,
                    pdf_url TEXT,
                    publication_date TEXT,
                    categories TEXT,
                    chunk_text TEXT,
                    embedding_json TEXT,
                    FOREIGN KEY (arxiv_id) REFERENCES indexed_papers (arxiv_id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id ON document_chunks (arxiv_id)
            """)
            conn.commit()

    def _migrate_database(self):
        """Safely adds missing metadata columns to existing tables without data loss."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Check indexed_papers columns
            cursor.execute("PRAGMA table_info(indexed_papers)")
            existing_paper_cols = {col[1] for col in cursor.fetchall()}
            
            paper_cols_to_add = [
                ("authors", "TEXT"),
                ("abstract", "TEXT"),
                ("local_pdf_path", "TEXT"),
                ("abs_url", "TEXT"),
                ("pdf_url", "TEXT"),
            ]
            for col_name, col_type in paper_cols_to_add:
                if col_name not in existing_paper_cols:
                    try:
                        cursor.execute(f"ALTER TABLE indexed_papers ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError:
                        pass

            # Check document_chunks columns
            cursor.execute("PRAGMA table_info(document_chunks)")
            existing_chunk_cols = {col[1] for col in cursor.fetchall()}

            chunk_cols_to_add = [
                ("authors", "TEXT"),
                ("local_pdf_path", "TEXT"),
                ("abs_url", "TEXT"),
                ("pdf_url", "TEXT"),
            ]
            for col_name, col_type in chunk_cols_to_add:
                if col_name not in existing_chunk_cols:
                    try:
                        cursor.execute(f"ALTER TABLE document_chunks ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError:
                        pass

            conn.commit()

    def is_paper_indexed(self, arxiv_id: str) -> bool:
        """Checks if a paper has already been ingested into the vector store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM indexed_papers WHERE arxiv_id = ?", (arxiv_id,))
            return cursor.fetchone() is not None

    def add_paper_chunks(
        self,
        paper_metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Ingests chunks and embeddings for a paper into the vector store.
        Safely replaces existing entries for the same paper to prevent duplicates.
        """
        if not chunks:
            return 0

        arxiv_id = paper_metadata.get("id", "unknown")
        title = paper_metadata.get("title", "Untitled Paper")
        authors = paper_metadata.get("authors", "Not available")
        abstract = paper_metadata.get("abstract", "")
        pdf_filename = paper_metadata.get("pdf_filename", f"{arxiv_id}.pdf")
        local_pdf_path = paper_metadata.get("local_pdf_path", str(PDF_DIR / pdf_filename))
        abs_url = paper_metadata.get("abs_url", f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id != "unknown" else "")
        pdf_url = paper_metadata.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id != "unknown" else "")
        year = paper_metadata.get("year")
        categories_str = " ".join(paper_metadata.get("categories", []))

        # Generate embeddings if not pre-supplied
        if embeddings is None or len(embeddings) != len(chunks):
            texts = [c["text"] for c in chunks]
            embeddings = self.embedding_engine.embed_documents(texts)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Delete existing chunks for this paper to ensure clean re-ingestion
            cursor.execute("DELETE FROM document_chunks WHERE arxiv_id = ?", (arxiv_id,))

            # Insert or replace paper record
            max_page = max((c.get("page_number", 1) for c in chunks), default=1)
            cursor.execute(
                """
                INSERT OR REPLACE INTO indexed_papers 
                (arxiv_id, title, authors, year, categories, abstract, total_pages, total_chunks, pdf_filename, local_pdf_path, abs_url, pdf_url, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    arxiv_id,
                    title,
                    authors,
                    year,
                    categories_str,
                    abstract,
                    max_page,
                    len(chunks),
                    pdf_filename,
                    local_pdf_path,
                    abs_url,
                    pdf_url,
                ),
            )

            # Insert chunks with full provenance
            for chunk, emb in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT INTO document_chunks 
                    (chunk_id, arxiv_id, page_number, chunk_index, paper_title, authors, pdf_filename, local_pdf_path, abs_url, pdf_url, publication_date, categories, chunk_text, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk["chunk_id"],
                        arxiv_id,
                        chunk["page_number"],
                        chunk.get("chunk_index", 1),
                        title,
                        authors,
                        pdf_filename,
                        local_pdf_path,
                        abs_url,
                        pdf_url,
                        str(year or "Not available"),
                        categories_str,
                        chunk["text"],
                        json.dumps(emb),
                    ),
                )
            conn.commit()

        logger.info(f"Ingested {len(chunks)} chunks for paper '{title[:40]}...' ({arxiv_id})")
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        paper_id_filter: Optional[str] = None,
        min_relevance_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant chunks using hybrid (dense semantic + lexical keyword) retrieval.
        Strictly enforces paper_id_filter when supplied.
        """
        # Dense query vector
        query_vec = np.array(self.embedding_engine.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Lexical query terms (filtering stop words)
        raw_words = re.findall(r"\w+", query.lower())
        query_terms = [w for w in raw_words if len(w) > 2 and w not in STOP_WORDS]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if paper_id_filter and paper_id_filter != "All":
                cursor.execute(
                    """
                    SELECT chunk_id, arxiv_id, page_number, paper_title, authors, pdf_filename, local_pdf_path, abs_url, pdf_url, publication_date, categories, chunk_text, embedding_json
                    FROM document_chunks
                    WHERE arxiv_id = ?
                """,
                    (paper_id_filter,),
                )
            else:
                cursor.execute("""
                    SELECT chunk_id, arxiv_id, page_number, paper_title, authors, pdf_filename, local_pdf_path, abs_url, pdf_url, publication_date, categories, chunk_text, embedding_json
                    FROM document_chunks
                """)

            rows = cursor.fetchall()

        if not rows:
            return []

        scored_chunks: List[Tuple[float, Dict[str, Any]]] = []

        for row in rows:
            (
                chunk_id,
                arxiv_id,
                page_number,
                paper_title,
                authors,
                pdf_filename,
                local_pdf_path,
                abs_url,
                pdf_url,
                pub_date,
                categories,
                chunk_text,
                embedding_json,
            ) = row

            # 1. Dense Semantic Cosine Similarity
            try:
                emb = np.array(json.loads(embedding_json), dtype=np.float32)
                e_norm = np.linalg.norm(emb)
                if e_norm > 0:
                    emb = emb / e_norm
                dense_score = max(0.0, float(np.dot(query_vec, emb)))
            except Exception:
                dense_score = 0.0

            # 2. Lexical Keyword Overlap Score
            lexical_score = 0.0
            if query_terms:
                chunk_lower = chunk_text.lower()
                matched_terms = sum(1 for term in query_terms if term in chunk_lower)
                lexical_score = matched_terms / len(query_terms)

            # 3. Combined Hybrid Score (weighted combination)
            # Chunks matching exact important keywords receive strong boost
            hybrid_score = 0.55 * dense_score + 0.45 * lexical_score

            # Adaptive filter: ignore completely unrelated noise
            if hybrid_score < min_relevance_threshold:
                continue

            scored_chunks.append((
                hybrid_score,
                {
                    "chunk_id": chunk_id,
                    "arxiv_id": arxiv_id,
                    "page_number": page_number,
                    "paper_title": paper_title or f"arXiv:{arxiv_id}",
                    "authors": authors or "Not available",
                    "pdf_filename": pdf_filename or f"{arxiv_id}.pdf",
                    "local_pdf_path": local_pdf_path or str(PDF_DIR / f"{arxiv_id}.pdf"),
                    "abs_url": abs_url or f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "publication_date": pub_date or "Not available",
                    "categories": categories.split() if categories else [],
                    "text": chunk_text,
                    "similarity_score": round(hybrid_score, 4),
                    "dense_score": round(dense_score, 4),
                    "lexical_score": round(lexical_score, 4),
                },
            ))

        # Sort descending by hybrid similarity score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics for dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM indexed_papers")
            total_papers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            total_chunks = cursor.fetchone()[0]

            cursor.execute("SELECT categories FROM indexed_papers")
            rows = cursor.fetchall()
            cats = set()
            for r in rows:
                if r[0]:
                    for c in r[0].split():
                        cats.add(c)

            cursor.execute("""
                SELECT arxiv_id, title, total_chunks, total_pages, indexed_at 
                FROM indexed_papers 
                ORDER BY indexed_at DESC LIMIT 5
            """)
            recent = [
                {
                    "arxiv_id": r[0],
                    "title": r[1],
                    "total_chunks": r[2],
                    "total_pages": r[3],
                    "indexed_at": r[4],
                }
                for r in cursor.fetchall()
            ]

        return {
            "total_papers_indexed": total_papers,
            "total_chunks_indexed": total_chunks,
            "categories_count": len(cats),
            "categories": sorted(list(cats)),
            "recent_papers": recent,
        }

    def list_indexed_papers(self) -> List[Dict[str, Any]]:
        """Returns all indexed papers in the library with complete provenance metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT arxiv_id, title, authors, year, categories, abstract, total_pages, total_chunks, pdf_filename, local_pdf_path, abs_url, pdf_url, indexed_at 
                FROM indexed_papers
                ORDER BY indexed_at DESC
            """)
            papers = []
            for r in cursor.fetchall():
                papers.append({
                    "arxiv_id": r[0],
                    "title": r[1],
                    "authors": r[2] or "Not available",
                    "year": r[3],
                    "categories": r[4].split() if r[4] else [],
                    "abstract": r[5] or "",
                    "total_pages": r[6],
                    "total_chunks": r[7],
                    "pdf_filename": r[8],
                    "local_pdf_path": r[9] or str(PDF_DIR / (r[8] or f"{r[0]}.pdf")),
                    "abs_url": r[10] or f"https://arxiv.org/abs/{r[0]}",
                    "pdf_url": r[11] or f"https://arxiv.org/pdf/{r[0]}.pdf",
                    "indexed_at": r[12],
                })
            return papers

    def delete_paper(self, arxiv_id: str) -> bool:
        """Removes a paper and all its chunks from the vector database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks WHERE arxiv_id = ?", (arxiv_id,))
            cursor.execute("DELETE FROM indexed_papers WHERE arxiv_id = ?", (arxiv_id,))
            conn.commit()
            return cursor.rowcount > 0


# Global singleton instance
_vector_store_instance: Optional[PersistentVectorStore] = None


def get_vector_store() -> PersistentVectorStore:
    """Returns singleton instance of PersistentVectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = PersistentVectorStore()
    return _vector_store_instance
