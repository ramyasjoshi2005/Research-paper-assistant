"""
Research Paper Assistant — Main Streamlit Application
A modern, hackathon-ready AI Research Paper Assistant powered by Retrieval-Augmented Generation (RAG).
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

# Configure Streamlit page (must be first Streamlit command)
st.set_page_config(
    page_title="Research Paper Assistant | RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
from datetime import datetime

from src.config import (
    PDF_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_DOWNLOAD_LIMIT,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    LLM_PROVIDER,
    GEMINI_MODEL,
    OPENAI_MODEL,
)
from src.metadata_loader import (
    build_metadata_sample,
    get_paper_by_id,
    sanitize_arxiv_id,
    DEFAULT_TARGET_CATEGORIES,
)
from src.paper_search import search_papers, get_available_categories
from papers.download_papers import download_single_paper, download_papers
from src.pdf_processor import extract_text_from_pdf
from src.rag_pipeline import get_rag_pipeline
from src.vector_store import get_vector_store
from src.llm_provider import get_llm_client
from src.summarizer import summarize_paper
from src.paper_comparison import compare_two_papers

# Custom Academic-Tech Theme CSS
st.markdown(
    """
    <style>
    /* Dark Navy Academic Aesthetic */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .app-header {
        padding: 1.2rem 1.8rem;
        background: linear-gradient(135deg, #111827 0%, #1e1b4b 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .app-header p {
        margin: 0.25rem 0 0 0;
        font-size: 0.9rem;
        color: #94a3b8;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.1rem;
        text-align: left;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* Paper Result Card */
    .paper-card {
        background: #131b2e;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .paper-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }
    .paper-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.35rem;
    }
    .paper-meta {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    .category-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .year-badge {
        display: inline-block;
        background: rgba(14, 165, 233, 0.18);
        color: #38bdf8;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .status-badge-green {
        display: inline-block;
        background: rgba(34, 197, 94, 0.18);
        color: #4ade80;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Evidence Citation Callout */
    .evidence-box {
        background: #0f172a;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    .evidence-box strong {
        color: #38bdf8;
    }
    
    /* Fallback Alert */
    .fallback-banner {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        color: #fde047;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Bar
st.markdown(
    """
    <div class="app-header">
        <div>
            <h1>🔬 Research Paper Assistant</h1>
            <p>Retrieval-Augmented Generation (RAG) with Grounded Citations & Evidence Verification</p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(99, 102, 241, 0.25); color: #c7d2fe; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.4);">Hackathon Edition 1.0</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def safe_render_dataframe(df: pd.DataFrame, use_container_width: bool = True, hide_index: bool = True):
    """
    Safely renders a pandas DataFrame in Streamlit without Arrow LargeUtf8 errors.
    Ensures string columns are represented as standard Python string objects (32-bit Utf8).
    Falls back gracefully to st.table() if Arrow serialization encounters any unrecognized types.
    """
    if df is None or df.empty:
        return
    try:
        clean_df = df.copy()
        for col in clean_df.columns:
            if clean_df[col].dtype == object or str(clean_df[col].dtype).startswith(("string", "large", "Large")):
                clean_df[col] = clean_df[col].astype(str)
        st.dataframe(clean_df, use_container_width=use_container_width, hide_index=hide_index)
    except Exception:
        st.table(df)


# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sample_papers" not in st.session_state:
    with st.spinner("Initializing paper discovery index..."):
        st.session_state.sample_papers = build_metadata_sample(sample_size=1000)

# Sidebar Navigation
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png",
        width=48,
    )
    st.markdown("### **Navigation**")
    nav_option = st.radio(
        "Select Module:",
        [
            "📊 Dashboard",
            "🔍 Discover Papers",
            "📚 My Library",
            "💬 Chat with Papers",
            "📝 Summarize a Paper",
            "⚖️ Compare Papers",
            "⚙️ Settings",
        ],
        index=0,
    )

    st.markdown("---")
    # Quick Status in Sidebar
    vec_store = get_vector_store()
    stats = vec_store.get_stats()
    st.markdown("### **System Status**")
    st.markdown(f"• **Indexed Papers:** `{stats['total_papers_indexed']}`")
    st.markdown(f"• **Text Chunks:** `{stats['total_chunks_indexed']}`")

    llm = get_llm_client()
    if llm.has_api_key():
        st.markdown(f"• **Active LLM:** `🟢 {llm.provider.capitalize()}`")
    else:
        st.markdown("• **LLM Mode:** `⚡ Local Extractive Fallback`")

    st.markdown("---")
    st.caption("Grounded in Cornell arXiv research data.")


# ==============================================================================
# 1. DASHBOARD
# ==============================================================================
if nav_option == "📊 Dashboard":
    st.markdown("## 📊 Research Assistant Dashboard")
    st.caption("Live metrics calculated directly from your local repository and vector database.")

    # Calculate real statistics
    vec_store = get_vector_store()
    stats = vec_store.get_stats()
    downloaded_pdfs = list(PDF_DIR.glob("*.pdf"))
    total_download_bytes = sum(f.stat().st_size for f in downloaded_pdfs) if downloaded_pdfs else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_papers_indexed']}</div>
                <div class="metric-label">Papers Ingested (RAG)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(downloaded_pdfs)}</div>
                <div class="metric-label">PDFs Downloaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_chunks_indexed']}</div>
                <div class="metric-label">Vector Chunks Indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['categories_count']}</div>
                <div class="metric-label">arXiv Categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Status & Quick Workflow
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown("### 📋 Recent Indexing Activity")
        if stats["recent_papers"]:
            recent_df = pd.DataFrame(stats["recent_papers"])
            recent_df = recent_df.rename(
                columns={
                    "arxiv_id": "arXiv ID",
                    "title": "Paper Title",
                    "total_chunks": "Chunks",
                    "total_pages": "Pages",
                    "indexed_at": "Indexed At",
                }
            )
            safe_render_dataframe(recent_df, use_container_width=True, hide_index=True)
        else:
            st.info("No papers have been indexed yet. Head to **Discover Papers** or **My Library** to download and index your first research paper.")

    with right_col:
        st.markdown("### 🚀 Quick Start Demonstration")
        st.markdown(
            """
            1. **🔍 Discover Papers:** Search 1,000+ cached AI/ML papers or filter by arXiv topic (e.g., Computer Vision, NLP).
            2. **📥 Download & Index:** Download selected PDFs from arXiv and index them into the persistent vector database.
            3. **💬 Grounded Chat:** Ask questions with verified page citations and confidence scores.
            4. **📝 Summarize & Compare:** Auto-generate structured 8-part summaries or compare two papers side-by-side.
            """
        )

        st.markdown("### 🛡️ Grounding & Verification Guarantee")
        st.markdown(
            """
            - **No Hallucinations:** Answers cite specific page numbers and exact passage text.
            - **Zero-API-Key Fallback:** Operates completely offline with transparent extractive synthesis.
            """
        )


# ==============================================================================
# 2. DISCOVER PAPERS
# ==============================================================================
elif nav_option == "🔍 Discover Papers":
    st.markdown("## 🔍 Discover Academic Research Papers")
    st.caption("Search arXiv research papers by topic, title, abstract, or author.")

    papers_pool = st.session_state.sample_papers

    # Search & Filter Controls
    search_col1, search_col2, search_col3 = st.columns([2.5, 1, 1])
    with search_col1:
        query = st.text_input("Enter search keywords or paper topic:", placeholder="e.g. neural network, attention mechanism, vision transformer")
    with search_col2:
        all_categories = ["All"] + get_available_categories(papers_pool)
        selected_category = st.selectbox("arXiv Category Filter:", all_categories, index=0)
    with search_col3:
        sort_by = st.selectbox(
            "Sort Results By:",
            [("relevance", "Most Relevant"), ("date_desc", "Newest First"), ("date_asc", "Oldest First")],
            format_func=lambda x: x[1],
        )[0]

    # Advanced Filters Expander
    with st.expander("⚙️ Advanced Search Filters"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            years = [p["year"] for p in papers_pool if p.get("year")]
            min_y = min(years) if years else 2000
            max_y = max(years) if years else datetime.now().year
            selected_years = st.slider("Publication Year Range:", min_value=min_y, max_value=max_y, value=(min_y, max_y))
        with f_col2:
            max_results = st.number_input("Maximum Results to Display:", min_value=5, max_value=100, value=20)

    # Perform Search
    matching_papers = search_papers(
        papers_pool,
        query=query,
        category=selected_category,
        min_year=selected_years[0],
        max_year=selected_years[1],
        sort_by=sort_by,
        max_results=max_results,
    )

    st.markdown(f"**Showing {len(matching_papers)} results**")

    if not matching_papers:
        st.warning("No papers match your search criteria. Try broadening your keywords or resetting category filters.")
    else:
        vec_store = get_vector_store()
        for paper in matching_papers:
            pid = paper["id"]
            pdf_path = PDF_DIR / f"{pid.replace('/', '_')}.pdf"
            is_downloaded = pdf_path.exists()
            is_indexed = vec_store.is_paper_indexed(pid)

            with st.container():
                st.markdown(
                    f"""
                    <div class="paper-card">
                        <div class="paper-title">{paper['title']}</div>
                        <div class="paper-meta">
                            <strong>Authors:</strong> {paper['authors'][:120]}{'...' if len(paper['authors']) > 120 else ''} &nbsp;|&nbsp;
                            <strong>arXiv:</strong> <a href="{paper['abs_url']}" target="_blank" style="color: #818cf8;">{paper['raw_id']}</a> &nbsp;|&nbsp;
                            <span class="year-badge">Year: {paper.get('year') or 'N/A'}</span>
                            {'<span class="status-badge-green">🟢 In Vector DB</span>' if is_indexed else ('<span style="color: #38bdf8; font-size:0.8rem;">📥 Downloaded</span>' if is_downloaded else '')}
                        </div>
                        <div style="margin-bottom: 0.6rem;">
                            {' '.join([f'<span class="category-badge">{cat}</span>' for cat in paper.get('categories', [])[:4]])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Abstract expander
                with st.expander(f"📄 View Abstract ({pid})"):
                    st.write(paper["abstract"])

                # Action buttons
                btn_col1, btn_col2, btn_col3, _ = st.columns([1.3, 1.6, 1.2, 2.5])
                with btn_col1:
                    if not is_downloaded:
                        if st.button(f"📥 Download PDF", key=f"dl_{pid}"):
                            with st.spinner(f"Downloading {pid} from arXiv..."):
                                success, msg, path = download_single_paper(pid)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                    else:
                        st.caption("✅ PDF on disk")

                with btn_col2:
                    if not is_indexed:
                        if st.button(f"⚡ Ingest & Index (RAG)", key=f"idx_{pid}"):
                            with st.spinner(f"Downloading & indexing {pid}..."):
                                if not is_downloaded:
                                    success, msg, path = download_single_paper(pid)
                                    if not success:
                                        st.error(f"Download failed: {msg}")
                                        st.stop()
                                    pdf_path = path

                                pipe = get_rag_pipeline()
                                res = pipe.ingest_paper_pdf(pdf_path, paper_metadata=paper)
                                if res["success"]:
                                    st.success(res["message"])
                                    st.rerun()
                                else:
                                    st.error(res["message"])
                    else:
                        st.caption("✅ Ready for Chat & RAG")

                with btn_col3:
                    st.markdown(f"[🔗 arXiv Page]({paper['abs_url']})")

                st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 0.5rem 0;'>", unsafe_allow_html=True)


# ==============================================================================
# 3. MY LIBRARY
# ==============================================================================
elif nav_option == "📚 My Library":
    st.markdown("## 📚 Downloaded Papers Library")
    st.caption("Manage local research PDFs, inspect extraction quality, and index documents for RAG.")

    vec_store = get_vector_store()
    downloaded_files = sorted(list(PDF_DIR.glob("*.pdf")))

    lib_top_col1, lib_top_col2 = st.columns([2, 1])
    with lib_top_col1:
        st.markdown(f"**Found {len(downloaded_files)} downloaded PDF documents in `papers/pdfs/`**")
    with lib_top_col2:
        if downloaded_files:
            if st.button("⚡ Index All Unindexed PDFs", type="primary"):
                pipe = get_rag_pipeline()
                progress = st.progress(0)
                status_text = st.empty()
                indexed_count = 0

                for i, pdf_file in enumerate(downloaded_files):
                    status_text.text(f"Processing {pdf_file.name} ({i+1}/{len(downloaded_files)})...")
                    res = pipe.ingest_paper_pdf(pdf_file)
                    if res["success"] and not res.get("already_indexed"):
                        indexed_count += 1
                    progress.progress((i + 1) / len(downloaded_files))

                st.success(f"Batch indexing complete! Ingested {indexed_count} new papers.")
                st.rerun()

    if not downloaded_files:
        st.info("No research papers have been downloaded yet. Visit the **Discover Papers** tab to search and download papers.")
    else:
        for pdf_path in downloaded_files:
            clean_id = sanitize_arxiv_id(pdf_path.stem.replace("_", "/"))
            file_kb = pdf_path.stat().st_size / 1024
            is_indexed = vec_store.is_paper_indexed(clean_id)
            meta = get_paper_by_id(clean_id) or {"title": pdf_path.stem, "year": None, "categories": []}

            with st.container():
                st.markdown(
                    f"""
                    <div class="paper-card">
                        <div class="paper-title">{meta['title']}</div>
                        <div class="paper-meta">
                            <strong>File:</strong> <code>{pdf_path.name}</code> ({file_kb:.1f} KB) &nbsp;|&nbsp;
                            <strong>arXiv ID:</strong> {clean_id} &nbsp;|&nbsp;
                            {'<span class="status-badge-green">🟢 Indexed in Vector DB</span>' if is_indexed else '<span style="color: #94a3b8;">⚪ Not Indexed</span>'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                row1, row2, row3, row4 = st.columns([1.5, 1.8, 1.2, 1.5])
                with row1:
                    if not is_indexed:
                        if st.button("⚡ Index for RAG", key=f"lib_idx_{clean_id}"):
                            with st.spinner("Extracting text, chunking, and embedding..."):
                                pipe = get_rag_pipeline()
                                res = pipe.ingest_paper_pdf(pdf_path, paper_metadata=meta)
                                if res["success"]:
                                    st.success(res["message"])
                                    st.rerun()
                                else:
                                    st.error(res["message"])
                    else:
                        st.caption("✅ Ingested")

                with row2:
                    if st.button("📄 Inspect Extracted Text", key=f"insp_{clean_id}"):
                        extraction = extract_text_from_pdf(pdf_path)
                        st.write(f"**Total Pages:** {extraction['total_pages']} | **Characters:** {extraction['total_characters']}")
                        if extraction["pages"]:
                            st.text_area(f"Page 1 Preview ({pdf_path.name})", extraction["pages"][0]["text"][:1000], height=150)

                with row3:
                    if st.button("🗑️ Delete PDF", key=f"del_{clean_id}"):
                        vec_store.delete_paper(clean_id)
                        if pdf_path.exists():
                            pdf_path.unlink()
                        st.success(f"Deleted {pdf_path.name}")
                        st.rerun()

                with row4:
                    abs_link = meta.get("abs_url", f"https://arxiv.org/abs/{clean_id}")
                    pdf_link = meta.get("pdf_url", f"https://arxiv.org/pdf/{clean_id}.pdf")
                    st.markdown(f"[🔗 arXiv Page]({abs_link}) &nbsp;|&nbsp; [📄 PDF]({pdf_link})")

                st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 0.5rem 0;'>", unsafe_allow_html=True)


# ==============================================================================
# 4. CHAT WITH PAPERS
# ==============================================================================
elif nav_option == "💬 Chat with Papers":
    st.markdown("## 💬 Grounded AI Research Chat")
    st.caption("Ask questions grounded strictly in passages retrieved from your indexed academic papers.")

    vec_store = get_vector_store()
    indexed_papers = vec_store.list_indexed_papers()

    if not indexed_papers:
        st.warning("No papers are currently indexed in the vector store. Please download and index at least one paper in **Discover Papers** or **My Library**.")
    else:
        # Paper Filter Selector with exact ID mapping
        paper_dict = {f"{p['title']} [{p['arxiv_id']}]": p['arxiv_id'] for p in indexed_papers}
        paper_options = ["All Indexed Papers (Cross-Paper Synthesis)"] + list(paper_dict.keys())
        selected_option = st.selectbox("Target Research Paper Scope:", paper_options, index=0)

        target_paper_id = None
        if selected_option != "All Indexed Papers (Cross-Paper Synthesis)":
            target_paper_id = paper_dict[selected_option]
            st.info(f"🔒 **Enforced Single-Paper Scope:** Searching exclusively within **{selected_option}**.")
        else:
            st.warning("🌐 **Cross-Paper Scope:** Searching across ALL indexed papers in your library.")

        # Notice banner for Extractive mode
        llm = get_llm_client()
        if not llm.has_api_key():
            st.markdown(
                """
                <div class="fallback-banner">
                    ⚠️ <strong>Extractive Fallback Mode Active:</strong> No LLM API key detected. Answers will display verbatim retrieved evidence with page citations. Add an API key in <strong>Settings</strong> to enable generative synthesis.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Quick Question Chips
        st.markdown("**Suggested Research Queries:**")
        chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
        quick_query = None
        with chip_col1:
            if st.button("💡 Explain in simple terms"):
                quick_query = "Explain this research paper in simple language that a student can understand."
        with chip_col2:
            if st.button("🎯 Problem solved"):
                quick_query = "What specific problem does this research solve and why is it important?"
        with chip_col3:
            if st.button("⚙️ Methodology used"):
                quick_query = "What methodology, architecture, or algorithm is proposed in this work?"
        with chip_col4:
            if st.button("📊 Findings & limits"):
                quick_query = "What are the main experimental findings, and what limitations do the authors report?"

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander(f"📚 Verified Evidence Citations ({len(msg['citations'])} sources)"):
                        for cit in msg["citations"]:
                            label = cit.get("citation_label", f"{cit.get('citation_id', '[1]')} {cit.get('paper_title', 'Paper')}")
                            links = []
                            if cit.get("abs_url"):
                                links.append(f"<a href='{cit['abs_url']}' target='_blank' style='color: #38bdf8;'>🔗 arXiv Abstract</a>")
                            if cit.get("pdf_url"):
                                links.append(f"<a href='{cit['pdf_url']}' target='_blank' style='color: #818cf8;'>📄 Direct PDF (p. {cit.get('page_number', '?')})</a>")
                            links_str = f"<div style='margin-top: 6px;'>{' · '.join(links)}</div>" if links else ""

                            st.markdown(
                                f"""
                                <div class="evidence-box">
                                    <strong>{label}</strong> &nbsp;|&nbsp;
                                    <strong>Match:</strong> {cit.get('similarity_score', 0.0)*100:.1f}%<br>
                                    <em>"{cit.get('snippet', '')}"</em>
                                    {links_str}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # Handle User Input
        user_input = st.chat_input("Ask a question about the indexed papers...") or quick_query

        if user_input:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving evidence passages and synthesizing answer..."):
                    pipe = get_rag_pipeline()
                    result = pipe.query(
                        question=user_input,
                        paper_id_filter=target_paper_id,
                        top_k=DEFAULT_TOP_K,
                    )
                    answer_text = result.get("answer", "")
                    citations = result.get("citations", [])

                    st.markdown(answer_text)

                    if citations:
                        with st.expander(f"📚 Verified Evidence Citations ({len(citations)} sources)"):
                            for cit in citations:
                                label = cit.get("citation_label", f"{cit.get('citation_id', '[1]')} {cit.get('paper_title', 'Paper')}")
                                links = []
                                if cit.get("abs_url"):
                                    links.append(f"<a href='{cit['abs_url']}' target='_blank' style='color: #38bdf8;'>🔗 arXiv Abstract</a>")
                                if cit.get("pdf_url"):
                                    links.append(f"<a href='{cit['pdf_url']}' target='_blank' style='color: #818cf8;'>📄 Direct PDF (p. {cit.get('page_number', '?')})</a>")
                                links_str = f"<div style='margin-top: 6px;'>{' · '.join(links)}</div>" if links else ""

                                st.markdown(
                                    f"""
                                    <div class="evidence-box">
                                        <strong>{label}</strong> &nbsp;|&nbsp;
                                        <strong>Match:</strong> {cit.get('similarity_score', 0.0)*100:.1f}%<br>
                                        <em>"{cit.get('snippet', '')}"</em>
                                        {links_str}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                    })

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


# ==============================================================================
# 5. SUMMARIZE A PAPER
# ==============================================================================
elif nav_option == "📝 Summarize a Paper":
    st.markdown("## 📝 Structured Research Paper Summary")
    st.caption("Generates an 8-dimension academic summary grounded strictly in the paper's indexed text.")

    vec_store = get_vector_store()
    indexed_papers = vec_store.list_indexed_papers()

    if not indexed_papers:
        st.warning("No papers have been indexed yet. Ingest a paper in **Discover Papers** or **My Library** first.")
    else:
        paper_dict = {f"{p['title']} ({p['arxiv_id']})": p['arxiv_id'] for p in indexed_papers}
        selected_paper_label = st.selectbox("Select Paper to Summarize:", list(paper_dict.keys()))
        selected_arxiv_id = paper_dict[selected_paper_label]

        if st.button("⚡ Generate Structured Academic Summary", type="primary"):
            with st.spinner("Analyzing methodology, problem, experiments, and limitations..."):
                summary_data = summarize_paper(selected_arxiv_id)
                if summary_data["success"]:
                    st.success(f"Generated summary for: {summary_data['paper_title']}")
                    st.markdown(summary_data["summary_text"])

                    if summary_data.get("citations"):
                        with st.expander(f"📚 Source Citations & References ({len(summary_data['citations'])})"):
                            for cit in summary_data["citations"]:
                                st.markdown(
                                    f"""
                                    <div class="evidence-box">
                                        <strong>{cit['citation_id']} Page {cit['page_number']}:</strong> {cit['snippet']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                else:
                    st.error(summary_data.get("error", "Failed to generate summary."))


# ==============================================================================
# 6. COMPARE PAPERS
# ==============================================================================
elif nav_option == "⚖️ Compare Papers":
    st.markdown("## ⚖️ Comparative Research Analysis")
    st.caption("Compare two research papers across 10 evaluation dimensions grounded in retrieved evidence.")

    vec_store = get_vector_store()
    indexed_papers = vec_store.list_indexed_papers()

    if len(indexed_papers) < 2:
        st.warning(f"At least two papers must be indexed to run a comparative analysis. Currently indexed: {len(indexed_papers)}.")
    else:
        paper_dict = {f"{p['title']} ({p['arxiv_id']})": p['arxiv_id'] for p in indexed_papers}
        labels = list(paper_dict.keys())

        c1, c2 = st.columns(2)
        with c1:
            paper_a_label = st.selectbox("Select Paper A:", labels, index=0)
            id_a = paper_dict[paper_a_label]
        with c2:
            paper_b_label = st.selectbox("Select Paper B:", labels, index=1 if len(labels) > 1 else 0)
            id_b = paper_dict[paper_b_label]

        if id_a == id_b:
            st.info("Please select two distinct research papers to compare.")
        else:
            if st.button("⚡ Run Comparative Analysis", type="primary"):
                with st.spinner(f"Comparing {id_a} vs {id_b} across 10 dimensions..."):
                    comp_result = compare_two_papers(id_a, id_b)
                    if comp_result["success"]:
                        st.success("Comparative matrix successfully compiled!")
                        st.markdown(comp_result["markdown_table"])

                        # Citations expander
                        with st.expander("📚 Source Evidence for Paper A"):
                            for c in comp_result.get("citations_a", []):
                                st.markdown(f"**Page {c['page_number']}:** {c['text'][:250]}...")
                        with st.expander("📚 Source Evidence for Paper B"):
                            for c in comp_result.get("citations_b", []):
                                st.markdown(f"**Page {c['page_number']}:** {c['text'][:250]}...")
                    else:
                        st.error(comp_result.get("error", "Comparison failed."))


# ==============================================================================
# 7. SETTINGS
# ==============================================================================
elif nav_option == "⚙️ Settings":
    st.markdown("## ⚙️ Application Settings & Configuration")
    st.caption("Configure LLM credentials, RAG chunking parameters, and vector database maintenance.")

    # LLM Configuration
    st.markdown("### 🤖 LLM & Provider Configuration")
    provider = st.selectbox(
        "Active LLM Provider:",
        ["Gemini", "OpenAI", "Local Extractive Mode (Zero API Key)"],
        index=0 if LLM_PROVIDER == "gemini" else (1 if LLM_PROVIDER == "openai" else 2),
    )

    if provider == "Gemini":
        gemini_key = st.text_input(
            "Google Gemini API Key:",
            value=GEMINI_API_KEY,
            type="password",
            help="Get your free Gemini API key from Google AI Studio (aistudio.google.com)",
        )
        model_choice = st.selectbox("Gemini Model:", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"])
        if st.button("Save Gemini Settings"):
            os.environ["LLM_PROVIDER"] = "gemini"
            os.environ["GEMINI_API_KEY"] = gemini_key
            os.environ["GEMINI_MODEL"] = model_choice
            st.success("Gemini configuration saved for current session.")

    elif provider == "OpenAI":
        openai_key = st.text_input("OpenAI API Key:", value=OPENAI_API_KEY, type="password")
        model_choice = st.selectbox("OpenAI Model:", ["gpt-4o-mini", "gpt-4o"])
        if st.button("Save OpenAI Settings"):
            os.environ["LLM_PROVIDER"] = "openai"
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["OPENAI_MODEL"] = model_choice
            st.success("OpenAI configuration saved for current session.")

    else:
        st.info("Local Extractive Mode is active. No API key is needed. Answers are formed from verbatim evidence passages.")

    st.markdown("---")

    # RAG Parameters
    st.markdown("### 🧩 RAG Chunking & Retrieval Parameters")
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        chunk_size = st.slider("Chunk Size (characters):", 400, 1500, DEFAULT_CHUNK_SIZE, 50)
    with r_col2:
        chunk_overlap = st.slider("Chunk Overlap (characters):", 50, 400, DEFAULT_CHUNK_OVERLAP, 25)
    with r_col3:
        top_k = st.slider("Top-K Retrieved Chunks:", 2, 10, DEFAULT_TOP_K)

    st.markdown("---")

    # Maintenance & Cache
    st.markdown("### 🛠️ Maintenance & Cache Management")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        if st.button("🔄 Rebuild Metadata Sample Cache (1,000 papers)"):
            with st.spinner("Streaming sample from arxiv-metadata.json..."):
                sample = build_metadata_sample(sample_size=1000, force_rebuild=True)
                st.session_state.sample_papers = sample
                st.success(f"Successfully rebuilt cache with {len(sample)} papers!")
    with m_col2:
        if st.button("🗑️ Reset Vector Database"):
            vec_store = get_vector_store()
            if vec_store.db_path.exists():
                vec_store.db_path.unlink()
                vec_store._init_database()
            st.success("Vector database successfully cleared.")
            st.rerun()

    # System Diagnostics
    st.markdown("### 🖥️ System Diagnostics")
    st.markdown(
        f"""
        - **Python Version:** `{sys.version}`
        - **Vector Database:** `SQLite + Numpy (Persistent)` at `{vec_store.db_path}`
        - **PDF Storage:** `{PDF_DIR}`
        - **Local Extracted Cache:** `{BASE_DIR / 'storage' / 'cache'}`
        """
    )

