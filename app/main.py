import os
import sys
from typing import List, Tuple, Dict

import streamlit as st

# Make sure we can import from src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.data_loader import load_scifact, build_corpus_texts
from src.retrieval_lexical import BM25Retriever
from src.retrieval_semantic import SemanticRetriever
from src.retrieval_hybrid import min_max_normalize


# --- Cached loading of models & data --------------------------------

@st.cache_resource
def load_resources():
    """
    Load corpus and build BM25 + Semantic retrievers.
    Cached so Streamlit doesn't rebuild on every interaction.
    """
    st.write("Loading SciFact dataset and building retrievers (first run only)...")
    corpus, queries, qrels = load_scifact(split="test")
    combined_texts = build_corpus_texts(corpus)

    bm25 = BM25Retriever(combined_texts)
    semantic = SemanticRetriever()

    return corpus, queries, qrels, bm25, semantic


def hybrid_search(
    query: str,
    bm25_retriever: BM25Retriever,
    semantic_retriever: SemanticRetriever,
    alpha: float = 0.5,
    bm25_top_k: int = 100,
    semantic_top_k: int = 100,
    final_top_k: int = 10,
) -> List[Tuple[str, float, float, float]]:
    """
    Run hybrid search and return:
        [(doc_id, hybrid_score, bm25_norm, semantic_norm), ...]
    sorted by hybrid_score.
    """
    # BM25 candidates
    bm25_results = bm25_retriever.search(query, top_k=bm25_top_k)
    bm25_scores_raw: Dict[str, float] = {doc_id: score for doc_id, score in bm25_results}

    # Semantic candidates
    semantic_results = semantic_retriever.search(query, top_k=semantic_top_k)
    semantic_scores_raw: Dict[str, float] = {doc_id: score for doc_id, score in semantic_results}

    # Normalize
    bm25_norm = min_max_normalize(bm25_scores_raw)
    semantic_norm = min_max_normalize(semantic_scores_raw)

    # Union of all candidate doc_ids
    all_doc_ids = set(bm25_norm.keys()) | set(semantic_norm.keys())

    results: List[Tuple[str, float, float, float]] = []
    for doc_id in all_doc_ids:
        b_score = bm25_norm.get(doc_id, 0.0)
        s_score = semantic_norm.get(doc_id, 0.0)
        hybrid_score = alpha * b_score + (1.0 - alpha) * s_score
        results.append((doc_id, hybrid_score, b_score, s_score))

    # Sort by hybrid score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:final_top_k]


# --- Streamlit UI ---------------------------------------------------

def main():
    st.title("Hybrid Lexical + Semantic Search Demo (SciFact)")

    st.markdown(
        """
        This demo compares three retrieval modes on the **SciFact** dataset:

        - **BM25** (lexical, keyword-based)
        - **Semantic** (Sentence-BERT embeddings + cosine similarity)
        - **Hybrid** (weighted combination of BM25 + Semantic scores)

        Use the controls on the sidebar to select a mode and number of results.
        """
    )

    with st.sidebar:
        st.header("Settings")

        mode = st.selectbox(
            "Retrieval mode",
            ["BM25 (Lexical)", "Semantic (SBERT)", "Hybrid (BM25 + SBERT)"],
        )

        top_k = st.slider("Top-k results to show", min_value=3, max_value=20, value=10, step=1)

        alpha = st.slider(
            "Hybrid weight α (BM25 importance)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="α = 1.0 → only BM25, α = 0.0 → only Semantic",
        )

        st.markdown("---")
        st.caption("Note: models load on first run, then cached.")

    # Load data + models
    corpus, queries, qrels, bm25_retriever, semantic_retriever = load_resources()

    # Query input
    default_example = next(iter(queries.values()))
    query = st.text_input(
        "Enter your query:",
        value=default_example,
        help="You can also paste your own question in natural language.",
    )

    if st.button("Search") or query.strip():
        query = query.strip()
        if not query:
            st.warning("Please enter a query.")
            return

        st.markdown(f"### Results for query: `{query}`")

        if mode.startswith("BM25"):
            results = bm25_retriever.search(query, top_k=top_k)

            for rank, (doc_id, score) in enumerate(results, start=1):
                doc = corpus[doc_id]
                title = doc.get("title") or "(no title)"
                text = doc.get("text") or ""
                snippet = text[:400].replace("\n", " ")

                st.markdown(f"**Rank {rank}** — DocID: `{doc_id}`")
                st.markdown(f"**BM25 score:** `{score:.4f}`")
                st.markdown(f"**Title:** {title}")
                st.markdown(f"{snippet}...")
                st.markdown("---")

        elif mode.startswith("Semantic"):
            results = semantic_retriever.search(query, top_k=top_k)

            for rank, (doc_id, score) in enumerate(results, start=1):
                doc = corpus[doc_id]
                title = doc.get("title") or "(no title)"
                text = doc.get("text") or ""
                snippet = text[:400].replace("\n", " ")

                st.markdown(f"**Rank {rank}** — DocID: `{doc_id}`")
                st.markdown(f"**Cosine score:** `{score:.4f}`")
                st.markdown(f"**Title:** {title}")
                st.markdown(f"{snippet}...")
                st.markdown("---")

        else:  # Hybrid
            results = hybrid_search(
                query,
                bm25_retriever=bm25_retriever,
                semantic_retriever=semantic_retriever,
                alpha=alpha,
                bm25_top_k=max(50, top_k),
                semantic_top_k=max(50, top_k),
                final_top_k=top_k,
            )

            for rank, (doc_id, hybrid_score, b_norm, s_norm) in enumerate(results, start=1):
                doc = corpus[doc_id]
                title = doc.get("title") or "(no title)"
                text = doc.get("text") or ""
                snippet = text[:400].replace("\n", " ")

                st.markdown(f"**Rank {rank}** — DocID: `{doc_id}`")
                st.markdown(
                    f"**Hybrid score:** `{hybrid_score:.4f}` "
                    f"(BM25 norm: `{b_norm:.3f}`, Semantic norm: `{s_norm:.3f}`)"
                )
                st.markdown(f"**Title:** {title}")
                st.markdown(f"{snippet}...")
                st.markdown("---")


if __name__ == "__main__":
    main()
