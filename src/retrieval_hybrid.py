import os
from typing import List, Tuple, Dict

from .data_loader import load_scifact
from .data_loader import build_corpus_texts
from .retrieval_lexical import BM25Retriever
from .retrieval_semantic import SemanticRetriever


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Min-max normalize a dict of {id: score} to [0, 1].
    If all scores are equal, return 0.0 for all.
    """
    if not scores:
        return {}

    values = list(scores.values())
    s_min = min(values)
    s_max = max(values)

    if s_max - s_min < 1e-10:
        # All scores identical
        return {k: 0.0 for k in scores.keys()}

    return {k: (v - s_min) / (s_max - s_min) for k, v in scores.items()}


class HybridRetriever:
    """
    Hybrid retriever combining:
    - BM25 (lexical)
    - SBERT (semantic)
    via weighted sum of normalized scores.
    """

    def __init__(self, alpha: float = 0.5, bm25_top_k: int = 100, semantic_top_k: int = 100):
        """
        Args:
            alpha: weight for BM25 in [0, 1]. (1 - alpha) is weight for semantic.
            bm25_top_k: how many BM25 candidates to consider before fusion.
            semantic_top_k: how many semantic candidates to consider before fusion.
        """
        self.alpha = alpha
        self.bm25_top_k = bm25_top_k
        self.semantic_top_k = semantic_top_k

        # Load dataset once
        self.corpus, self.queries, self.qrels = load_scifact(split="test")
        combined_texts = build_corpus_texts(self.corpus)

        # Build BM25 and semantic retrievers
        print("Building BM25 retriever for hybrid...")
        self.bm25_retriever = BM25Retriever(combined_texts)

        print("Initializing Semantic retriever for hybrid...")
        self.semantic_retriever = SemanticRetriever()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, float, float]]:
        """
        Run hybrid search.

        Returns:
            List of (doc_id, hybrid_score, bm25_score_norm, semantic_score_norm)
            sorted by hybrid_score descending.
        """
        # --- BM25 scores ---
        bm25_results = self.bm25_retriever.search(query, top_k=self.bm25_top_k)
        bm25_scores_raw: Dict[str, float] = {doc_id: score for doc_id, score in bm25_results}

        # --- Semantic scores ---
        semantic_results = self.semantic_retriever.search(query, top_k=self.semantic_top_k)
        semantic_scores_raw: Dict[str, float] = {doc_id: score for doc_id, score in semantic_results}

        # --- Normalize both sets of scores separately ---
        bm25_norm = min_max_normalize(bm25_scores_raw)
        semantic_norm = min_max_normalize(semantic_scores_raw)

        # --- Combine doc IDs ---
        all_doc_ids = set(bm25_norm.keys()) | set(semantic_norm.keys())

        # --- Weighted sum fusion ---
        alpha = self.alpha
        results: List[Tuple[str, float, float, float]] = []

        for doc_id in all_doc_ids:
            b_score = bm25_norm.get(doc_id, 0.0)
            s_score = semantic_norm.get(doc_id, 0.0)
            hybrid_score = alpha * b_score + (1.0 - alpha) * s_score
            results.append((doc_id, hybrid_score, b_score, s_score))

        # Sort by hybrid score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return only top_k
        return results[:top_k]


if __name__ == "__main__":
    print("Loading SciFact and building HybridRetriever...")
    hybrid = HybridRetriever(alpha=0.5, bm25_top_k=100, semantic_top_k=100)

    corpus = hybrid.corpus
    queries = hybrid.queries

    sample_qid = next(iter(queries.keys()))
    sample_query = queries[sample_qid]

    print("\nSample query:")
    print(f"  ID:   {sample_qid}")
    print(f"  Text: {sample_query}")

    results = hybrid.search(sample_query, top_k=5)

    print("\nTop-5 Hybrid results (alpha=0.5):")
    for rank, (doc_id, hybrid_score, b_score, s_score) in enumerate(results, start=1):
        doc = corpus[doc_id]
        title = doc.get("title") or ""
        text = doc.get("text") or ""
        snippet = text[:200].replace("\n", " ")

        print(f"\nRank {rank} | DocID {doc_id}")
        print(f"  Hybrid score:   {hybrid_score:.4f}")
        print(f"  BM25 norm:      {b_score:.4f}")
        print(f"  Semantic norm:  {s_score:.4f}")
        print(f"  Title: {title}")
        print(f"  Snippet: {snippet}...")
