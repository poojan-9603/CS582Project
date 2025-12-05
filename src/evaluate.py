import os
from typing import Dict, List, Tuple

from .data_loader import load_scifact, build_corpus_texts
from .retrieval_lexical import BM25Retriever
from .retrieval_semantic import SemanticRetriever
from .retrieval_hybrid import min_max_normalize

import numpy as np
import csv


# ---------------------------------------------------------------------
# IR METRICS
# ---------------------------------------------------------------------

def precision_at_k(ranked_docs: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_docs[:k]
    rel_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return rel_in_top_k / k


def recall_at_k(ranked_docs: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    if not relevant_docs:
        return 0.0
    top_k = ranked_docs[:k]
    rel_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return rel_in_top_k / len(relevant_docs)


def dcg_at_k(ranked_docs: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    """
    Compute DCG@k assuming relevance scores from qrels.
    """
    dcg = 0.0
    for i in range(k):
        if i >= len(ranked_docs):
            break
        doc_id = ranked_docs[i]
        rel = relevant_docs.get(doc_id, 0)
        if rel > 0:
            # common DCG variant: (2^rel - 1) / log2(i+2)
            dcg += (2**rel - 1) / np.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_docs: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    """
    NDCG@k = DCG@k / IDCG@k
    """
    if not relevant_docs:
        return 0.0

    dcg = dcg_at_k(ranked_docs, relevant_docs, k)

    # Ideal DCG: sort relevant docs by relevance descending
    rels_sorted = sorted(relevant_docs.values(), reverse=True)
    ideal_docs = rels_sorted[:k]

    idcg = 0.0
    for i, rel in enumerate(ideal_docs):
        idcg += (2**rel - 1) / np.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------
# HYBRID SEARCH FOR EVALUATION (reuses BM25 + Semantic retrievers)
# ---------------------------------------------------------------------

def hybrid_ranked_docs(
    query: str,
    bm25_retriever: BM25Retriever,
    semantic_retriever: SemanticRetriever,
    alpha: float = 0.5,
    bm25_top_k: int = 100,
    semantic_top_k: int = 100,
    final_top_k: int = 100,
) -> List[str]:
    """
    Run hybrid search and return a list of doc_ids sorted by hybrid score.

    This mirrors HybridRetriever.search(), but uses existing retrievers
    so we don't rebuild models during evaluation.
    """
    # BM25 candidates
    bm25_results = bm25_retriever.search(query, top_k=bm25_top_k)
    bm25_scores_raw: Dict[str, float] = {doc_id: score for doc_id, score in bm25_results}

    # Semantic candidates
    semantic_results = semantic_retriever.search(query, top_k=semantic_top_k)
    semantic_scores_raw: Dict[str, float] = {doc_id: score for doc_id, score in semantic_results}

    # Normalize scores separately
    bm25_norm = min_max_normalize(bm25_scores_raw)
    semantic_norm = min_max_normalize(semantic_scores_raw)

    all_doc_ids = set(bm25_norm.keys()) | set(semantic_norm.keys())

    results: List[Tuple[str, float]] = []
    for doc_id in all_doc_ids:
        b_score = bm25_norm.get(doc_id, 0.0)
        s_score = semantic_norm.get(doc_id, 0.0)
        hybrid_score = alpha * b_score + (1.0 - alpha) * s_score
        results.append((doc_id, hybrid_score))

    # Sort by hybrid score descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Return only doc_ids
    ranked_doc_ids = [doc_id for doc_id, _ in results[:final_top_k]]
    return ranked_doc_ids


# ---------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------

def evaluate_models(
    k_values: List[int] = [5, 10, 20],
    alpha: float = 0.5,
    bm25_top_k: int = 100,
    semantic_top_k: int = 100,
) -> None:
    """
    Evaluate BM25, Semantic, and Hybrid models on SciFact test split.
    Prints aggregate metrics and saves a CSV in results/.
    """
    # Load data
    print("Loading SciFact dataset...")
    corpus, queries, qrels = load_scifact(split="test")
    combined_texts = build_corpus_texts(corpus)

    print(f"Number of documents: {len(corpus)}")
    print(f"Number of queries:   {len(queries)}")

    # Build retrievers
    print("\nBuilding BM25 retriever...")
    bm25_retriever = BM25Retriever(combined_texts)

    print("Initializing Semantic retriever...")
    semantic_retriever = SemanticRetriever()

    # Models to evaluate
    model_names = ["bm25", "semantic", "hybrid"]

    # Metrics storage: {model: {k: [values per query]}}
    precision_scores = {m: {k: [] for k in k_values} for m in model_names}
    recall_scores = {m: {k: [] for k in k_values} for m in model_names}
    ndcg_scores = {m: {k: [] for k in k_values} for m in model_names}

    print("\nRunning evaluation for all queries...")
    query_ids = list(queries.keys())

    for qid in query_ids:
        query_text = queries[qid]
        relevant_docs = qrels.get(qid, {})

        # Skip queries with no relevant docs (shouldn't happen with SciFact)
        if not relevant_docs:
            continue

        # BM25 ranking
        bm25_ranked = [doc_id for doc_id, _ in bm25_retriever.search(query_text, top_k=bm25_top_k)]

        # Semantic ranking
        semantic_ranked = [doc_id for doc_id, _ in semantic_retriever.search(query_text, top_k=semantic_top_k)]

        # Hybrid ranking
        hybrid_ranked = hybrid_ranked_docs(
            query_text,
            bm25_retriever,
            semantic_retriever,
            alpha=alpha,
            bm25_top_k=bm25_top_k,
            semantic_top_k=semantic_top_k,
            final_top_k=max(bm25_top_k, semantic_top_k),
        )

        rankings = {
            "bm25": bm25_ranked,
            "semantic": semantic_ranked,
            "hybrid": hybrid_ranked,
        }

        for model in model_names:
            ranked_docs = rankings[model]
            for k in k_values:
                p = precision_at_k(ranked_docs, relevant_docs, k)
                r = recall_at_k(ranked_docs, relevant_docs, k)
                n = ndcg_at_k(ranked_docs, relevant_docs, k)

                precision_scores[model][k].append(p)
                recall_scores[model][k].append(r)
                ndcg_scores[model][k].append(n)

    # Aggregate metrics
    print("\n=== Aggregated Results ===")
    for model in model_names:
        print(f"\nModel: {model}")
        for k in k_values:
            p_mean = float(np.mean(precision_scores[model][k])) if precision_scores[model][k] else 0.0
            r_mean = float(np.mean(recall_scores[model][k])) if recall_scores[model][k] else 0.0
            n_mean = float(np.mean(ndcg_scores[model][k])) if ndcg_scores[model][k] else 0.0

            print(f"  k={k:2d} | P@k={p_mean:.4f} | R@k={r_mean:.4f} | NDCG@k={n_mean:.4f}")

    # Save to CSV
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "metrics_scifact.csv")

    print(f"\nSaving metrics to: {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "k", "precision", "recall", "ndcg"])
        for model in model_names:
            for k in k_values:
                p_mean = float(np.mean(precision_scores[model][k])) if precision_scores[model][k] else 0.0
                r_mean = float(np.mean(recall_scores[model][k])) if recall_scores[model][k] else 0.0
                n_mean = float(np.mean(ndcg_scores[model][k])) if ndcg_scores[model][k] else 0.0
                writer.writerow([model, k, f"{p_mean:.4f}", f"{r_mean:.4f}", f"{n_mean:.4f}"])

    print("Done.")


if __name__ == "__main__":
    # You can tweak k_values and alpha if you want to experiment
    evaluate_models(k_values=[5, 10, 20], alpha=0.5, bm25_top_k=100, semantic_top_k=100)
