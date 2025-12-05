import os
import json
from typing import List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from .data_loader import load_scifact


# --- Paths -----------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCIFACT_PATH = os.path.join(PROJECT_ROOT, "data", "scifact")

EMBEDDINGS_PATH = os.path.join(SCIFACT_PATH, "doc_embeddings.npy")
DOC_IDS_PATH = os.path.join(SCIFACT_PATH, "doc_ids.json")


# --- Semantic Retriever ----------------------------------------------

class SemanticRetriever:
    """
    Dense (semantic) retriever using:
    - Precomputed document embeddings (SBERT)
    - On-the-fly query encoding with SBERT
    - Cosine similarity for ranking
    """

    def __init__(self):
        # Load doc_ids and embeddings
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(
                f"Embeddings not found: {EMBEDDINGS_PATH}. "
                f"Run `python -m src.encode_corpus` first."
            )

        if not os.path.exists(DOC_IDS_PATH):
            raise FileNotFoundError(
                f"Doc IDs file not found: {DOC_IDS_PATH}. "
                f"Run `python -m src.encode_corpus` first."
            )

        print("Loading document embeddings from disk...")
        self.doc_embeddings = np.load(EMBEDDINGS_PATH)  # shape: [num_docs, dim]

        print("Loading document IDs...")
        with open(DOC_IDS_PATH, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        if self.doc_embeddings.shape[0] != len(self.doc_ids):
            raise ValueError(
                "Mismatch between # of embeddings and # of doc_ids "
                f"({self.doc_embeddings.shape[0]} vs {len(self.doc_ids)})"
            )

        # Precompute norms for cosine similarity
        self.doc_norms = np.linalg.norm(self.doc_embeddings, axis=1) + 1e-10

        # Load the same SBERT model used for encoding
        print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Encode query, compute cosine similarity with all documents,
        and return top_k (doc_id, score) pairs sorted by score descending.
        """
        # Encode query to embedding
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=False
        )[0]  # shape: [dim]

        # Cosine similarity: (d · q) / (||d|| * ||q||)
        query_norm = np.linalg.norm(query_embedding) + 1e-10
        dot_products = np.dot(self.doc_embeddings, query_embedding)  # [num_docs]
        cosine_scores = dot_products / (self.doc_norms * query_norm)

        # Get top_k indices
        ranked_indices = np.argsort(-cosine_scores)  # descending sort
        top_indices = ranked_indices[:top_k]

        results: List[Tuple[str, float]] = []
        for idx in top_indices:
            doc_id = self.doc_ids[int(idx)]
            score = float(cosine_scores[int(idx)])
            results.append((doc_id, score))

        return results


# --- Sanity-check main ----------------------------------------------

if __name__ == "__main__":
    print("Loading SciFact dataset (for titles/snippets)...")
    corpus, queries, qrels = load_scifact(split="test")

    print("Initializing SemanticRetriever...")
    retriever = SemanticRetriever()

    # Take the same sample query (first in dict)
    sample_qid = next(iter(queries.keys()))
    sample_query = queries[sample_qid]

    print("\nSample query:")
    print(f"  ID:   {sample_qid}")
    print(f"  Text: {sample_query}")

    results = retriever.search(sample_query, top_k=5)

    print("\nTop-5 Semantic (SBERT) results:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        doc = corpus[doc_id]
        title = doc.get("title") or ""
        text = doc.get("text") or ""
        snippet = text[:200].replace("\n", " ")

        print(f"\nRank {rank} | DocID {doc_id} | Cosine {score:.4f}")
        print(f"  Title: {title}")
        print(f"  Snippet: {snippet}...")
