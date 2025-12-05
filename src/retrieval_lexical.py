import os
import re
from typing import List, Tuple, Dict

from rank_bm25 import BM25Okapi

from .data_loader import load_scifact, build_corpus_texts


# --- Simple tokenizer ------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer:
    - lowercase
    - keep only alphanumeric 'words'
    """
    return re.findall(r"\w+", text.lower())


# --- BM25 Wrapper ----------------------------------------------------

class BM25Retriever:
    """
    Thin wrapper around rank_bm25.BM25Okapi to:
    - build an index from document texts
    - run queries and return (doc_id, score) pairs
    """

    def __init__(self, doc_texts: Dict[str, str]):
        """
        Args:
            doc_texts: {doc_id: full_text}
        """
        self.doc_ids: List[str] = list(doc_texts.keys())
        self.tokenized_docs: List[List[str]] = [
            tokenize(doc_texts[doc_id]) for doc_id in self.doc_ids
        ]

        # Create BM25 model
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Run BM25 search for a single query string.

        Returns:
            List of (doc_id, score), sorted by score descending.
        """
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # scores is a list aligned with self.doc_ids
        # We sort indices by score
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        results: List[Tuple[str, float]] = []
        for idx in ranked_indices[:top_k]:
            doc_id = self.doc_ids[idx]
            score = float(scores[idx])
            results.append((doc_id, score))

        return results


# --- Convenience builder ---------------------------------------------

def build_bm25_retriever() -> Tuple[BM25Retriever, Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Load SciFact, build combined doc texts, and return:
        - BM25Retriever instance
        - corpus (with title/text)
        - queries (text)
    This is mainly for quick experiments; in main code, you'll likely
    want to manage loading/building separately.
    """
    corpus, queries, qrels = load_scifact(split="test")
    combined_texts = build_corpus_texts(corpus)

    retriever = BM25Retriever(combined_texts)
    return retriever, corpus, queries


# --- Sanity-check main ----------------------------------------------

if __name__ == "__main__":
    print("Loading SciFact dataset and building BM25 index...")
    retriever, corpus, queries = build_bm25_retriever()

    print(f"Number of documents indexed: {len(corpus)}")
    print(f"Number of queries: {len(queries)}")

    # Take the first query and search
    sample_qid = next(iter(queries.keys()))
    sample_query = queries[sample_qid]

    print("\nSample query:")
    print(f"  ID:   {sample_qid}")
    print(f"  Text: {sample_query}")

    results = retriever.search(sample_query, top_k=5)

    print("\nTop-5 BM25 results:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        doc = corpus[doc_id]
        title = doc.get("title") or ""
        text = doc.get("text") or ""
        snippet = text[:200].replace("\n", " ")

        print(f"\nRank {rank} | DocID {doc_id} | Score {score:.4f}")
        print(f"  Title: {title}")
        print(f"  Snippet: {snippet}...")
