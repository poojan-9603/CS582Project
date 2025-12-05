import os
import json
import csv
from typing import Dict, Tuple


# --- Paths -----------------------------------------------------------

# Base path to your project (this file is in src/, so go one level up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the SciFact dataset inside data/
SCIFACT_PATH = os.path.join(PROJECT_ROOT, "data", "scifact")


# --- Loaders ---------------------------------------------------------

def load_corpus(
    dataset_path: str = SCIFACT_PATH,
) -> Dict[str, Dict[str, str]]:
    """
    Load BEIR-style corpus.jsonl into:
        {
          doc_id: {
            "title": str,
            "text": str
          },
          ...
        }
    """
    corpus_file = os.path.join(dataset_path, "corpus.jsonl")
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    corpus: Dict[str, Dict[str, str]] = {}

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj.get("_id")
            title = obj.get("title") or ""
            text = obj.get("text") or ""
            corpus[doc_id] = {"title": title, "text": text}

    return corpus


def load_queries(
    dataset_path: str = SCIFACT_PATH,
    split: str = "test",
) -> Dict[str, str]:
    """
    Load BEIR-style queries.jsonl into:
        {
          query_id: query_text,
          ...
        }

    We will later restrict to only the queries that appear in qrels[split].
    """
    queries_file = os.path.join(dataset_path, "queries.jsonl")
    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    queries: Dict[str, str] = {}

    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("_id")
            text = obj.get("text") or ""
            queries[qid] = text

    return queries


def load_qrels(
    dataset_path: str = SCIFACT_PATH,
    split: str = "test",
) -> Dict[str, Dict[str, int]]:
    """
    Load BEIR-style qrels/split.tsv into:
        {
          query_id: {
            doc_id: relevance_score (int),
            ...
          },
          ...
        }
    """
    qrels_file = os.path.join(dataset_path, "qrels", f"{split}.tsv")
    if not os.path.exists(qrels_file):
        raise FileNotFoundError(f"Qrels file not found: {qrels_file}")

    qrels: Dict[str, Dict[str, int]] = {}

    with open(qrels_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)  # Skip header row

        for row in reader:
            if len(row) < 3:
                continue
            qid, doc_id, score_str = row[0], row[1], row[2]
            score = int(score_str)

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = score

    return qrels


def load_scifact(
    split: str = "test",
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, Dict[str, int]],
]:
    """
    Convenience function: load the full SciFact dataset (one split).

    Returns:
        corpus, queries, qrels
    - corpus: {doc_id: {"title": str, "text": str}}
    - queries: {query_id: text} (only for queries that appear in qrels)
    - qrels: {query_id: {doc_id: relevance}}
    """
    corpus = load_corpus(SCIFACT_PATH)
    all_queries = load_queries(SCIFACT_PATH, split=split)
    qrels = load_qrels(SCIFACT_PATH, split=split)

    # Keep only queries that actually have qrels
    queries = {qid: all_queries[qid] for qid in qrels.keys() if qid in all_queries}

    return corpus, queries, qrels


# --- Helper for later stages ----------------------------------------

def build_corpus_texts(
    corpus: Dict[str, Dict[str, str]]
) -> Dict[str, str]:
    """
    For each document, concatenate title + text into a single string.
    This is what we'll feed into BM25 and SBERT.
    """
    combined: Dict[str, str] = {}

    for doc_id, fields in corpus.items():
        title = fields.get("title") or ""
        text = fields.get("text") or ""
        # Simple concatenation: "title. text"
        if title:
            combined[doc_id] = f"{title}. {text}"
        else:
            combined[doc_id] = text

    return combined


if __name__ == "__main__":
    # Simple sanity check: load the SciFact test split and print stats
    corpus, queries, qrels = load_scifact(split="test")

    print("Loaded SciFact dataset (test split):")
    print(f"  # Documents: {len(corpus)}")
    print(f"  # Queries with qrels: {len(queries)}")

    # Pick one random query (first key) and show its info
    sample_qid = next(iter(queries.keys()))
    print("\nExample query ID:", sample_qid)
    print("Query text:", queries[sample_qid])
    print("Relevant docs:", list(qrels[sample_qid].keys())[:5])
