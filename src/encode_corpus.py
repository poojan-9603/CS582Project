import os
import json
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .data_loader import load_scifact, build_corpus_texts


# --- Paths -----------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCIFACT_PATH = os.path.join(PROJECT_ROOT, "data", "scifact")

EMBEDDINGS_PATH = os.path.join(SCIFACT_PATH, "doc_embeddings.npy")
DOC_IDS_PATH = os.path.join(SCIFACT_PATH, "doc_ids.json")


def encode_corpus(batch_size: int = 32) -> None:
    """
    Load SciFact corpus, build combined text, encode with SBERT, and save:
        - doc_embeddings.npy : float32 [num_docs, dim]
        - doc_ids.json       : list of doc_ids in same order
    """
    print("Loading SciFact corpus...")
    corpus, queries, qrels = load_scifact(split="test")
    combined_texts: Dict[str, str] = build_corpus_texts(corpus)

    doc_ids: List[str] = list(combined_texts.keys())
    texts: List[str] = [combined_texts[doc_id] for doc_id in doc_ids]

    print(f"Number of documents to encode: {len(doc_ids)}")

    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding documents into embeddings...")
    # encode() can take a list of strings and return a numpy array
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll handle normalization at search time
    )

    print("Embeddings shape:", embeddings.shape)

    # Save to disk
    print(f"Saving embeddings to: {EMBEDDINGS_PATH}")
    np.save(EMBEDDINGS_PATH, embeddings.astype("float32"))

    print(f"Saving doc_ids to: {DOC_IDS_PATH}")
    with open(DOC_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    print("Done encoding and saving corpus embeddings.")


if __name__ == "__main__":
    encode_corpus()
