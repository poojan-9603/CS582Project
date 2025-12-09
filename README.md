CS582 Final Project – Hybrid Lexical + Semantic Retrieval

This project implements a hybrid information retrieval system that combines
BM25 (lexical search) with Sentence-BERT (semantic search) to retrieve
scientific abstracts from the SciFact dataset. The goal was to explore whether
a simple fusion of sparse and dense signals can improve retrieval performance
for scientific claim verification.

Live Streamlit demo:
https://poojan-9603-cs582project-appmain-t5npoi.streamlit.app/

GitHub repository:
https://github.com/poojan-9603/CS582Project

Getting Started
1. Create a virtual environment

Windows:
python -m venv .venv
..venv\Scripts\Activate.ps1

macOS / Linux:
python -m venv .venv
source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. (If embeddings are missing) Encode the SciFact corpus

python -m src.encode_corpus

This generates:

doc_embeddings.npy

doc_ids.json

Running Retrieval Models

BM25 (Lexical):
python -m src.retrieval_lexical

SBERT (Semantic):
python -m src.retrieval_semantic

Hybrid (BM25 + SBERT):
python -m src.retrieval_hybrid

Evaluation

Run all three retrieval systems and compute Precision@k, Recall@k, and NDCG@k:
python -m src.evaluate

Results will be saved to:
results/metrics_scifact.csv

Streamlit Demo (Local)

Run the interactive app:
streamlit run app/main.py

This lets you test BM25, SBERT, and Hybrid retrieval interactively.

Project Structure

src/ – retrieval models and evaluation code
app/ – Streamlit UI
results/ – evaluation outputs
data/ – SciFact dataset + embeddings (not included by default)
requirements.txt
README.md

Dataset

The SciFact dataset is not included.
Download it from: https://github.com/allenai/scifact

Place the dataset files under:
data/scifact/

Team

Poojan Patel
Sanjith Jayasankar