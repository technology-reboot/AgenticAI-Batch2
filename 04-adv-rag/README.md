# Session 3B — Advanced RAG Training Lab

This folder contains a simple, readable training version of the advanced RAG lab.
The code is intentionally lightweight so it is easier to understand during a workshop.

## Folder structure

- `common.py` — shared helpers for loading data and simple retrieval
- `lab1_crag_adaptive_rag.py` — CRAG-style adaptive RAG demo
- `lab2_graph_rag.py` — graph-style reasoning over company facts
- `lab3_ragas_evaluation.py` — simple evaluation script for the sample dataset
- `data/` — company profile text files and evaluation data

## Setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Copy the environment template and add your key if you want OpenAI-based answers

```bash
copy .env.example .env
```

## Run the labs

### Lab 1

```bash
python lab1_crag_adaptive_rag.py
```

### Lab 2

```bash
python lab2_graph_rag.py
```

### Lab 3

```bash
python lab3_ragas_evaluation.py
```

## Notes

- The code uses a simple keyword-based retrieval approach instead of a full vector database.
- This keeps the lab easy to follow while still teaching the same core concepts.
- If no OpenAI key is set, the scripts fall back to a deterministic local answer.
