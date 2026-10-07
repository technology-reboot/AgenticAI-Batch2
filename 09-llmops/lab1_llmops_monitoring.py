"""
Lab 1 - LLMOps monitoring dashboard: drift detection & alerting.

A lightweight monitoring loop around a small RAG agent:
  1. Run the RAG agent over a 20-query evaluation set and record
     faithfulness, latency and cost per request.
  2. Store the metrics in a pandas DataFrame and compute a baseline.
  3. Simulate drift: from --drift-start onwards, replace 5 queries with
     out-of-domain questions and re-run.
  4. check_drift() compares each window to the baseline and fires an alert
     for every threshold that is breached.
  5. Plot faithfulness and p95 latency over time as a 2-panel figure.

The RAG agent uses the company-profile corpus from Module 3 (03-rag).

Install:  pip install openai python-dotenv pandas matplotlib numpy
Run:      python lab1_llmops_monitoring.py [--windows 6] [--drift-start 3]

Annotation task (students): from the printed alerts and the plot, note
  - which metric degraded first, and
  - which remediation you would trigger for it:
      RAG index refresh  -> retrieval is stale or missing the facts
      prompt update      -> answers stop following the context
      model swap         -> same prompt and context, but quality or cost regressed
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ============================================================================
# CONFIG
# ============================================================================

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# USD per 1M tokens. Check current pricing before relying on the cost numbers.
PRICE_PER_M = {
    CHAT_MODEL: {"input": 0.15, "output": 0.60},
    EMBED_MODEL: {"input": 0.02},
}

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "03-rag" / "data" / "company_profiles"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs"
TOP_K = 3
CHUNK_CHARS = 600

# Alert thresholds, relative to the baseline window.
THRESHOLDS = {
    "faithfulness": {"direction": "drop", "limit_pct": 10.0},
    "latency_p95_ms": {"direction": "rise", "limit_pct": 25.0},
    "cost_per_request_usd": {"direction": "rise", "limit_pct": 25.0},
}

# 20-query evaluation set, all answerable from the company profiles.
EVAL_SET = [
    "Who is the CEO of TCS?",
    "When was Infosys founded and by whom?",
    "What is Infosys revenue growth guidance for FY2025?",
    "In how many countries does Wipro serve clients?",
    "Where is HCLTech headquartered?",
    "Which group does Tech Mahindra belong to?",
    "Compare the founding years of TCS and Infosys.",
    "Which of these companies is headquartered in Pune?",
    "Which company started as a vegetable products business?",
    "Where is Wipro headquartered?",
    "Who leads HCLTech?",
    "When did Mohit Joshi become CEO of Tech Mahindra?",
    "How many employees does TCS have?",
    "Where is TCS headquartered?",
    "Which company is part of the Tata Group?",
    "What is the headquarters of Tech Mahindra?",
    "Who is the chief executive of Infosys?",
    "Which company was founded in 1945?",
    "What sectors does TCS serve?",
    "Which company was founded in 1976?",
]

# Out-of-domain queries used to simulate drift.
OOD_SET = [
    "Who won the FIFA World Cup in 2022?",
    "What is the boiling point of water at sea level?",
    "Give me a recipe for banana bread.",
    "Who wrote the novel Pride and Prejudice?",
    "What is the capital of Australia?",
]

# Deliberate knob: the prompt lets the model fall back on its own knowledge
# when the context is thin. In-domain queries stay grounded. Out-of-domain
# queries get answered from memory, which lowers faithfulness.
ANSWER_PROMPT = """Use the context below to help you answer the question.
If the context is not enough, you may answer from your general knowledge.
Answer in 2-3 sentences.

Context:
{context}

Question: {question}

Answer:"""

JUDGE_PROMPT = """You are checking whether an answer is grounded in the context.
Split the answer into atomic factual claims. Count how many are directly supported by the context.
An answer with no factual claims (for example a refusal) has total_claims = 0.

Context:
{context}

Answer:
{answer}

Reply with JSON only: {{"total_claims": <int>, "supported_claims": <int>}}"""


# ============================================================================
# HELPERS
# ============================================================================

def cost_of(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """USD cost of one API call from its token usage."""
    price = PRICE_PER_M[model]
    return (input_tokens * price["input"] + output_tokens * price.get("output", 0.0)) / 1_000_000


def load_chunks(corpus_dir: Path) -> list[dict]:
    """Split each company profile into paragraph-based chunks of about CHUNK_CHARS."""
    chunks = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        buf = ""
        for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
            if not para:
                continue
            if buf and len(buf) + len(para) > CHUNK_CHARS:
                chunks.append({"source": path.stem, "text": buf})
                buf = ""
            buf = f"{buf}\n\n{para}".strip()
        if buf:
            chunks.append({"source": path.stem, "text": buf})
    return chunks


def build_queries(drifted: bool) -> list[dict]:
    """Return the evaluation set. With drift, positions 0-4 become out-of-domain."""
    queries = [{"text": q, "type": "in_domain"} for q in EVAL_SET]
    if drifted:
        for i, ood in enumerate(OOD_SET):
            queries[i] = {"text": ood, "type": "out_of_domain"}
    return queries


# ============================================================================
# RAG AGENT (the system being monitored)
# ============================================================================

class RAGAgent:
    def __init__(self, client: OpenAI, chunks: list[dict]):
        self.client = client
        self.chunks = chunks
        self.chunk_vecs, _ = self._embed([c["text"] for c in chunks])

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, float]:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs = np.array([d.embedding for d in resp.data])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs, cost_of(EMBED_MODEL, resp.usage.prompt_tokens)

    def answer(self, question: str) -> dict:
        """Retrieve, generate, and time the request. Returns answer, contexts, latency and cost."""
        t0 = time.perf_counter()
        q_vec, embed_cost = self._embed([question])
        top = np.argsort(-(self.chunk_vecs @ q_vec[0]))[:TOP_K]
        contexts = [self.chunks[i]["text"] for i in top]

        prompt = ANSWER_PROMPT.format(context="\n\n---\n\n".join(contexts), question=question)
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        cost = embed_cost + cost_of(CHAT_MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return {
            "answer": resp.choices[0].message.content.strip(),
            "contexts": contexts,
            "latency_ms": latency_ms,
            "cost_usd": cost,
        }


# ============================================================================
# EVALUATION (faithfulness via LLM judge)
# ============================================================================

def faithfulness(client: OpenAI, answer: str, contexts: list[str]) -> tuple[float, float]:
    """Share of answer claims supported by the context. Returns (score, judge cost)."""
    prompt = JUDGE_PROMPT.format(context="\n\n---\n\n".join(contexts), answer=answer)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    cost = cost_of(CHAT_MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    try:
        data = json.loads(resp.choices[0].message.content)
        total, supported = int(data["total_claims"]), int(data["supported_claims"])
    except (ValueError, KeyError, TypeError):
        return float("nan"), cost
    return (1.0 if total == 0 else supported / total), cost


def run_window(agent: RAGAgent, client: OpenAI, queries: list[dict], window: int, phase: str) -> list[dict]:
    """Run every query once and log one row per request."""
    rows = []
    for q in queries:
        out = agent.answer(q["text"])
        score, judge_cost = faithfulness(client, out["answer"], out["contexts"])
        rows.append({
            "window": window,
            "phase": phase,
            "query_type": q["type"],
            "query": q["text"],
            "answer": out["answer"],
            "faithfulness": score,
            "latency_ms": out["latency_ms"],
            # Includes the judge call, since that is part of the cost of monitoring each request.
            "cost_usd": out["cost_usd"] + judge_cost,
        })
    return rows


# ============================================================================
# DRIFT DETECTION & ALERTING
# ============================================================================

def summarize(rows: pd.DataFrame) -> dict:
    """Window-level metrics: mean faithfulness, p95 latency, mean cost per request."""
    return {
        "faithfulness": rows["faithfulness"].mean(),
        "latency_p95_ms": rows["latency_ms"].quantile(0.95),
        "cost_per_request_usd": rows["cost_usd"].mean(),
    }


def check_drift(current_metrics: dict, baseline: dict, thresholds: dict = THRESHOLDS) -> list[dict]:
    """Return one alert dict for each threshold the current window breaches."""
    alerts = []
    for metric, rule in thresholds.items():
        base, cur = baseline[metric], current_metrics[metric]
        if pd.isna(cur) or not base:
            continue
        if rule["direction"] == "drop":
            breach_pct = (base - cur) / base * 100
        else:
            breach_pct = (cur - base) / base * 100
        if breach_pct >= rule["limit_pct"]:
            alerts.append({
                "metric": metric,
                "current_value": round(float(cur), 4),
                "baseline_val": round(float(base), 4),
                "breach_pct": round(float(breach_pct), 1),
            })
    return alerts


def plot_metrics(window_df: pd.DataFrame, out_path: Path, drift_start: int) -> None:
    """2-panel figure: faithfulness (top) and p95 latency (bottom) per window."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    w = window_df["window"]

    ax1.plot(w, window_df["faithfulness"], marker="o", color="tab:blue")
    ax1.set_ylabel("Faithfulness (mean)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("RAG monitoring: quality and latency over time")

    ax2.plot(w, window_df["latency_p95_ms"], marker="o", color="tab:orange")
    ax2.set_ylabel("p95 latency (ms)")
    ax2.set_xlabel("Monitoring window")
    ax2.set_xticks(list(w))

    for ax in (ax1, ax2):
        if drift_start < len(window_df):
            ax.axvline(drift_start - 0.5, linestyle="--", color="grey")
        ax.grid(alpha=0.3)
    if drift_start < len(window_df):
        ax1.text(drift_start - 0.4, 0.05, "drift injected", color="grey")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================================
# MAIN LOOP
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="LLMOps drift monitoring lab")
    parser.add_argument("--windows", type=int, default=6, help="number of monitoring windows")
    parser.add_argument("--drift-start", type=int, default=3, help="first window with drift injected")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="company profiles directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="where to write CSVs and the plot")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing. Add it to your .env file.")
    client = OpenAI(api_key=api_key)

    chunks = load_chunks(args.corpus)
    print(f"Loaded {len(chunks)} chunks from {args.corpus}")
    agent = RAGAgent(client, chunks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    window_metrics: list[dict] = []
    baseline: dict | None = None

    for w in range(args.windows):
        drifted = w >= args.drift_start
        phase = "drift" if drifted else "clean"
        print(f"\n=== Window {w} ({phase}) ===")

        rows = run_window(agent, client, build_queries(drifted), w, phase)
        all_rows.extend(rows)
        metrics = summarize(pd.DataFrame(rows))
        window_metrics.append({"window": w, "phase": phase, **metrics})

        print(
            f"faithfulness={metrics['faithfulness']:.3f}  "
            f"p95_latency={metrics['latency_p95_ms']:.0f}ms  "
            f"cost/req=${metrics['cost_per_request_usd']:.5f}"
        )

        if baseline is None:
            baseline = metrics
            print("baseline set from this window")
            continue

        for alert in check_drift(metrics, baseline):
            print(f"[ALERT] {alert}")

    query_log = pd.DataFrame(all_rows)
    window_df = pd.DataFrame(window_metrics)
    query_log.to_csv(args.out_dir / "lab1_query_log.csv", index=False)
    window_df.to_csv(args.out_dir / "lab1_window_metrics.csv", index=False)

    plot_path = args.out_dir / "lab1_monitoring.png"
    plot_metrics(window_df, plot_path, args.drift_start)
    print(f"\nSaved query log, window metrics and plot to {args.out_dir}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
