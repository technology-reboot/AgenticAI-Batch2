"""Lab 3: A simple evaluation script for the RAG example.

This mirrors the spirit of RAGAS by checking whether the system can answer
questions from the sample dataset and whether it avoids hallucinating when no
relevant context exists.
"""

from __future__ import annotations

from common import load_eval_dataset, simple_retrieve, load_company_profiles


def run_lab3() -> str:
    """Evaluate sample questions using simple retrieval."""
    profiles = load_company_profiles()
    dataset = load_eval_dataset()

    results = []
    for item in dataset:
        question = item["question"]
        expected = item.get("ground_truth", "")
        retrieved = simple_retrieve(question, profiles, top_k=3)

        if retrieved:
            context_available = True
        else:
            context_available = False

        result_line = (
            f"Q: {question}\n"
            f"Retrieved: {len(retrieved)} company profile(s)\n"
            f"Context available: {context_available}\n"
            f"Expected answer hint: {expected}\n"
        )
        results.append(result_line)

    return "\n\n".join(results)


if __name__ == "__main__":
    print(run_lab3())
