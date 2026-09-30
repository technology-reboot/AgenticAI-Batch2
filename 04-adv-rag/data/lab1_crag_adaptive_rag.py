"""Lab 1: Simple CRAG-style adaptive RAG demo.

This file shows the core ideas of:
1. Classifying a question into a route.
2. Retrieving relevant context from documents.
3. Grading that context before answering.

The implementation is intentionally simple so it is easy to read in training.
"""

from __future__ import annotations

from common import answer_with_llm, fallback_answer, load_company_profiles, simple_retrieve


def classify_query(query: str) -> str:
    """Classify the query into a simple route.

    Routes:
    - NO_RETRIEVAL: general knowledge questions.
    - SINGLE_HOP: questions about one company or one fact.
    - MULTI_HOP: questions that compare or connect multiple facts.
    """
    lowered = query.lower()

    if any(word in lowered for word in ["what is", "who is", "define", "capital"]):
        return "NO_RETRIEVAL"

    if any(word in lowered for word in ["compare", "difference", "between", "and"]):
        return "MULTI_HOP"

    return "SINGLE_HOP"


def grade_context(query: str, retrieved_docs: list[tuple[str, str]]) -> dict:
    """Score the retrieved context for usefulness.

    For training, we simply count if the company's name appears and whether
    the query contains related keywords.
    """
    query_words = set(query.lower().split())
    graded = []

    for company, content in retrieved_docs:
        content_words = set(content.lower().split())
        overlap = len(query_words & content_words)
        score = min(1.0, overlap / 10.0)

        if score >= 0.7:
            verdict = "CORRECT"
        elif score >= 0.3:
            verdict = "AMBIGUOUS"
        else:
            verdict = "INCORRECT"

        graded.append({"company": company, "score": round(score, 2), "verdict": verdict})

    overall = "CORRECT" if any(item["verdict"] == "CORRECT" for item in graded) else "AMBIGUOUS"
    return {"overall": overall, "items": graded}


def run_lab1(query: str) -> str:
    """Run the complete lab flow."""
    profiles = load_company_profiles()
    route = classify_query(query)

    if route == "NO_RETRIEVAL":
        answer = answer_with_llm(query)
        if answer:
            return f"Route: {route}\nAnswer: {answer}"
        return f"Route: {route}\nAnswer: {fallback_answer(query, profiles)}"

    retrieved = simple_retrieve(query, profiles, top_k=3)
    grading = grade_context(query, retrieved)

    if grading["overall"] == "CORRECT":
        context = "\n\n".join(f"[{company}]\n{content[:600]}" for company, content in retrieved)
    else:
        context = "No highly relevant document was retrieved."

    prompt = (
        "You are answering a question using the provided context.\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer briefly and truthfully."
    )

    answer = answer_with_llm(prompt)
    if answer:
        return f"Route: {route}\nGrading: {grading}\nAnswer: {answer}"

    return f"Route: {route}\nGrading: {grading}\nAnswer: {fallback_answer(query, profiles)}"


if __name__ == "__main__":
    sample_queries = [
        "What does ERP stand for?",
        "Who is the CEO of TCS?",
        "Compare TCS and Infosys founding years",
    ]

    for query in sample_queries:
        print("=" * 70)
        print(f"Query: {query}")
        print(run_lab1(query))
        print()
