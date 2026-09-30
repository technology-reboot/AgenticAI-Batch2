"""Shared helpers for the training RAG labs.

This file keeps the examples simple and readable. The code uses plain Python
logic instead of heavy framework wiring so learners can focus on the concepts.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COMPANY_PROFILES_DIR = DATA_DIR / "company_profiles"
EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"


def load_company_profiles() -> Dict[str, str]:
    """Load all company profile text files into a dictionary."""
    profiles: Dict[str, str] = {}
    for path in sorted(COMPANY_PROFILES_DIR.glob("*.txt")):
        company_name = path.stem.replace("_", " ").title()
        profiles[company_name] = path.read_text(encoding="utf-8")
    return profiles


def load_eval_dataset() -> List[dict]:
    """Load the sample evaluation dataset."""
    return json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    """Lowercase and remove extra whitespace/punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_keywords(text: str) -> set[str]:
    """Convert a piece of text into a simple keyword set."""
    return set(normalize_text(text).split())


def simple_retrieve(query: str, profiles: Dict[str, str], top_k: int = 3) -> List[Tuple[str, str]]:
    """Retrieve the most relevant profile snippets using keyword overlap.

    This is intentionally simple and transparent for training purposes.
    """
    query_terms = extract_keywords(query)
    if not query_terms:
        return []

    scored: List[Tuple[int, str, str]] = []
    for company, profile_text in profiles.items():
        profile_terms = extract_keywords(profile_text)
        overlap = len(query_terms & profile_terms)
        if overlap > 0:
            scored.append((overlap, company, profile_text))

    scored.sort(reverse=True)
    ranked = []
    for _, company, profile_text in scored[:top_k]:
        ranked.append((company, profile_text))
    return ranked


def build_simple_graph(profiles: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Create a small knowledge graph-like structure from profile text.

    Each company is a node and the graph stores a few key facts as edges.
    This mirrors graph-based reasoning without introducing complex dependencies.
    """
    graph: Dict[str, Dict[str, str]] = {}
    for company, profile_text in profiles.items():
        facts: Dict[str, str] = {}
        for label, pattern in {
            "founded": r"founded\s+(?:in\s+)?(\d{4})",
            "headquarters": r"hq[:\- ]+([A-Za-z .]+)",
            "ceo": r"ceo[:\- ]+([A-Za-z .]+)",
            "revenue": r"revenue(?:\s+of)?(?:\s+approximately)?\s*[:\- ]*([₹$A-Za-z0-9, .]+)",
        }.items():
            match = re.search(pattern, profile_text, flags=re.IGNORECASE)
            if match:
                facts[label] = match.group(1).strip()
        graph[company] = facts
    return graph


def answer_with_llm(prompt: str) -> str | None:
    """Try using OpenAI if an API key is available; otherwise return None."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover - optional dependency
        return None

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def fallback_answer(query: str, profiles: Dict[str, str], graph: Dict[str, Dict[str, str]] | None = None) -> str:
    """Create a simple deterministic answer when no LLM is hooked up."""
    normalized_query = normalize_text(query)

    def find_company() -> str | None:
        for company in profiles:
            company_key = normalize_text(company)
            if company_key and company_key in normalized_query:
                return company
        return None

    company = find_company()

    if "ceo" in normalized_query and company:
        if graph and company in graph:
            return f"{company} is led by {graph[company].get('ceo', 'a listed executive')}"
        return f"{company} is led by a listed executive in the training profile."

    if ("headquarter" in normalized_query or "hq" in normalized_query) and company:
        if graph and company in graph:
            return f"{company} is headquartered in {graph[company].get('headquarters', 'an Indian city')}"
        return f"{company} is headquartered in a listed Indian city."

    if "founded" in normalized_query and company:
        if graph and company in graph:
            return f"{company} was founded in {graph[company].get('founded', 'the reported year')}"
        return f"{company} has a listed founding year in the training profile."

    if "revenue" in normalized_query and company:
        if graph and company in graph:
            return f"{company} reported revenue of {graph[company].get('revenue', 'a reported amount')}"
        return f"{company} has a listed revenue figure in the training profile."

    # Generic fallback: return the first matching profile summary.
    for profile_company, profile_text in profiles.items():
        if normalize_text(profile_company) in normalized_query:
            return profile_text[:400]

    return "I do not have enough information to answer this question from the training data."
