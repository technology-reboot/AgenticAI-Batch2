"""Lab 2: Simple graph-style reasoning over company profile facts.

This example shows how a graph-like structure can help connect facts such as
company, founder year, headquarters, and CEO. The goal is not to be production
ready, but to make the concept easy to understand in a training lab.
"""

from __future__ import annotations

from common import build_simple_graph, load_company_profiles


def find_related_facts(query: str, graph: dict[str, dict[str, str]]) -> list[str]:
    """Return a small set of graph facts that are relevant to the query."""
    lowered = query.lower()
    relevant = []

    for company, facts in graph.items():
        if company.lower() in lowered:
            if "founded" in facts:
                relevant.append(f"{company} founded in {facts['founded']}")
            if "headquarters" in facts:
                relevant.append(f"{company} HQ is {facts['headquarters']}")
            if "ceo" in facts:
                relevant.append(f"{company} CEO is {facts['ceo']}")

    return relevant


def run_lab2(query: str) -> str:
    """Run the graph-based reasoning example."""
    profiles = load_company_profiles()
    graph = build_simple_graph(profiles)
    facts = find_related_facts(query, graph)

    if not facts:
        return "No matching company facts were found in the sample graph."

    return "Related facts:\n- " + "\n- ".join(facts)


if __name__ == "__main__":
    sample_queries = [
        "Tell me about TCS",
        "What is the CEO of Infosys?",
        "Where is Wipro headquartered?",
    ]

    for query in sample_queries:
        print("=" * 70)
        print(f"Query: {query}")
        print(run_lab2(query))
        print()
