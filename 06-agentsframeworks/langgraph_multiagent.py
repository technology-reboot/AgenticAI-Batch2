import operator
import os
import time
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

load_dotenv()

# LangChain and LangGraph send traces to LangSmith automatically when
# LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set (see .env).
# LANGSMITH_PROJECT groups this lab's runs in the LangSmith UI.
os.environ.setdefault("LANGSMITH_PROJECT", "agentic-lab-langgraph")
if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and not os.getenv("LANGSMITH_API_KEY"):
    print("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is missing; traces will not upload.")

MODEL = "gpt-4o-mini"
TOPIC = "India's DPDP Act obligations for AI systems handling customer data"
MAX_REVISIONS = 2
APPROVED = "APPROVED"


class LabState(TypedDict):
    topic: str
    search_query: str
    findings: str
    draft: str
    review: str          # reviewer feedback; empty means the draft passed
    revisions: int
    status: str
    log: Annotated[list[str], operator.add]   # each node appends, nothing is overwritten


# Each agent has its own model instance and system prompt. The tool is plain
# Python, wrapped in @traceable so it shows up as a tool run in LangSmith.
researcher_llm = ChatOpenAI(model=MODEL, temperature=0)
writer_llm = ChatOpenAI(model=MODEL, temperature=0.3)
reviewer_llm = ChatOpenAI(model=MODEL, temperature=0)


@traceable(run_type="tool", name="web_search")
def web_search(query: str) -> str:
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=5))
    if not hits:
        return "No results found."
    return "\n".join(
        f"- {h.get('title', '')}: {h.get('body', '')[:200]} ({h.get('href', '')})"
        for h in hits
    )


# Agent 1: Researcher. Writes a search query, searches, then summarises the results with sources.
def researcher(state: LabState) -> dict:
    query = researcher_llm.invoke([
        SystemMessage(content="You write one concise web search query. Reply with the query only."),
        HumanMessage(content=state["topic"]),
    ]).content.strip()

    results = web_search(query)

    notes = researcher_llm.invoke([
        SystemMessage(content=(
            "You are a Research Analyst. Use only the search results you are given. "
            "Return 5 bullet findings, each with its source URL. Never state what the results do not say."
        )),
        HumanMessage(content=f"Topic: {state['topic']}\n\nSearch results:\n{results}"),
    ]).content
    return {
        "search_query": query,
        "findings": notes,
        "log": [f"researcher: searched '{query}'"],
    }


# Agent 2: Writer. Drafts a short brief. On a revision pass it also receives the reviewer's feedback.
def writer(state: LabState) -> dict:
    feedback = (
        f"\n\nFix these issues from the reviewer:\n{state['review']}" if state["review"] else ""
    )
    draft = writer_llm.invoke([
        SystemMessage(content=(
            "You are a Briefing Writer. Write a 150-word brief using only the findings. "
            "Do not add facts that are not in the findings."
        )),
        HumanMessage(content=f"Topic: {state['topic']}\n\nFindings:\n{state['findings']}{feedback}"),
    ]).content
    version = state["revisions"] + 1
    return {"draft": draft, "log": [f"writer: draft v{version}"]}


# Agent 3: Reviewer. Checks every claim in the draft against the findings.
def reviewer(state: LabState) -> dict:
    verdict = reviewer_llm.invoke([
        SystemMessage(content=(
            "You are a Quality Critic. Treat a claim as unsupported unless it appears in the findings. "
            f"If every claim is supported, reply with exactly {APPROVED}. "
            "Otherwise reply with a numbered list of required revisions."
        )),
        HumanMessage(content=f"FINDINGS:\n{state['findings']}\n\nDRAFT:\n{state['draft']}"),
    ]).content.strip()

    if verdict.upper().startswith(APPROVED):
        return {"review": "", "log": ["reviewer: approved"]}
    return {
        "review": verdict,
        "revisions": state["revisions"] + 1,
        "log": ["reviewer: revisions requested"],
    }


def route_after_review(state: LabState) -> Literal["writer", "publish", "escalate"]:
    if not state["review"]:
        return "publish"
    if state["revisions"] >= MAX_REVISIONS:
        return "escalate"   # recovery branch: stop looping and hand off to a human
    return "writer"


def publish(state: LabState) -> dict:
    return {"status": "approved", "log": ["publish: brief approved"]}


def escalate(state: LabState) -> dict:
    return {"status": "needs_human_review",
            "log": [f"escalate: still unsupported after {state['revisions']} revision(s)"]}


graph = StateGraph(LabState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)
graph.add_node("publish", publish)
graph.add_node("escalate", escalate)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_conditional_edges(
    "reviewer",
    route_after_review,
    {"writer": "writer", "publish": "publish", "escalate": "escalate"},
)
graph.add_edge("publish", END)
graph.add_edge("escalate", END)

app = graph.compile()


def main():
    initial = {
        "topic": TOPIC, "search_query": "", "findings": "", "draft": "",
        "review": "", "revisions": 0, "status": "", "log": [],
    }
    t0 = time.perf_counter()
    result = app.invoke(
        initial,
        config={
            "run_name": "langgraph-3-agent-brief",
            "tags": ["lab3", "langgraph"],
            "metadata": {"topic": TOPIC, "max_revisions": MAX_REVISIONS},
        },
    )
    wall = time.perf_counter() - t0

    print("\nRun log:")
    for line in result["log"]:
        print(f"  - {line}")
    print(f"\nStatus: {result['status']} (revisions requested: {result['revisions']})")
    print(f"\n{result['draft']}")
    print(f"\nWall time: {wall:.1f}s")
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        print(f"Trace sent to LangSmith project '{os.getenv('LANGSMITH_PROJECT')}'.")


if __name__ == "__main__":
    main()
