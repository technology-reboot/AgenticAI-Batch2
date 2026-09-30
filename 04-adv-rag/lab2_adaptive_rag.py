"""
Advanced RAG · Lab 2 — Adaptive RAG
===================================

WHAT THIS LAB TEACHES
---------------------
Lab 1 (CRAG) corrected retrieval *after* it happened. Adaptive RAG asks a
different question first: **should we retrieve at all, and from where?**

Two mechanisms, and they are independent — you can adopt either without the
other:

  1. QUERY ROUTING (before retrieval)
     An LLM classifies the incoming question into one of three routes:
         no_retrieval  — general knowledge; retrieval would only add noise
         vectorstore   — internal, private, in-corpus
         web_search    — current or external; the corpus cannot help
     Most production RAG systems retrieve unconditionally. That is a bug, not
     a simplification: it wastes tokens and it drags irrelevant context into
     questions that were fine without it.

  2. SELF-CORRECTION (after generation)
     Two graders run on the generated answer:
         groundedness — is every claim supported by the retrieved documents?
         usefulness   — does it actually address what was asked?
     A failure on either one sends control back to an earlier step rather than
     returning the answer. Groundedness failure -> regenerate. Usefulness
     failure -> rewrite the query and retrieve again.

STATE MACHINE, NOT A CHAIN
--------------------------
This is written as an explicit `state` dict passed between node functions,
with a `path` list recording every node visited. That is deliberate: it is the
same shape as a LangGraph StateGraph, but with no graph API in the way, so you
can read the control flow directly. The mapping is one-to-one —

    def route_question(state) -> state       ->  a node
    if state["route"] == "vectorstore": ...  ->  a conditional edge
    state["path"]                            ->  what LangGraph traces for you

Once you have read this file, the LangGraph version is a mechanical translation.

LLM CALLS PER QUESTION
----------------------
    1 router
    + 1 grader per retrieved document (4)
    + 1 generation
    + 1 groundedness grader
    + 1 usefulness grader
    + 1 query rewrite per correction loop
    ~ 8-14 calls. Loops are capped at MAX_LOOPS.

SETUP
-----
    pip install langchain langchain-openai langchain-chroma \
                langchain-text-splitters ddgs python-dotenv

    .env:
        OPENAI_API_KEY=sk-...

    python lab2_adaptive_rag.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Missing OPENAI_API_KEY — add it to .env and re-run.")

GRADER_MODEL = "gpt-4o-mini"
GENERATOR_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 4
MAX_LOOPS = 2  # hard ceiling on self-correction. Without this, a stubborn
# grader and a stubborn generator will argue until your budget is gone.

grader_llm = ChatOpenAI(model=GRADER_MODEL, temperature=0)
generator_llm = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)


# --------------------------------------------------------------------------
# Corpus — same Meridian policy handbook as Lab 1, so you can compare
# behaviour across the two labs on identical data.
# --------------------------------------------------------------------------

CORPUS: list[dict[str, str]] = [
    {
        "id": "HR-014",
        "title": "Parental Leave Policy",
        "text": """
        Meridian Financial Services grants 26 weeks of paid maternity leave to
        birthing parents with at least 80 days of service in the preceding
        twelve months. Non-birthing parents receive 15 working days of paid
        parental leave, to be taken within six months of the birth or adoption.
        Adoptive parents of a child below three years of age receive 12 weeks
        of paid leave. An extension of up to 8 additional weeks may be granted
        on unpaid basis; extensions are approved by the Head of People
        Operations and require the reporting manager's written endorsement.
        """,
    },
    {
        "id": "HR-021",
        "title": "Remote and Hybrid Work",
        "text": """
        Meridian operates a hybrid model. Employees in Bengaluru, Mumbai and
        Gurugram are expected in office three days per week. Employees whose
        registered address is more than 60 kilometres from the nearest Meridian
        office may apply for fully remote status, renewed annually. Remote
        employees receive a one-time home-office setup allowance of Rs 35,000
        and a monthly connectivity allowance of Rs 1,500. Hybrid employees
        receive the connectivity allowance only.
        """,
    },
    {
        "id": "HR-030",
        "title": "Leave Types and Accrual",
        "text": """
        Confirmed employees accrue 1.75 days of earned leave per completed
        month, to a maximum carry-forward of 45 days. Casual leave is granted
        at 7 days per calendar year and does not carry forward. Sick leave is
        granted at 12 days per calendar year; absences beyond three consecutive
        days require a registered medical practitioner's certificate. Leave
        encashment is permitted once per financial year, capped at 15 days.
        """,
    },
    {
        "id": "FIN-007",
        "title": "Employee Trading and Conflicts of Interest",
        "text": """
        All employees must pre-clear personal securities transactions through
        the Compliance Portal. Pre-clearance is valid for two trading days.
        Employees in Investment Research, Treasury and Corporate Finance are
        subject to a 30-day minimum holding period. Trading in any security on
        the Restricted List is prohibited without exception. Breaches are
        escalated to the Chief Compliance Officer and may result in
        disgorgement of profits.
        """,
    },
    {
        "id": "IT-011",
        "title": "Acceptable Use of Technology",
        "text": """
        Meridian-issued devices are for business use. Installation of
        unapproved software is prohibited. Company data must not be uploaded to
        personal cloud storage or to third-party AI services that have not been
        assessed by the Data Office. The approved internal assistant is
        available at assistant.meridian.internal. Data classified Confidential
        or above may not be entered into any external tool.
        """,
    },
    {
        "id": "HR-045",
        "title": "Expense and Travel Reimbursement",
        "text": """
        Domestic travel is booked through the Meridian travel desk. Economy
        class is standard for flights under three hours. Employees at Vice
        President grade and above may book business class for flights over five
        hours. Per-diem for domestic travel is Rs 2,200 per day in metro cities
        and Rs 1,600 elsewhere. Claims must be submitted within 30 days of
        travel completion with original receipts.
        """,
    },
]


def build_vectorstore() -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80, separators=["\n\n", "\n", ". ", " "]
    )
    docs: list[Document] = []
    for entry in CORPUS:
        for chunk in splitter.split_text(textwrap.dedent(entry["text"]).strip()):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"doc_id": entry["id"], "title": entry["title"]},
                )
            )
    print(f"  indexed {len(docs)} chunks from {len(CORPUS)} documents")
    return Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name="meridian_adaptive",
    )


# ==========================================================================
# STRUCTURED GRADERS
#
# Every decision in this pipeline is made by an LLM returning a typed object,
# not by parsing prose. If you take one production habit from this lab, take
# this one: never regex an LLM's opinion out of a paragraph.
# ==========================================================================


class Route(BaseModel):
    destination: Literal["no_retrieval", "vectorstore", "web_search"] = Field(
        description=(
            "'no_retrieval' for general knowledge answerable without any "
            "documents. 'vectorstore' for questions about Meridian's own "
            "internal policies. 'web_search' for current events, regulations, "
            "or anything external to Meridian."
        )
    )
    reason: str = Field(description="One short sentence.")


class DocRelevance(BaseModel):
    relevant: bool = Field(description="True only if the document helps answer the question.")
    reason: str = Field(description="One short sentence.")


class Groundedness(BaseModel):
    grounded: bool = Field(
        description="True only if every factual claim in the answer is supported by the documents."
    )
    unsupported_claim: str = Field(
        description="The first unsupported claim, verbatim, or empty string if fully grounded."
    )


class Usefulness(BaseModel):
    useful: bool = Field(description="True if the answer actually addresses the question asked.")
    reason: str = Field(description="One short sentence.")


router = grader_llm.with_structured_output(Route)
doc_grader = grader_llm.with_structured_output(DocRelevance)
grounded_grader = grader_llm.with_structured_output(Groundedness)
useful_grader = grader_llm.with_structured_output(Usefulness)


# ==========================================================================
# NODES
#
# Each function takes the state dict, mutates it, and returns it. This is
# exactly the LangGraph node signature. `state["path"]` is the trace.
# ==========================================================================

State = dict[str, Any]


def log(state: State, node: str, detail: str = "") -> None:
    state["path"].append(node)
    print(f"  -> {node:<22} {detail}")


ROUTER_PROMPT = """You are routing a user's question to the right knowledge source \
for Meridian Financial Services, an Indian NBFC.

Meridian's internal vectorstore contains its own HR, finance, compliance and IT
policies — leave, remote work, expenses, employee trading, acceptable use.

Route to 'no_retrieval' only when the question is general knowledge that needs
no documents at all.

Question: {question}
"""


def route_question(state: State) -> State:
    r = router.invoke(ROUTER_PROMPT.format(question=state["question"]))
    state["route"] = r.destination
    log(state, "route_question", f"{r.destination}  ({r.reason})")
    return state


def answer_directly(state: State) -> State:
    """The no_retrieval branch. No documents, no grading — just answer."""
    state["answer"] = generator_llm.invoke(state["question"]).content
    state["documents"] = []
    log(state, "answer_directly", "no retrieval performed")
    return state


def retrieve(state: State) -> State:
    docs = state["store"].similarity_search(state["question"], k=TOP_K)
    state["documents"] = docs
    log(state, "retrieve", ", ".join(d.metadata["doc_id"] for d in docs))
    return state


def grade_documents(state: State) -> State:
    """Filter retrieval down to documents that actually help. One call each."""
    kept: list[Document] = []
    for doc in state["documents"]:
        g = doc_grader.invoke(
            "Question: {q}\n\nDocument:\n{d}\n\nDoes this document help answer "
            "the question? Be strict — shared vocabulary is not relevance.".format(
                q=state["question"], d=doc.page_content
            )
        )
        mark = "keep" if g.relevant else "drop"
        print(f"       {mark}  {doc.metadata['doc_id']:<8} {g.reason}")
        if g.relevant:
            kept.append(doc)
    state["documents"] = kept
    log(state, "grade_documents", f"{len(kept)} of {TOP_K} kept")
    return state


def web_search_node(state: State) -> State:
    query = state.get("question", "")
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=4))
        docs = [
            Document(
                page_content=f"{h.get('title', '')}\n{h.get('body', '')}",
                metadata={"doc_id": "web", "title": h.get("href", "web")},
            )
            for h in hits
        ]
    except Exception as exc:  # noqa: BLE001 — search must never break the lab
        docs = [
            Document(
                page_content=f"External search unavailable ({type(exc).__name__}).",
                metadata={"doc_id": "web", "title": "unavailable"},
            )
        ]
    state["documents"] = docs
    log(state, "web_search", f"{len(docs)} result(s)")
    return state


GENERATE_PROMPT = """Answer the question using only the context below.

If the context does not contain the answer, say so plainly. Do not fill gaps
from your own knowledge — this is an internal policy assistant and a plausible
guess is worse than an admission.

Context:
{context}

Question: {question}

Answer:"""


def generate(state: State) -> State:
    context = "\n\n".join(
        f"[{d.metadata['doc_id']}] {d.page_content}" for d in state["documents"]
    ) or "No documents were retrieved."
    state["answer"] = generator_llm.invoke(
        GENERATE_PROMPT.format(context=context, question=state["question"])
    ).content
    log(state, "generate", f"from {len(state['documents'])} document(s)")
    return state


def grade_generation(state: State) -> str:
    """Run both post-generation graders and return the next edge to follow.

    Returns one of: 'accept' | 'regenerate' | 'rewrite'
    """
    context = "\n\n".join(d.page_content for d in state["documents"])

    g = grounded_grader.invoke(
        "Documents:\n{c}\n\nAnswer:\n{a}\n\nIs every factual claim in the answer "
        "supported by the documents?".format(c=context, a=state["answer"])
    )
    print(f"       grounded={g.grounded}  {g.unsupported_claim or ''}")
    if not g.grounded:
        log(state, "grade_generation", "NOT GROUNDED -> regenerate")
        return "regenerate"

    u = useful_grader.invoke(
        "Question: {q}\n\nAnswer:\n{a}\n\nDoes the answer address the question "
        "that was asked?".format(q=state["question"], a=state["answer"])
    )
    print(f"       useful={u.useful}  {u.reason}")
    if not u.useful:
        log(state, "grade_generation", "NOT USEFUL -> rewrite query")
        return "rewrite"

    log(state, "grade_generation", "accepted")
    return "accept"


REWRITE_PROMPT = """The following question did not retrieve useful documents from \
an internal policy corpus. Rewrite it to be more retrievable: use the vocabulary
a policy document would use, and make the subject explicit. Return only the
rewritten question.

Original: {question}
"""


def transform_query(state: State) -> State:
    new_q = generator_llm.invoke(
        REWRITE_PROMPT.format(question=state["question"])
    ).content.strip()
    log(state, "transform_query", f"{state['question'][:40]!r} -> {new_q!r}")
    state["question"] = new_q
    return state


# ==========================================================================
# THE GRAPH
#
# Read this function as the edge list. Every `if` is a conditional edge.
# ==========================================================================


def adaptive_rag(store: Chroma, question: str) -> State:
    state: State = {
        "question": question,
        "original_question": question,
        "store": store,
        "documents": [],
        "answer": "",
        "path": [],
        "loops": 0,
    }

    # entry edge — routing decides everything downstream
    route_question(state)

    if state["route"] == "no_retrieval":
        answer_directly(state)
        return state  # no grading loop: there are no documents to ground against

    if state["route"] == "web_search":
        web_search_node(state)
    else:
        retrieve(state)
        grade_documents(state)

        # conditional edge: retrieval came back empty after grading
        if not state["documents"]:
            log(state, "decide", "no relevant documents -> fall back to web")
            web_search_node(state)

    # generation + self-correction loop
    while True:
        generate(state)
        verdict = grade_generation(state)

        if verdict == "accept":
            return state

        state["loops"] += 1
        if state["loops"] > MAX_LOOPS:
            log(state, "decide", f"loop ceiling ({MAX_LOOPS}) hit -> returning as-is")
            state["answer"] += (
                "\n\n[Adaptive RAG note: returned after hitting the correction "
                "loop ceiling. Treat this answer as unverified.]"
            )
            return state

        if verdict == "regenerate":
            continue  # same documents, another attempt at the answer

        # verdict == 'rewrite' — go back further, to retrieval
        transform_query(state)
        retrieve(state)
        grade_documents(state)
        if not state["documents"]:
            web_search_node(state)


# ==========================================================================
# DEMO
# ==========================================================================

QUESTIONS = [
    (
        "Q1",
        "What is the capital city of Karnataka?",
        "Should route to no_retrieval. Watch that the pipeline does NOT touch "
        "the vectorstore — a naive RAG system would retrieve four policy chunks "
        "here and pay for the privilege.",
    ),
    (
        "Q2",
        "How much is the home-office setup allowance for a fully remote employee?",
        "Should route to vectorstore, grade most documents out, keep HR-021, "
        "and pass both post-generation graders on the first attempt.",
    ),
    (
        "Q3",
        "What are the RBI's current guidelines on digital lending?",
        "Should route to web_search — the corpus has nothing on regulators. "
        "Note that groundedness is now graded against web snippets, which is a "
        "materially weaker guarantee.",
    ),
    (
        "Q4",
        "What does Meridian pay people who work from far away?",
        "Deliberately vague phrasing. Retrieval may come back thin and the "
        "usefulness grader may reject the first answer, triggering "
        "transform_query. This is the self-correction loop doing its job — "
        "watch the rewritten question in the trace.",
    ),
]


def banner(text: str, char: str = "=") -> None:
    print(f"\n{char * 74}\n {text}\n{char * 74}")


def main() -> None:
    banner("BUILDING THE INDEX")
    store = build_vectorstore()

    for tag, question, note in QUESTIONS:
        banner(f"{tag} — {question}")
        print(f" Why this question: {note}\n")
        print(" TRACE")
        state = adaptive_rag(store, question)

        print("\n ANSWER")
        print(textwrap.indent(textwrap.fill(state["answer"], 70), "  "))

        print(f"\n PATH: {' -> '.join(state['path'])}")
        if state["question"] != state["original_question"]:
            print(f" QUERY WAS REWRITTEN: {state['question']!r}")
        print(f" CORRECTION LOOPS: {state['loops']}")

    banner("NOW ANSWER THESE, IN YOUR OWN WORDS", "-")
    print(
        """
 1. On Q1, how many LLM calls did the adaptive pipeline make? How many would
    an unconditional RAG pipeline have made? Where did the saving come from —
    the LLM calls, or the embedding call?

 2. The router is a single LLM call deciding the fate of the whole request.
    What happens when it routes a Meridian policy question to web_search?
    Try it: change the ROUTER_PROMPT to remove the description of what the
    vectorstore contains, and re-run Q2.

 3. Set MAX_LOOPS = 0 and re-run Q4. What do you get back, and is it worse
    than the looped answer or just cheaper?

 4. The groundedness grader checks the answer against the documents. It cannot
    check whether the DOCUMENTS are correct. Name one class of failure this
    pipeline cannot catch at all.
"""
    )


if __name__ == "__main__":
    main()
