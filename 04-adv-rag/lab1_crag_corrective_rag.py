"""
Advanced RAG · Lab 1 — Corrective RAG (CRAG)
===========================================

PART 1 builds an ordinary RAG pipeline: chunk -> embed -> retrieve -> generate.
        One LLM call. It fails silently and confidently when the answer is not
        in the corpus.

PART 2 layers CRAG on top. A retrieval evaluator sits between retrieval and
        generation and picks one of three actions:

            CORRECT    all retrieved docs relevant -> refine, then generate
            AMBIGUOUS  some relevant, some not      -> refine + web, then generate
            INCORRECT  nothing relevant             -> discard retrieval, web, generate

        Mirrors the CRAG paper (Yan et al., 2024), using an LLM with structured
        output in place of the paper's fine-tuned evaluator.

SETUP
    pip install langchain langchain-openai langchain-community faiss-cpu \
                langchain-text-splitters ddgs python-dotenv
    .env:  OPENAI_API_KEY=sk-...
    python lab1_crag_corrective_rag.py

Web search degrades gracefully offline — it returns a labelled "no web results"
string instead of crashing, so the CRAG path still runs in a classroom.
"""

# Enable postponed evaluation of annotations so `list[dict[str, str]]` style hints
# work on older Python versions without importing from `typing`.
from __future__ import annotations

import json  # parse the corpus file, which is stored as JSON on disk
import os  # read environment variables (the OpenAI API key)
import sys  # exit early with a message when prerequisites are missing
import textwrap  # dedent corpus text and pretty-wrap answers for the console
from pathlib import Path  # build a filesystem path to the corpus relative to this file
from typing import Literal  # constrain the grader's score to a fixed set of strings

from dotenv import load_dotenv  # load key=value pairs from a local .env file into the environment
from pydantic import BaseModel, Field  # declare the structured schema the grader LLM must return

from langchain_core.documents import Document  # the (text + metadata) unit passed through the pipeline
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # chat model + embedding model clients
from langchain_community.vectorstores import FAISS  # in-memory similarity-search index
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits long text into overlapping chunks

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

load_dotenv()  # pull OPENAI_API_KEY (and anything else) from .env into os.environ

# Abort with a clear message rather than failing deep inside an API call later.
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Missing OPENAI_API_KEY — add it to .env and re-run.")

# Two models on purpose: grading is high-volume / low-difficulty, generation is
# low-volume / high-difficulty. Routing them separately is the main cost lever.
GRADER_MODEL = "gpt-4o-mini"  # cheap, fast model for the many evaluator/refine/rewrite calls
GENERATOR_MODEL = "gpt-4o"  # stronger, pricier model reserved for the final answer

EMBEDDING_MODEL = "text-embedding-3-small"  # model used to vectorise chunks and queries
TOP_K = 4  # chunks similarity_search returns per question

grader_llm = ChatOpenAI(model=GRADER_MODEL, temperature=0)  # deterministic grader client
generator_llm = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)  # deterministic generator client


# --------------------------------------------------------------------------
# The corpus — an internal policy handbook for a fictional Indian NBFC.
# Loaded from data/lab1_corpus.json. Deliberately absent from that file:
# any regulator guidance and any statutory minimums. That absence is what
# Q2 and Q3 exploit.
# --------------------------------------------------------------------------

# Resolve <this file's directory>/data/lab1_corpus.json regardless of the cwd.
CORPUS_PATH = Path(__file__).resolve().parent / "data" / "lab1_corpus.json"


def load_corpus() -> list[dict[str, str]]:
    """Read the policy corpus from disk."""
    # Fail fast with the expected path if the data file is missing.
    if not CORPUS_PATH.exists():
        sys.exit(f"Missing corpus file — expected it at {CORPUS_PATH}")
    # Parse the JSON array of {id, title, text} records.
    entries = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for entry in entries:  # collapse runs of spaces/newlines to one space
        entry["text"] = " ".join(entry["text"].split())  # normalise whitespace in place
    return entries  # hand back the cleaned list of records


CORPUS: list[dict[str, str]] = load_corpus()  # load once at import time; reused everywhere


# ==========================================================================
# PART 1 — BASIC RAG
# The whole of a working RAG pipeline. Part 2 does not replace any of it.
# ==========================================================================


def build_vectorstore() -> FAISS:
    """Chunk the corpus, embed it, and return an in-memory FAISS store."""
    # Configure the splitter: ~600-char chunks, 80-char overlap, split on the
    # most natural boundary available (paragraph > line > sentence > word).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )

    docs: list[Document] = []  # accumulator for every chunk across every corpus entry
    for entry in CORPUS:  # walk each policy document
        clean = textwrap.dedent(entry["text"]).strip()  # remove common indentation and edge whitespace
        for chunk in splitter.split_text(clean):  # break the document into overlapping chunks
            docs.append(
                Document(
                    page_content=chunk,  # the chunk text that gets embedded
                    metadata={"doc_id": entry["id"], "title": entry["title"]},  # provenance for display
                )
            )

    print(f"  indexed {len(docs)} chunks from {len(CORPUS)} documents")  # progress line for the demo
    # Embed every chunk and build the searchable FAISS index in one call.
    return FAISS.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )


# Prompt template for Part 1: strict closed-book-over-context instruction.
BASIC_RAG_PROMPT = """You are Meridian's internal policy assistant.
Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer:"""


def basic_rag(store: FAISS, question: str) -> tuple[str, list[Document]]:
    """Retrieve top-k chunks and generate an answer. Exactly one LLM call.

    It never asks whether the retrieved chunks are actually about the question —
    that is the failure mode CRAG addresses.
    """
    docs = store.similarity_search(question, k=TOP_K)  # nearest-neighbour lookup in embedding space
    # Concatenate the retrieved chunks into one context block, each prefixed with its source.
    context = "\n\n".join(
        f"[{d.metadata['doc_id']} — {d.metadata['title']}]\n{d.page_content}" for d in docs
    )
    prompt = BASIC_RAG_PROMPT.format(context=context, question=question)  # fill the template
    answer = generator_llm.invoke(prompt).content  # the single LLM call; `.content` is the text
    return answer, docs  # return the answer plus the chunks used (for display)


# ==========================================================================
# PART 2 — CORRECTIVE RAG
# Everything below sits BETWEEN retrieval and generation. Retriever and
# generator from Part 1 are unchanged; CRAG is a control layer.
# ==========================================================================


class RelevanceGrade(BaseModel):
    """Structured verdict from the retrieval evaluator."""

    # The three-way relevance label; the enum is enforced by structured output.
    score: Literal["relevant", "ambiguous", "irrelevant"] = Field(
        description=(
            "'relevant' if the document contains information that directly "
            "helps answer the question. 'ambiguous' if it is on a related "
            "topic but does not contain the answer. 'irrelevant' if it is "
            "about something else entirely."
        )
    )
    # A short rationale, mostly so a human watching the demo can sanity-check the label.
    reason: str = Field(description="One short sentence justifying the score.")


grader = grader_llm.with_structured_output(RelevanceGrade)  # wrap the LLM so it returns a RelevanceGrade

# Prompt for the evaluator: deliberately strict to punish vocabulary overlap.
GRADER_PROMPT = """You are grading whether a retrieved document is useful for \
answering a user's question.

Be strict. A document that merely shares vocabulary with the question is not
relevant. A document is only 'relevant' if a careful reader could extract part
of the answer from it.

Question: {question}

Document:
{document}
"""


def grade_document(question: str, doc: Document) -> RelevanceGrade:
    """One LLM call — the retrieval evaluator from the CRAG paper."""
    # Fill the grader prompt with this question/chunk pair and return the parsed verdict.
    return grader.invoke(GRADER_PROMPT.format(question=question, document=doc.page_content))


# Prompt for knowledge refinement: keep only answer-bearing sentences, verbatim.
REFINE_PROMPT = """Extract only the sentences from the document below that help \
answer the question. Copy them verbatim. Drop everything else. If nothing in the
document helps, reply with exactly: NONE

Question: {question}

Document:
{document}
"""


def refine_document(question: str, doc: Document) -> str:
    """Knowledge refinement — strip a chunk to its answer-bearing sentences.

    One LLM call per relevant document. Returns "" when nothing helps.
    """
    # Ask the cheap model to pull out the useful sentences from this chunk.
    result = grader_llm.invoke(
        REFINE_PROMPT.format(question=question, document=doc.page_content)
    ).content
    # Treat a "NONE" reply as "nothing useful" -> empty string; otherwise trim and keep.
    return "" if result.strip().upper().startswith("NONE") else result.strip()


# Prompt to turn an internal-sounding question into a public web search query.
REWRITE_PROMPT = """Rewrite the user's question as a short, keyword-style web \
search query. Strip any company-internal names that a public search engine would
not know. Return only the query, nothing else.

Question: {question}
"""


def rewrite_for_web(question: str) -> str:
    """One LLM call. Internal phrasing rarely makes a good public search query."""
    # Generate the query, then strip whitespace and any wrapping quote marks.
    return grader_llm.invoke(REWRITE_PROMPT.format(question=question)).content.strip().strip('"')


def web_search(query: str, max_results: int = 4) -> str:
    """Best-effort external knowledge. Never raises — a classroom may be offline."""
    try:
        from ddgs import DDGS  # lazy import so the lab loads without the package

        with DDGS() as ddgs:  # open a search session (context manager closes the client)
            hits = list(ddgs.text(query, max_results=max_results))  # run the text search, materialise results
        if not hits:  # search worked but found nothing
            return "NO WEB RESULTS — the external search returned nothing."
        # Format each hit as a labelled title / snippet / url block.
        return "\n\n".join(
            f"[web] {h.get('title', '')}\n{h.get('body', '')}\n{h.get('href', '')}" for h in hits
        )
    except Exception as exc:  # noqa: BLE001 — search must never break the lab
        # Any failure (offline, package missing, rate limit) degrades to a labelled string.
        return f"NO WEB RESULTS — external search unavailable ({type(exc).__name__}: {exc})."


# Final-answer prompt for CRAG: source-aware, and explicitly forbids guessing.
CRAG_PROMPT = """You are Meridian's internal policy assistant.

Answer the question using the knowledge below. The knowledge is labelled by
source. Internal policy is authoritative for Meridian's own rules; web results
are supporting context only and may be out of date.

If the knowledge does not contain the answer, say so plainly. Do not guess, and
do not fill gaps from your own training data.

Knowledge:
{knowledge}

Question: {question}

Answer:"""


def crag(store: FAISS, question: str) -> str:
    """Corrective RAG: retrieve, evaluate, act, then generate."""

    # step 1: retrieve, exactly as Part 1 does
    docs = store.similarity_search(question, k=TOP_K)  # same nearest-neighbour lookup as basic_rag
    print(f"  retrieved {len(docs)} chunks")  # demo progress line

    # step 2: evaluate every retrieved document (one grader call per chunk)
    grades = []  # list of (Document, RelevanceGrade) pairs
    for doc in docs:  # grade each retrieved chunk independently
        g = grade_document(question, doc)  # LLM verdict for this chunk
        grades.append((doc, g))  # keep the chunk alongside its grade
        print(f"    {doc.metadata['doc_id']:<8} {g.score:<11} {g.reason}")  # show the verdict

    relevant = [d for d, g in grades if g.score == "relevant"]  # chunks the grader trusts
    irrelevant = [d for d, g in grades if g.score == "irrelevant"]  # chunks the grader rejects
    # "ambiguous"-scored chunks fall into neither list — they're dropped

    # step 3: pick an action — this three-way branch IS Corrective RAG
    if relevant and not irrelevant:  # everything retrieved is usable
        action = "CORRECT"
    elif not relevant:  # nothing retrieved is usable
        action = "INCORRECT"
    else:  # a mix of usable and unusable
        action = "AMBIGUOUS"

    print(f"\n  ACTION: {action}")  # announce the branch taken

    knowledge_parts: list[str] = []  # collect the knowledge blocks fed to the generator

    # step 4a: internal knowledge refinement
    if action in ("CORRECT", "AMBIGUOUS"):  # these branches keep internal docs
        print(f"  refining {len(relevant)} relevant document(s)...")
        for doc in relevant:  # refine each relevant chunk down to key sentences
            strips = refine_document(question, doc)  # verbatim answer-bearing sentences (or "")
            if strips:  # only add a block when refinement kept something
                knowledge_parts.append(
                    f"[internal policy {doc.metadata['doc_id']} — "
                    f"{doc.metadata['title']}]\n{strips}"
                )

    # step 4b: external knowledge acquisition
    if action in ("AMBIGUOUS", "INCORRECT"):  # these branches need the web
        query = rewrite_for_web(question)  # convert the question to a search query
        print(f"  web query: {query!r}")
        knowledge_parts.append(web_search(query))  # append the (possibly "no results") web block

    # Join all knowledge blocks, or fall back to an explicit "nothing found" marker.
    knowledge = "\n\n".join(knowledge_parts) if knowledge_parts else "No usable knowledge found."

    # step 5: generate on the strong model
    return generator_llm.invoke(
        CRAG_PROMPT.format(knowledge=knowledge, question=question)  # fill the final prompt
    ).content  # return just the answer text


# ==========================================================================
# DEMO — two questions: one exercises AMBIGUOUS, one exercises INCORRECT.
# (CORRECT is the case where basic RAG already ties CRAG; discussed verbally.)
# ==========================================================================

# Each tuple: (tag, question text, teaching note explaining what to watch for).
QUESTIONS = [
    (
        "Q2",
        "How does Meridian's maternity leave compare with the statutory "
        "minimum required under Indian law?",
        "Half in the corpus, half not. Expect AMBIGUOUS: the internal policy "
        "is retrievable, the statutory minimum is not. Watch what basic RAG "
        "does with the half it cannot find.",
    ),
    (
        "Q3",
        "What do the RBI's digital lending guidelines require regarding "
        "loan disbursal to borrower bank accounts?",
        "Nothing in the corpus touches this. Expect INCORRECT — retrieval is "
        "discarded entirely. This is where basic RAG is at its most dangerous, "
        "because it will still return four confident-looking policy chunks.",
    ),
]


def banner(text: str, char: str = "=") -> None:
    """Print `text` framed by a full-width rule for readable console sections."""
    print(f"\n{char * 74}\n {text}\n{char * 74}")


def main() -> None:
    """Build the index once, then run both pipelines against every demo question."""
    banner("BUILDING THE INDEX")  # section header
    store = build_vectorstore()  # chunk + embed the corpus into a FAISS index

    for tag, question, note in QUESTIONS:  # iterate over the demo questions
        banner(f"{tag} — {question}")  # per-question header
        print(f" Why this question: {note}\n")  # show the teaching note

        print("-" * 74)
        print(" PART 1 — BASIC RAG")
        print("-" * 74)
        basic_answer, basic_docs = basic_rag(store, question)  # run the one-shot pipeline
        print("  retrieved: " + ", ".join(d.metadata["doc_id"] for d in basic_docs))  # which chunks
        print("\n" + textwrap.indent(textwrap.fill(basic_answer, 70), "  "))  # wrapped, indented answer

        print("\n" + "-" * 74)
        print(" PART 2 — CORRECTIVE RAG")
        print("-" * 74)
        crag_answer = crag(store, question)  # run the corrective pipeline (prints its own trace)
        print("\n" + textwrap.indent(textwrap.fill(crag_answer, 70), "  "))  # wrapped, indented answer

    banner("NOW ANSWER THESE, IN YOUR OWN WORDS", "-")  # closing exercise section
    print(
        """
 1. On Q3, what exactly did basic RAG do with four irrelevant chunks? Quote
    the sentence that would have got someone in trouble.

 2. The evaluator is an LLM. What happens to CRAG when the evaluator is
    wrong — and which of its three errors is most costly: calling a relevant
    doc irrelevant, or calling an irrelevant doc relevant?

 3. Change TOP_K from 4 to 8 and re-run Q2. Does the action change? Should it?
"""
    )


# Standard entry-point guard: only run the demo when executed directly.
if __name__ == "__main__":
    main()
