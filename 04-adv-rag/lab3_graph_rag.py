"""
Advanced RAG · Lab 3 — Graph RAG
================================

WHAT THIS LAB TEACHES
---------------------
Vector RAG retrieves chunks that are *similar to the question*. That works when
the answer sits inside one chunk. It fails on two shapes of question:

  MULTI-HOP    "Which compliance system does the team that led the Kalyani
                Motors deal use?"
                Three facts, three different documents, no single chunk is
                similar to the question. Vector search returns the chunk that
                mentions Kalyani Motors and stops there.

  GLOBAL       "What are the recurring risk themes across Meridian?"
                No chunk contains the answer. The answer is a property of the
                whole corpus.

Graph RAG addresses both by building an explicit structure first:

  1. EXTRACTION   an LLM reads each document and emits (subject, relation,
                  object) triples. This is the expensive step, done once at
                  index time, and it is a real LLM call per document.

  2. LOCAL SEARCH  extract entities from the question, find them in the graph,
                   walk N hops, serialise that subgraph as text, then generate.
                   This is what answers multi-hop questions — the walk assembles
                   facts that were never co-located in any chunk.

  3. GLOBAL SEARCH detect communities in the graph, have an LLM summarise each
                   one, then map-reduce over those summaries. This is the
                   Microsoft GraphRAG "global search" idea in miniature.

The lab runs vector RAG and graph RAG side by side on the same questions so the
difference is observed, not asserted.

HONEST COST NOTE
----------------
Graph extraction costs one LLM call per document at index time, every time you
re-index. On this 8-document corpus that is trivial. On 200,000 documents it is
a five-figure decision that needs a business case. Graph RAG is not a free
upgrade — it trades index-time cost for query-time capability.

LLM CALLS
---------
    Index time  : 1 extraction call per document (8)
                  + 1 summarisation call per community (typically 3-5)
    Local query : 1 entity-extraction + 1 generation
    Global query: 1 map call per community + 1 reduce call

SETUP
-----
    pip install langchain langchain-openai langchain-chroma \
                langchain-text-splitters networkx python-dotenv

    .env:
        OPENAI_API_KEY=sk-...

    python lab3_graph_rag.py
"""

from __future__ import annotations

import os
import sys
import textwrap

import networkx as nx
from dotenv import load_dotenv
from networkx.algorithms.community import greedy_modularity_communities
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

EXTRACTION_MODEL = "gpt-4o"  # extraction quality determines everything downstream;
# a weak model here produces a graph that is quietly wrong
GENERATOR_MODEL = "gpt-4o"
SUMMARY_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 4
HOPS = 2  # how far to walk from a matched entity

extract_llm = ChatOpenAI(model=EXTRACTION_MODEL, temperature=0)
generator_llm = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)
summary_llm = ChatOpenAI(model=SUMMARY_MODEL, temperature=0)


# --------------------------------------------------------------------------
# Corpus — Meridian Financial Services, told as connected narrative rather
# than as standalone policies.
#
# Read these and note something: no single document contains the chain
# "Kalyani Motors deal -> which team -> which compliance system". That chain
# exists only across three documents. Vector search cannot assemble it. This
# is the whole reason the lab exists.
# --------------------------------------------------------------------------

CORPUS: list[dict[str, str]] = [
    {
        "id": "DOC-01",
        "title": "Corporate Finance Group",
        "text": """
        The Corporate Finance Group at Meridian Financial Services is led by
        Ananya Rao, who reports to the Chief Executive, Vikram Set. The group
        contains two teams: the Structured Credit Desk and the Acquisitions
        Desk. Ananya Rao joined Meridian in 2019 from Sundaram Capital. The
        Corporate Finance Group operates from the Mumbai office.
        """,
    },
    {
        "id": "DOC-02",
        "title": "The Kalyani Motors Transaction",
        "text": """
        In March 2025 Meridian advised on the acquisition of Kalyani Motors
        Finance, a Pune-based auto-loan originator. The transaction was led by
        the Acquisitions Desk. Kalyani Motors Finance was acquired by Sundaram
        Capital, with Meridian acting as sole financial adviser. The deal
        closed at an enterprise value of Rs 1,840 crore.
        """,
    },
    {
        "id": "DOC-03",
        "title": "Compliance Tooling by Desk",
        "text": """
        The Acquisitions Desk uses the Sentinel compliance system for deal
        clearance and conflict checks. The Structured Credit Desk uses Aegis
        for the same purpose. Sentinel is maintained by the Technology Office
        and was commissioned in 2023. Aegis is a vendor product licensed from
        Torvale Systems. Both systems feed into the central Restricted List.
        """,
    },
    {
        "id": "DOC-04",
        "title": "Technology Office",
        "text": """
        The Technology Office is led by Rehan Fernandes and is based in
        Bengaluru. It maintains Sentinel, the Meridian Data Platform, and the
        internal assistant. Rehan Fernandes reports to the Chief Operating
        Officer, Priya Nambiar. The Technology Office runs a quarterly
        vendor-risk review covering all licensed systems including Aegis.
        """,
    },
    {
        "id": "DOC-05",
        "title": "Structured Credit Desk",
        "text": """
        The Structured Credit Desk is headed by Joseph Mathew and originates
        securitisation transactions for non-bank lenders. The desk was formed in
        2021 and operates from Mumbai. Joseph Mathew previously led origination
        at Torvale Systems' financial services division. The desk's largest
        exposure is to auto-loan portfolios.
        """,
    },
    {
        "id": "DOC-06",
        "title": "Acquisitions Desk",
        "text": """
        The Acquisitions Desk is headed by Meera Iyer and advises on buy-side
        and sell-side mandates for financial services targets. The desk was
        formed in 2018 and operates from Mumbai. Meera Iyer reports to Ananya
        Rao. The desk completed nine mandates in the 2025 financial year.
        """,
    },
    {
        "id": "DOC-07",
        "title": "Risk Committee Minutes, April 2025",
        "text": """
        The Risk Committee noted concentration risk in auto-loan exposures
        following the Kalyani Motors transaction. The Committee also flagged
        vendor concentration: Torvale Systems supplies both Aegis and the
        treasury reconciliation engine. Priya Nambiar was asked to present a
        vendor-diversification plan. The Committee is chaired by Vikram Set.
        """,
    },
    {
        "id": "DOC-08",
        "title": "Data Platform and Access",
        "text": """
        The Meridian Data Platform holds transaction records for all desks.
        Access is granted by desk, not by individual. The Acquisitions Desk has
        read access to deal records; the Structured Credit Desk has read access
        to portfolio records. The Data Platform is maintained by the Technology
        Office and audited annually by the Risk Committee.
        """,
    },
]


# ==========================================================================
# BASELINE — plain vector RAG, for comparison
# ==========================================================================


def build_vectorstore() -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
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
    print(f"  indexed {len(docs)} chunks")
    return Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name="meridian_graph_baseline",
    )


VECTOR_PROMPT = """Answer the question using only the context below. If the \
context does not contain enough to answer, say exactly what is missing.

Context:
{context}

Question: {question}

Answer:"""


def vector_rag(store: Chroma, question: str) -> tuple[str, list[str]]:
    docs = store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(f"[{d.metadata['doc_id']}] {d.page_content}" for d in docs)
    answer = generator_llm.invoke(
        VECTOR_PROMPT.format(context=context, question=question)
    ).content
    return answer, [d.metadata["doc_id"] for d in docs]


# ==========================================================================
# STEP 1 — GRAPH EXTRACTION
#
# One LLM call per document. This is the index-time cost of graph RAG.
# ==========================================================================


class Triple(BaseModel):
    subject: str = Field(description="The entity the fact is about. A proper noun where possible.")
    relation: str = Field(description="A short verb phrase, lowercase, e.g. 'is led by', 'uses'.")
    object: str = Field(description="The entity or value the subject relates to.")


class Extraction(BaseModel):
    triples: list[Triple] = Field(description="All factual relationships stated in the document.")


extractor = extract_llm.with_structured_output(Extraction)

EXTRACTION_PROMPT = """Extract every factual relationship in the document below \
as (subject, relation, object) triples.

Rules:
- Use the exact entity names as written. Do not abbreviate or paraphrase them.
- One fact per triple. Split compound sentences.
- Relations should be short lowercase verb phrases: "is led by", "uses",
  "reports to", "was acquired by", "is based in", "was formed in".
- Include organisational, ownership, location, tooling and reporting facts.
- Do not invent facts that are not stated.

Document ({doc_id} — {title}):
{text}
"""


def build_graph() -> nx.MultiDiGraph:
    """Read every document with an LLM and assemble a knowledge graph."""
    g = nx.MultiDiGraph()

    for entry in CORPUS:
        result = extractor.invoke(
            EXTRACTION_PROMPT.format(
                doc_id=entry["id"],
                title=entry["title"],
                text=textwrap.dedent(entry["text"]).strip(),
            )
        )
        print(f"  {entry['id']}  {len(result.triples):>2} triples")
        for t in result.triples:
            s, o = t.subject.strip(), t.object.strip()
            if not s or not o:
                continue
            # Nodes are keyed by lowercase name so 'Ananya Rao' and 'ananya rao'
            # collapse. Real systems need proper entity resolution; this is the
            # cheapest thing that works on a corpus this size.
            g.add_node(s.lower(), label=s)
            g.add_node(o.lower(), label=o)
            g.add_edge(s.lower(), o.lower(), relation=t.relation.strip(), source=entry["id"])

    print(f"\n  graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


# ==========================================================================
# STEP 2 — LOCAL SEARCH (multi-hop)
# ==========================================================================


class Entities(BaseModel):
    names: list[str] = Field(description="Proper nouns and named things mentioned in the question.")


entity_extractor = summary_llm.with_structured_output(Entities)


def match_nodes(graph: nx.MultiDiGraph, names: list[str]) -> list[str]:
    """Substring match against node keys. Crude, transparent, good enough here.

    In production this is an embedding lookup over node labels, or a proper
    entity-resolution service. It is also the step where graph RAG most often
    quietly fails — if the question names an entity the graph does not have,
    the walk starts nowhere and you get an empty subgraph.
    """
    matched: list[str] = []
    for name in names:
        key = name.lower().strip()
        for node in graph.nodes:
            if key and (key in node or node in key) and node not in matched:
                matched.append(node)
    return matched


def subgraph_triples(graph: nx.MultiDiGraph, seeds: list[str], hops: int = HOPS) -> list[str]:
    """Walk `hops` steps out from each seed and serialise the edges as text."""
    undirected = graph.to_undirected(as_view=True)
    reachable: set[str] = set()
    for seed in seeds:
        if seed in undirected:
            reachable |= set(nx.single_source_shortest_path_length(undirected, seed, cutoff=hops))

    lines: list[str] = []
    for u, v, data in graph.edges(data=True):
        if u in reachable and v in reachable:
            lines.append(
                f"{graph.nodes[u]['label']} --[{data['relation']}]--> "
                f"{graph.nodes[v]['label']}   ({data['source']})"
            )
    return sorted(set(lines))


LOCAL_PROMPT = """You are answering from a knowledge graph. Each line below is a \
fact of the form: SUBJECT --[relation]--> OBJECT (source document).

Chain the facts together to answer the question. Show the chain you followed.
If the facts do not connect, say which link is missing.

Facts:
{facts}

Question: {question}

Answer:"""


def local_search(graph: nx.MultiDiGraph, question: str) -> tuple[str, int]:
    ents = entity_extractor.invoke(
        f"List the named entities in this question: {question}"
    ).names
    seeds = match_nodes(graph, ents)
    print(f"  entities: {ents}")
    print(f"  matched nodes: {seeds or 'NONE — the walk will start nowhere'}")

    facts = subgraph_triples(graph, seeds)
    print(f"  subgraph: {len(facts)} facts within {HOPS} hops")

    if not facts:
        return "No entities from the question were found in the graph.", 0

    answer = generator_llm.invoke(
        LOCAL_PROMPT.format(facts="\n".join(facts), question=question)
    ).content
    return answer, len(facts)


# ==========================================================================
# STEP 3 — GLOBAL SEARCH (community summarisation, map-reduce)
# ==========================================================================


def detect_communities(graph: nx.MultiDiGraph) -> list[list[str]]:
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes)
    simple.add_edges_from((u, v) for u, v, _ in graph.edges(data=True))
    return [sorted(c) for c in greedy_modularity_communities(simple)]


COMMUNITY_PROMPT = """Below are the facts belonging to one cluster of a knowledge \
graph about Meridian Financial Services.

Write a 3-4 sentence summary of what this cluster is about, naming the key
entities and what connects them.

Facts:
{facts}
"""


def summarise_communities(graph: nx.MultiDiGraph) -> list[str]:
    """One LLM call per community. This is GraphRAG's index-time summarisation."""
    summaries: list[str] = []
    for i, community in enumerate(detect_communities(graph), start=1):
        members = set(community)
        facts = [
            f"{graph.nodes[u]['label']} --[{d['relation']}]--> {graph.nodes[v]['label']}"
            for u, v, d in graph.edges(data=True)
            if u in members and v in members
        ]
        if len(facts) < 2:
            continue  # a two-node fragment is not a community worth summarising
        s = summary_llm.invoke(COMMUNITY_PROMPT.format(facts="\n".join(facts))).content
        print(f"  community {i}: {len(members)} nodes, {len(facts)} facts")
        summaries.append(s.strip())
    return summaries


REDUCE_PROMPT = """You are answering a broad question about an organisation using \
summaries of different clusters of its knowledge graph. No single summary holds
the whole answer — synthesise across them.

Cluster summaries:
{summaries}

Question: {question}

Answer:"""


def global_search(summaries: list[str], question: str) -> str:
    joined = "\n\n".join(f"--- cluster {i} ---\n{s}" for i, s in enumerate(summaries, 1))
    return generator_llm.invoke(
        REDUCE_PROMPT.format(summaries=joined, question=question)
    ).content


# ==========================================================================
# DEMO
# ==========================================================================


def banner(text: str, char: str = "=") -> None:
    print(f"\n{char * 74}\n {text}\n{char * 74}")


MULTIHOP_QUESTIONS = [
    (
        "Q1",
        "Which compliance system is used by the team that led the Kalyani "
        "Motors transaction?",
        "Three hops across three documents: Kalyani Motors -> Acquisitions "
        "Desk -> Sentinel. No chunk contains the chain. Expect vector RAG to "
        "retrieve DOC-02 and stop.",
    ),
    (
        "Q2",
        "Who does the head of the desk that uses Sentinel ultimately report to?",
        "Sentinel -> Acquisitions Desk -> Meera Iyer -> Ananya Rao -> Vikram "
        "Set. Four hops. Increase HOPS if the graph walk comes back short.",
    ),
    (
        "Q3",
        "What is the connection between Torvale Systems and the Structured "
        "Credit Desk?",
        "Two independent links: Torvale licenses Aegis which the desk uses, "
        "AND Joseph Mathew came from Torvale. A good graph answer finds both; "
        "vector RAG typically finds one.",
    ),
]

GLOBAL_QUESTION = (
    "What are the main concentration risks facing Meridian, and which parts of "
    "the organisation do they touch?"
)


def main() -> None:
    banner("BASELINE — building the vector index")
    store = build_vectorstore()

    banner("GRAPH EXTRACTION — one LLM call per document")
    graph = build_graph()

    # ---------------------------------------------------------------- local
    for tag, question, note in MULTIHOP_QUESTIONS:
        banner(f"{tag} — {question}")
        print(f" Why this question: {note}\n")

        print("-" * 74)
        print(" VECTOR RAG")
        print("-" * 74)
        v_answer, v_ids = vector_rag(store, question)
        print(f"  retrieved: {', '.join(v_ids)}")
        print(textwrap.indent(textwrap.fill(v_answer, 70), "  "))

        print("\n" + "-" * 74)
        print(" GRAPH RAG — local search")
        print("-" * 74)
        g_answer, n_facts = local_search(graph, question)
        print(textwrap.indent(textwrap.fill(g_answer, 70), "  "))

    # --------------------------------------------------------------- global
    banner("COMMUNITY DETECTION AND SUMMARISATION")
    summaries = summarise_communities(graph)

    banner(f"GLOBAL — {GLOBAL_QUESTION}")
    print(
        " Why this question: no chunk and no single subgraph contains the "
        "answer.\n It is a property of the whole corpus.\n"
    )

    print("-" * 74)
    print(" VECTOR RAG")
    print("-" * 74)
    v_answer, v_ids = vector_rag(store, GLOBAL_QUESTION)
    print(f"  retrieved: {', '.join(v_ids)}")
    print(textwrap.indent(textwrap.fill(v_answer, 70), "  "))

    print("\n" + "-" * 74)
    print(" GRAPH RAG — global search over community summaries")
    print("-" * 74)
    print(textwrap.indent(textwrap.fill(global_search(summaries, GLOBAL_QUESTION), 70), "  "))

    banner("NOW ANSWER THESE, IN YOUR OWN WORDS", "-")
    print(
        """
 1. On Q1, what did vector RAG actually retrieve, and what did its answer say
    about the part it could not find? Was it honest or was it confident?

 2. Print the extracted graph (add `print(graph.edges(data=True))`). Find one
    triple that is wrong or badly normalised. Extraction quality is the
    ceiling on everything downstream — what would that error break?

 3. Set HOPS = 1 and re-run Q2. Where does the chain break, and what does the
    generator do when the chain is incomplete?

 4. Count the index-time LLM calls for graph RAG versus vector RAG on this
    8-document corpus. Now multiply by 200,000 documents. At what corpus size
    does this stop being an obvious win, and what would you measure to decide?

 5. Global search summarises communities at index time. What happens to those
    summaries when one new document is added — and what does that imply for a
    corpus that changes daily?
"""
    )


if __name__ == "__main__":
    main()
