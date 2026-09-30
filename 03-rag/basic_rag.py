"""One-file basic RAG lab: corpus, FAISS, grounded generation, and evaluation."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PERSIST_DIR = Path("faiss_index")
COLLECTION_NAME = "it_companies"
CORPUS_DIR = Path("data/company_profiles")
EVAL_PATH = Path("eval_set.json")
OUTPUT_DIR = Path("outputs")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 5
THRESHOLDS = [0.5, 0.7, 0.8]
REFUSAL = "I don't have that information."
STORE_KWARGS = {
    "normalize_L2": True,
    "distance_strategy": DistanceStrategy.EUCLIDEAN_DISTANCE,
}

# Company code per corpus file, used to tag chunks for retrieval evaluation.
COMPANY_BY_FILE = {
    "tcs.txt": "TCS",
    "infosys.txt": "Infosys",
    "wipro.txt": "Wipro",
    "hcltech.txt": "HCLTech",
    "tech_mahindra.txt": "TechMahindra",
}

GROUNDED_PROMPT = ChatPromptTemplate.from_template(
    '''Answer the question using ONLY the information in the context below.

If the answer is not in the context, reply exactly:
"I don't have that information."

Answer in 2-3 sentences. Start by directly answering the question.
After your answer, add a line beginning "Source:" citing the context used.

Context:
{context}

Question: {question}

Answer:'''
)
UNGROUNDED_PROMPT = ChatPromptTemplate.from_template(
    '''Use the context below to help you answer the question.

Answer in 2-3 sentences. Start by directly answering the question.
After your answer, add a line beginning "Source:" citing the context used.

Context:
{context}

Question: {question}

Answer:'''
)


def cosine_relevance(distance: float) -> float:
    """Convert normalized squared-L2 distance into cosine similarity."""
    return 1.0 - distance / 2.0


def require_api_key() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Add it to a .env file in this folder.")


def load_documents():
    documents = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        document = TextLoader(str(path), encoding="utf-8").load()[0]
        document.metadata.update(
            company=COMPANY_BY_FILE.get(path.name, path.stem),
            source=path.name,
            doc_type="profile",
        )
        documents.append(document)
    if not documents:
        raise SystemExit(f"No .txt profiles found in {CORPUS_DIR}.")
    return documents


def load_or_build_store(chunks, embeddings, rebuild: bool):
    index_path = PERSIST_DIR / f"{COLLECTION_NAME}.faiss"
    if rebuild or not index_path.exists():
        ids = [f"{chunk.metadata['company']}_{index:03d}" for index, chunk in enumerate(chunks)]
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
            ids=ids,
            relevance_score_fn=cosine_relevance,
            **STORE_KWARGS,
        )
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(PERSIST_DIR), COLLECTION_NAME)
    else:
        vectorstore = FAISS.load_local(
            str(PERSIST_DIR),
            embeddings,
            COLLECTION_NAME,
            allow_dangerous_deserialization=True,
            relevance_score_fn=cosine_relevance,
            **STORE_KWARGS,
        )
        if vectorstore.index.ntotal != len(chunks):
            raise SystemExit("Saved index does not match corpus. Re-run with --rebuild.")
    return vectorstore


def format_docs(documents) -> str:
    return "\n\n---\n\n".join(document.page_content for document in documents)


def print_retrieval_examples(vectorstore) -> None:
    examples = [
        ("Who is the CEO of TCS?", "in-domain"),
        ("What is Infosys revenue growth guidance for FY2025?", "in-domain"),
        ("Compare the founding years of TCS and Infosys", "multi-hop"),
        ("Which Indian IT company was founded earliest?", "multi-hop"),
        ("Who won the FIFA World Cup in 2022?", "out-of-domain"),
    ]
    print("\n=== 1. Similarity retrieval ===")
    for index, (question, question_type) in enumerate(examples, 1):
        results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
        print(f"\n[Q{index}] {question} ({question_type})")
        for rank, (document, score) in enumerate(results, 1):
            preview = " ".join(document.page_content.split())[:100]
            print(
                f"  {rank}. score={score:.3f} "
                f"company={document.metadata.get('company', '?'):<12} {preview}..."
            )
        scores = [score for _, score in results]
        print(f"  top={max(scores, default=0):.3f}; mean={mean(scores) if scores else 0:.3f}")


def build_chain(retriever, prompt, llm):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def run_rag_examples(vectorstore, skip_ablation: bool) -> None:
    print("\n=== 2. Grounded LCEL RAG ===")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rag_chain = build_chain(retriever, GROUNDED_PROMPT, llm)
    tests = [
        ("Who is the CEO of TCS?", ["krithivasan"]),
        ("What is Infosys revenue guidance for FY2025?", ["3", "4"]),
        ("What is the stock price of TCS today?", [REFUSAL.lower()]),
        ("Compare the founding years of TCS and Infosys", ["1968", "1981"]),
        ("Where is Wipro headquartered and how many countries does it serve?", ["bengaluru", "66"]),
    ]
    passed = 0
    for index, (question, expected) in enumerate(tests, 1):
        answer = rag_chain.invoke(question)
        okay = all(value.lower() in answer.lower() for value in expected)
        passed += int(okay)
        print(f"\n[Q{index}] {question}\n{answer}\nCheck: {'PASS' if okay else 'CHECK'}")
    print(f"RAG checks: {passed}/{len(tests)}")

    if not skip_ablation:
        ungrounded_chain = build_chain(retriever, UNGROUNDED_PROMPT, llm)
        question = "What is the stock price of TCS today?"
        grounded = rag_chain.invoke(question)
        ungrounded = ungrounded_chain.invoke(question)
        verdict = "refused" if REFUSAL.lower() in ungrounded.lower() else "attempted an unsupported answer"
        print("\n--- Prompt ablation: remove ONLY and the refusal rule ---")
        print(f"Grounded:   {grounded}\nUngrounded: {ungrounded}\nUngrounded model {verdict}.")


def load_eval_set():
    if not EVAL_PATH.exists():
        raise SystemExit(f"Evaluation set not found: {EVAL_PATH}")
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def evaluate_question(retrieved, gold_companies):
    companies = [document.metadata.get("company") for document in retrieved]
    relevant = sum(company in gold_companies for company in companies)
    if not gold_companies:
        return (1.0 if not retrieved else 0.0), None, 0, relevant
    covered = len(set(companies) & set(gold_companies))
    precision = relevant / len(retrieved) if retrieved else 0.0
    recall = covered / len(gold_companies)
    return precision, recall, covered, relevant


def run_retrieval_evaluation(vectorstore) -> None:
    print("\n=== 3. Threshold evaluation ===")
    evaluation = load_eval_set()
    summary = []
    for threshold in THRESHOLDS:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": threshold, "k": TOP_K},
        )
        rows = []
        for item in evaluation:
            retrieved = retriever.invoke(item["question"])
            precision, recall, covered, relevant = evaluate_question(
                retrieved, item["gold_companies"]
            )
            rows.append(
                {
                    "id": item["id"],
                    "type": item["type"],
                    "retrieved": len(retrieved),
                    "relevant": relevant,
                    "gold_covered": (
                        f"{covered}/{len(item['gold_companies'])}"
                        if item["gold_companies"]
                        else "-"
                    ),
                    "precision": precision,
                    "recall": recall,
                }
            )
        detail = pd.DataFrame(rows)
        print(f"\nThreshold {threshold:.2f}")
        print(detail.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

        precision = detail.precision.mean()
        recall = detail.recall.dropna().mean()
        print(f"Macro mean: precision={precision:.3f}; recall={recall:.3f}")
        summary.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0,
                "mean_chunks_retrieved": detail.retrieved.mean(),
            }
        )

    summary_frame = pd.DataFrame(summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "threshold_results.csv"
    plot_path = OUTPUT_DIR / "precision_recall.png"
    summary_frame.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(summary_frame.threshold, summary_frame.precision, marker="o", label="Precision")
    plt.plot(summary_frame.threshold, summary_frame.recall, marker="o", label="Recall")
    plt.axhline(0.80, linestyle="--", label="Precision target 0.80")
    plt.axhline(0.75, linestyle="--", label="Recall target 0.75")
    plt.title("Retrieval precision and recall by FAISS score threshold")
    plt.ylim(0, 1.05)
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Macro score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    qualifying = summary_frame[
        (summary_frame.precision >= 0.80) & (summary_frame.recall >= 0.75)
    ]
    recommendation = (
        qualifying.sort_values("f1").iloc[-1]
        if not qualifying.empty
        else summary_frame.sort_values("f1").iloc[-1]
    )
    print("\nSummary by threshold:")
    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Recommended threshold by F1: {recommendation.threshold:.2f}")
    print(f"Saved {csv_path} and {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="recreate the persisted FAISS index")
    parser.add_argument("--skip-rag", action="store_true", help="skip chat completions; run retrieval and evaluation only")
    parser.add_argument("--skip-ablation", action="store_true", help="skip the extra prompt ablation completions")
    args = parser.parse_args()

    require_api_key()
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"Documents loaded: {len(documents)}")
    print(
        f"Chunks: {len(chunks)}; length min/mean/max: "
        f"{min(lengths)}/{mean(lengths):.1f}/{max(lengths)} characters"
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = load_or_build_store(chunks, embeddings, args.rebuild)
    print(f"FAISS index '{COLLECTION_NAME}' contains {vectorstore.index.ntotal} vectors")

    print_retrieval_examples(vectorstore)
    if not args.skip_rag:
        run_rag_examples(vectorstore, args.skip_ablation)
    run_retrieval_evaluation(vectorstore)


if __name__ == "__main__":
    main()
