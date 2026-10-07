import os  # Standard library module used here to read environment variables (API keys, model name)
import re  # Regular expressions: used to tokenise text for the offline hashing embedding
from typing import Any, Dict, List, Tuple  # Type hints for mappings, sequences and fixed-shape tuples

import faiss  # FAISS: Facebook AI Similarity Search — the in-process vector index used for RAG retrieval
import numpy as np  # Numerical arrays: used to build and normalise the embedding vectors fed into FAISS
from fastapi import FastAPI, HTTPException  # FastAPI: web framework; HTTPException: raise HTTP error responses
from pydantic import BaseModel  # BaseModel: declarative request-body validation/parsing

try:  # Attempt an optional import so the app still runs without the openai package installed
    from openai import OpenAI  # OpenAI client class used to call the Chat Completions and Embeddings APIs
except Exception:  # pragma: no cover - optional dependency for local demos  # Any import failure is tolerated
    OpenAI = None  # Sentinel: code later checks `OpenAI is not None` before using it

app = FastAPI(title="Customer Support RAG Agent", version="1.0.0")  # Create the ASGI app with metadata for docs

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # Embedding model name, overridable via env
EMBED_DIM = 512  # Dimensionality of the offline hashing embedding (only used when OpenAI is unavailable)
SIMILARITY_THRESHOLD = 0.2  # Minimum cosine score for a vector hit to be trusted as a relevant answer
STOPWORDS = frozenset(  # Common low-signal words dropped before hashing so the offline embedding stays sharp
    "a an the is are was were be been do does did to of in on at for with and or how what "
    "i my me you your it its this that can could would should will please need help".split()
)


class AgentRequest(BaseModel):  # Schema describing the JSON body accepted by the /agent endpoint
    question: str  # Required field: the user's support question
    context: str | None = None  # Optional field: extra context; defaults to None if omitted


FAQ_KB: Dict[str, str] = {  # In-memory "knowledge base": maps a topic keyword to a canned answer
    "refund": "Refunds are processed within 5-7 business days for eligible orders. To request a refund, open the billing section in your account and select 'Request refund'.",  # Answer returned when the question is about refunds
    "password": "Reset your password from the sign-in screen by selecting 'Forgot password'. If you are locked out, contact support with your account email for manual assistance.",  # Answer for password questions
    "billing": "Billing updates are available in the Settings > Billing page. You can add a payment method, review invoices, and update your subscription plan there.",  # Answer for billing questions
    "shipping": "Standard shipping usually arrives in 3-5 business days. Tracking details are shown in the order history section once the shipment has been dispatched.",  # Answer for shipping questions
    "subscription": "You can downgrade or cancel your plan from Settings > Subscription. Changes take effect at the end of the current billing cycle.",  # Answer for subscription questions
    "delivery": "If your package is delayed, please check the tracking link or contact support with your order number and courier reference.",  # Answer for delivery questions
}


def _hash_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:  # Deterministic offline embedding (no network)
    vector = np.zeros(dim, dtype="float32")  # Start with an all-zeros vector of the target dimensionality
    for token in re.findall(r"[a-z0-9]+", text.lower()):  # Lowercase then split into alphanumeric word tokens
        if len(token) < 3 or token in STOPWORDS:  # Skip very short and low-signal tokens that only add noise
            continue  # Move on to the next token
        vector[hash(token) % dim] += 1.0  # Bag-of-words hashing trick: bump the bucket this token maps to
    norm = float(np.linalg.norm(vector))  # L2 norm of the vector (0.0 if the text had no usable tokens)
    return vector / norm if norm else vector  # Return a unit vector so dot products act as cosine similarity


def embed_texts(texts: List[str]) -> Tuple[np.ndarray, str]:  # Embed a batch of texts, reporting which backend ran
    api_key = os.getenv("OPENAI_API_KEY")  # Read the OpenAI API key from the environment (None if unset)
    if api_key and OpenAI is not None:  # Only call the API if we have both a key and the client library
        try:  # Network/API calls can fail; fall back to the offline embedding on any error
            client = OpenAI(api_key=api_key)  # Instantiate the OpenAI client with the provided key
            response = client.embeddings.create(model=EMBED_MODEL, input=texts)  # Request one embedding per input
            matrix = np.array([item.embedding for item in response.data], dtype="float32")  # Stack into a 2D array
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # Per-row L2 norms for normalisation
            norms[norms == 0] = 1.0  # Guard against divide-by-zero for any all-zero row
            return matrix / norms, "openai"  # Unit-normalised embeddings, tagged with the backend used
        except Exception:  # Any error (auth, rate limit, network, parsing) -> degrade to offline embeddings
            pass  # Fall through to the hashing embedding below
    return np.vstack([_hash_embedding(text) for text in texts]), "hashing"  # Offline fallback embedding matrix


class FaissVectorStore:  # In-process vector database backed by a FAISS index: replaces the ChromaDB dependency
    def __init__(self, documents: Dict[str, str]) -> None:  # Build the index from a {key: text} knowledge base
        self.keys: List[str] = list(documents.keys())  # Stable ordered list of document keys (topic names)
        self.texts: List[str] = list(documents.values())  # Matching ordered list of document bodies (answers)
        passages = [f"{key}: {text}" for key, text in zip(self.keys, self.texts)]  # Combine key + body per doc
        matrix, self.backend = embed_texts(passages)  # Embed every document; remember which backend produced it
        self.dim: int = matrix.shape[1]  # Embedding dimensionality (depends on which backend produced the vectors)
        self.index = faiss.IndexFlatIP(self.dim)  # Flat index over inner product; unit vectors -> cosine similarity
        self.index.add(np.ascontiguousarray(matrix, dtype="float32"))  # Load all document vectors into the index

    def search(self, query: str, top_k: int = 1) -> List[Tuple[str, str, float]]:  # Nearest-neighbour lookup
        query_matrix, _ = embed_texts([query])  # Embed the query with the same backend logic as the documents
        vectors = np.ascontiguousarray(query_matrix, dtype="float32")  # FAISS needs a contiguous float32 array
        scores, indices = self.index.search(vectors, min(top_k, len(self.keys)))  # Top-k search: scores + row ids
        return [  # Map FAISS row ids back to (key, answer, score) tuples, best match first
            (self.keys[idx], self.texts[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1  # FAISS returns -1 for empty slots when fewer than top_k results exist
        ]


VECTOR_STORE = FaissVectorStore(FAQ_KB)  # Build the FAISS-backed vector store once at startup from the FAQ KB


def retrieve_context(question: str) -> str | None:  # Pull the most relevant KB answer for a question, if any
    matches = VECTOR_STORE.search(question, top_k=1)  # Ask the vector store for the single best match
    if matches and matches[0][2] >= SIMILARITY_THRESHOLD:  # Trust it only when the similarity clears the threshold
        return matches[0][1]  # Return the matched answer text to use as grounding context
    return None  # No confident match found


def fallback_rag_answer(question: str) -> str:  # Offline answer generator used when the LLM is unavailable
    q = question.lower()  # Normalise to lowercase so keyword matching is case-insensitive
    for keyword, answer in FAQ_KB.items():  # First try an exact keyword match for fully deterministic behaviour
        if keyword in q:  # If the topic keyword appears anywhere in the question text
            return answer  # Return that topic's canned answer immediately
    context = retrieve_context(question)  # Otherwise fall back to semantic retrieval from the vector store
    if context:  # A confident vector match was found
        return context  # Use the retrieved answer directly
    return (  # Nothing matched: return a generic help message
        "I can help with account, billing, shipping, and subscription issues. "  # First part of the default reply
        "Please share your order number or account email if you need a more specific resolution."  # Ask for identifying info
    )


def get_rag_answer(question: str) -> str:  # Main answer function: retrieve context, try the LLM, fall back to the KB
    context = retrieve_context(question)  # Retrieve grounding context from the FAISS vector store
    api_key = os.getenv("OPENAI_API_KEY")  # Read the OpenAI API key from the environment (None if unset)
    if api_key and OpenAI is not None:  # Only call the API if we have both a key and the client library
        try:  # Network/API calls can fail; guard so failures degrade gracefully
            client = OpenAI(api_key=api_key)  # Instantiate the OpenAI client with the provided key
            user_content = question if not context else (  # Prepend retrieved context to the question when available
                f"Knowledge base context:\n{context}\n\nCustomer question: {question}"  # Grounded prompt format
            )
            response = client.chat.completions.create(  # Send a chat completion request
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # Model name from env, defaulting to gpt-4o-mini
                messages=[  # Conversation passed to the model
                    {  # System message: sets the assistant's role and guardrails
                        "role": "system",  # Marks this as instruction context, not user input
                        "content": (  # The instruction text itself
                            "You are a helpful customer support agent. Answer clearly and concisely "  # Tone/brevity guidance
                            "using only the information in the knowledge base context and avoid making up unsupported details."  # Anti-hallucination guidance
                        ),
                    },
                    {"role": "user", "content": user_content},  # User message: the (optionally grounded) question
                ],
                temperature=0.2,  # Low randomness for consistent, factual support answers
                max_tokens=250,  # Cap the response length to keep answers short and costs bounded
            )
            return response.choices[0].message.content.strip() or fallback_rag_answer(question)  # Trimmed model text, or fall back if empty
        except Exception:  # Any error (auth, rate limit, network, parsing)
            return fallback_rag_answer(question)  # Degrade gracefully to the offline KB answer
    return fallback_rag_answer(question)  # No key or no library: use the offline KB answer


def check_openai() -> Dict[str, Any]:  # Health probe: report whether the OpenAI API is usable
    api_key = os.getenv("OPENAI_API_KEY")  # Read the API key from the environment
    if not api_key:  # No key configured
        return {"status": "skipped", "detail": "OPENAI_API_KEY is not set"}  # Not an error, just not configured
    if OpenAI is None:  # Key present but the client library isn't installed
        return {"status": "error", "detail": "openai package is not installed"}  # Report the missing dependency
    try:  # The actual connectivity check can raise
        client = OpenAI(api_key=api_key)  # Build the client
        client.models.list()  # Cheap authenticated call to verify the key and reachability
        return {"status": "ok", "detail": "OpenAI API reachable"}  # Success
    except Exception as exc:  # pragma: no cover - depends on external service  # Any failure calling the API
        return {"status": "error", "detail": str(exc)}  # Surface the error message for debugging


def check_vector_store() -> Dict[str, Any]:  # Health probe: report whether the FAISS vector store is ready
    try:  # Building/querying the store can raise (bad embeddings, empty KB)
        matches = VECTOR_STORE.search("refund policy", top_k=1)  # Run a known query to confirm search works end-to-end
        return {  # Success response describing the store
            "status": "ok",
            "detail": (  # Human-readable summary of the store's shape and backend
                f"FAISS index ready: {VECTOR_STORE.index.ntotal} vector(s), "
                f"backend={VECTOR_STORE.backend}, dim={VECTOR_STORE.dim}, "
                f"top match='{matches[0][0]}'"
            ),
        }
    except Exception as exc:  # pragma: no cover - depends on local state  # Any failure building/querying the store
        return {"status": "error", "detail": str(exc)}  # Surface the error message


@app.get("/health")  # Register an HTTP GET endpoint at /health
def health() -> Dict[str, Any]:  # Aggregate health check for all downstream services
    openai_status = check_openai()  # Run the OpenAI probe
    vector_status = check_vector_store()  # Run the FAISS vector store probe
    services = {  # Collect individual probe results by service name
        "openai": openai_status,
        "vector_store": vector_status,
    }
    all_services_ok = all(service["status"] == "ok" for service in services.values())  # True only if every probe returned "ok"
    return {  # Overall health payload
        "status": "ok" if all_services_ok else "degraded",  # "ok" when everything passes, otherwise "degraded"
        "all_services_ok": all_services_ok,  # Boolean convenience flag for callers
        "services": services,  # Per-service detail for diagnosis
    }


@app.post("/agent")  # Register an HTTP POST endpoint at /agent
def agent_route(payload: AgentRequest) -> Dict[str, Any]:  # Handler; FastAPI parses/validates the body into AgentRequest
    question = (payload.question or "").strip()  # Normalise: treat missing text as "" and trim surrounding whitespace
    if not question:  # Reject empty or whitespace-only questions
        raise HTTPException(status_code=400, detail="Question must not be empty.")  # Return HTTP 400 Bad Request

    answer = get_rag_answer(question)  # Produce an answer (retrieval + LLM if available, otherwise KB fallback)
    return {  # JSON response echoed back to the caller
        "question": question,  # The cleaned question that was answered
        "answer": answer,  # The generated answer
        "status": "ok",  # Indicates the request was handled successfully
    }


if __name__ == "__main__":  # Only run the server when this file is executed directly (not imported)
    import uvicorn  # ASGI server used to serve the FastAPI app

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # Serve app on all interfaces, port 8000, auto-reload on code changes
