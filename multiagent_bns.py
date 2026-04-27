import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BNS_SOURCE_URL = "https://en.wikisource.org/wiki/Bharatiya_Nyaya_Sanhita,_2023"


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    meta: Dict[str, Any]


class FaissMemory:
    """Stores both law corpus chunks and conversation memory in one FAISS index."""

    def __init__(self, dim: int, db_dir: str = "./vector_db"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.db_dir / "index.faiss"
        self.meta_path = self.db_dir / "meta.json"

        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.meta: List[Dict[str, Any]] = json.loads(self.meta_path.read_text())
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.meta = []

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.meta.extend(
            [
                {
                    "id": c.id,
                    "text": c.text,
                    "source": c.source,
                    "meta": c.meta,
                }
                for c in chunks
            ]
        )
        self.persist()

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype("float32")
        faiss.normalize_L2(query_vector)
        scores, idxs = self.index.search(query_vector, k)

        rows = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            row = dict(self.meta[idx])
            row["score"] = float(score)
            rows.append(row)
        return rows

    def persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.meta, indent=2))


class Embeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return arr.astype("float32")


class BNSIngestor:
    @staticmethod
    def fetch_bns_text(url: str = BNS_SOURCE_URL) -> str:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        content = soup.find("div", class_="mw-parser-output")
        if not content:
            raise RuntimeError("Unable to parse Bharatiya Nyaya Sanhita page")

        paragraphs = []
        for tag in content.find_all(["p", "li"]):
            text = tag.get_text(" ", strip=True)
            if not text:
                continue
            if len(text) < 30:
                continue
            paragraphs.append(text)

        return "\n".join(paragraphs)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= chunk_size:
            return [normalized]

        chunks: List[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + chunk_size)
            chunks.append(normalized[start:end])
            if end == len(normalized):
                break
            start = max(0, end - overlap)
        return chunks

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        pages: List[str] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if not page_text:
                continue
            pages.append(f"[Page {page_idx}]\n{page_text}")
        if not pages:
            raise RuntimeError(f"No readable text found in PDF: {pdf_path}")
        return "\n\n".join(pages)


class MultiAgentSystem:
    """Three-agent architecture with OpenAI-style tool calling over a vLLM endpoint."""

    def __init__(
        self,
        model_name: str,
        vllm_base_url: str,
        api_key: str = "EMPTY",
        db_dir: str = "./vector_db",
    ):
        self.client = OpenAI(base_url=vllm_base_url, api_key=api_key)
        self.model_name = model_name
        self.embedder = Embeddings()
        self.memory = FaissMemory(dim=384, db_dir=db_dir)

    # ---------- ingestion ----------
    def ingest_bns(self, pdf_path: str | None = None) -> None:
        if pdf_path:
            raw = BNSIngestor.extract_text_from_pdf(pdf_path)
            source_meta = {"doc": "Bharatiya Nyaya Sanhita, PDF", "path": pdf_path}
        else:
            raw = BNSIngestor.fetch_bns_text()
            source_meta = {"doc": "Bharatiya Nyaya Sanhita, 2023", "url": BNS_SOURCE_URL}
        chunks = BNSIngestor.chunk_text(raw)

        records = [
            Chunk(
                id=str(uuid.uuid4()),
                text=chunk,
                source="bns",
                meta=source_meta,
            )
            for chunk in chunks
        ]
        vecs = self.embedder.embed([r.text for r in records])
        self.memory.add(vecs, records)
        print(f"Ingested {len(records)} BNS chunks into FAISS from {'PDF' if pdf_path else 'web source'}.")

    # ---------- conversation memory ----------
    def store_turn(self, user_query: str, assistant_answer: str) -> None:
        text = f"User: {user_query}\nAssistant: {assistant_answer}"
        vec = self.embedder.embed([text])
        row = Chunk(
            id=str(uuid.uuid4()),
            text=text,
            source="conversation",
            meta={"kind": "chat_turn"},
        )
        self.memory.add(vec, [row])

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qvec = self.embedder.embed([query])
        return self.memory.search(qvec, k=k)

    # ---------- agents ----------
    def supervisor_route(self, query: str) -> Literal["direct", "rag"]:
        """Supervisor decides whether retrieval is needed via tool-calling."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "route_query",
                    "description": "Route user query either to direct answer or RAG pipeline",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "route": {
                                "type": "string",
                                "enum": ["direct", "rag"],
                                "description": "Choose rag for legal/factual questions that need grounding in corpus.",
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["route", "reason"],
                    },
                },
            }
        ]

        system = (
            "You are a supervisor agent for a legal assistant. "
            "If the question can be answered safely from general reasoning, choose direct. "
            "If the question needs statute-grounded information or exact legal context, choose rag."
        )

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )

        msg = completion.choices[0].message
        if msg.tool_calls:
            args = json.loads(msg.tool_calls[0].function.arguments)
            route = args.get("route", "rag")
            print(f"Supervisor route: {route}. Reason: {args.get('reason', '')}")
            return route  # type: ignore[return-value]

        # fallback if model doesn't produce tool call
        text = (msg.content or "").lower()
        return "rag" if "rag" in text or "retriev" in text else "direct"

    def rag_agent_answer(self, query: str) -> str:
        ctx_rows = self.retrieve_context(query, k=6)
        ctx = "\n\n".join([f"[{r['source']}] {r['text']}" for r in ctx_rows])
        system = (
            "You are the RAG legal agent. Answer only from provided context. "
            "If context is insufficient, explicitly say so."
        )
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{ctx}"},
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content or "No answer generated."

    def direct_agent_answer(self, query: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a concise helpful assistant."},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content or "No answer generated."

    def critique_agent(self, query: str, draft_answer: str, route: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a critique agent. Evaluate correctness, missing details, and whether routing "
                        "(direct vs rag) looked appropriate. Then provide an improved final answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nRoute used: {route}\nDraft answer: {draft_answer}",
                },
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content or draft_answer

    def ask(self, query: str) -> Dict[str, str]:
        route = self.supervisor_route(query)
        if route == "rag":
            draft = self.rag_agent_answer(query)
        else:
            draft = self.direct_agent_answer(query)

        final = self.critique_agent(query, draft, route)
        self.store_turn(query, final)
        return {"route": route, "draft": draft, "final": final}

    def evaluate_routing(self, test_queries: List[str]) -> List[Dict[str, str]]:
        """Quick harness to test if GPT-OSS routes as expected using a simple heuristic label."""
        results = []
        legal_keywords = ["section", "punishment", "offence", "bns", "bharatiya", "nyaya", "law", "legal"]

        for q in test_queries:
            predicted = self.supervisor_route(q)
            expected = "rag" if any(k in q.lower() for k in legal_keywords) else "direct"
            results.append(
                {
                    "query": q,
                    "predicted": predicted,
                    "expected": expected,
                    "correct": str(predicted == expected),
                }
            )
        return results


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "gpt-oss")
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    bns_pdf_path = os.getenv("BNS_PDF_PATH")

    agent = MultiAgentSystem(model_name=model_name, vllm_base_url=vllm_base_url, api_key=api_key)

    if os.getenv("INGEST_BNS", "false").lower() == "true":
        agent.ingest_bns(pdf_path=bns_pdf_path)

    q = "What is punishment for theft under Bharatiya Nyaya Sanhita?"
    result = agent.ask(q)
    print("\n--- RESULT ---")
    print(json.dumps(result, indent=2))

    eval_queries = [
        "Hi, summarize this conversation.",
        "What is section 103 in BNS?",
        "Explain negligence in simple words.",
        "Punishment for murder under Bharatiya Nyaya Sanhita",
    ]
    report = agent.evaluate_routing(eval_queries)
    print("\n--- ROUTING EVAL ---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
