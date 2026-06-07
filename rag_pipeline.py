"""
Production-grade RAG pipeline for financial document Q&A.

Architecture:
  - Document ingestion: PDF/text financial documents (fund factsheets,
    analyst reports, compliance docs, earnings transcripts)
  - Chunking: recursive character splitting with overlap
  - Embeddings: HuggingFace sentence-transformers (local, no API cost)
  - Vector store: FAISS (local, no external dependency)
  - LLM: GPT-4o via OpenAI API
  - Evals: faithfulness, answer relevancy, context precision (RAGAs-style)

Usage:
    python rag_pipeline.py --docs data/sample_docs/ --query "What is the fund's expense ratio?"
"""

import os
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.3        

    # Embedding model 
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    llm_model: str = "gpt-4o"
    temperature: float = 0.0            # deterministic
    max_tokens: int = 1024

    # Storage
    index_dir: str = "faiss_index"
    eval_log: str = "logs/eval_results.jsonl"


# ── Prompt ────────────────────────────────────────────────────────────────────

FINANCIAL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a financial analyst assistant. Answer the question using ONLY the provided context.
If the answer cannot be determined from the context, say "I don't have enough information in the provided documents."
Never fabricate financial figures, dates, or entity names.

Context:
{context}

Question: {question}

Answer (be precise and cite the source document where possible):"""
)


# Document Ingestion 

class DocumentIngester:
    """Loads and chunks financial documents."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def load_directory(self, docs_path: str) -> list[Document]:
        path = Path(docs_path)
        documents = []

        for file in path.rglob("*"):
            if file.suffix == ".pdf":
                try:
                    loader = PyPDFLoader(str(file))
                    documents.extend(loader.load())
                    print(f"  ✓ Loaded PDF: {file.name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {file.name}: {e}")

            elif file.suffix in [".txt", ".md"]:
                try:
                    loader = TextLoader(str(file), encoding="utf-8")
                    documents.extend(loader.load())
                    print(f"  ✓ Loaded text: {file.name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {file.name}: {e}")

        print(f"\n  Loaded {len(documents)} raw document pages/sections")
        return documents

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        # Add chunk hash for deduplication
        for chunk in chunks:
            chunk.metadata["chunk_hash"] = hashlib.md5(
                chunk.page_content.encode()
            ).hexdigest()[:8]
        print(f"  Split into {len(chunks)} chunks "
              f"(avg {sum(len(c.page_content) for c in chunks)//len(chunks)} chars)")
        return chunks


# Vector Store

class VectorStoreManager:
    """Manages FAISS index: build, save, load."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def build(self, chunks: list[Document]) -> FAISS:
        print(f"\n  Building FAISS index over {len(chunks)} chunks...")
        t0 = time.time()
        store = FAISS.from_documents(chunks, self.embeddings)
        print(f"  Index built in {time.time()-t0:.1f}s")
        return store

    def save(self, store: FAISS) -> None:
        Path(self.config.index_dir).mkdir(parents=True, exist_ok=True)
        store.save_local(self.config.index_dir)
        print(f"  Index saved → {self.config.index_dir}/")

    def load(self) -> FAISS:
        print(f"  Loading existing index from {self.config.index_dir}/")
        return FAISS.load_local(
            self.config.index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def index_exists(self) -> bool:
        return (Path(self.config.index_dir) / "index.faiss").exists()


# Retrieval Chain

class FinancialRAG:
    """End-to-end RAG chain with source attribution."""

    def __init__(self, store: FAISS, config: RAGConfig):
        self.config = config
        self.retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config.top_k,
                "score_threshold": config.score_threshold,
            },
        )
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": FINANCIAL_QA_PROMPT},
        )

    def query(self, question: str) -> dict:
        t0 = time.time()
        result = self.chain.invoke({"query": question})
        latency = round(time.time() - t0, 2)

        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in result.get("source_documents", [])
        })

        return {
            "question": question,
            "answer": result["result"],
            "sources": sources,
            "num_chunks_retrieved": len(result.get("source_documents", [])),
            "latency_s": latency,
        }


# Evaluation

class RAGEvaluator:
    """
    Lightweight RAGAs-style evaluation without the RAGAs dependency.
    Runs 3 checks per QA pair:
      1. Faithfulness    — is the answer grounded in the retrieved context?
      2. Answer Relevancy — does the answer address the question?
      3. Context Precision — are retrieved chunks actually relevant to the question?
    """

    def __init__(self, llm_model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0, max_tokens=256)

    def _llm_score(self, prompt: str) -> float:
        """Ask LLM to score 0.0–1.0 and parse the float."""
        response = self.llm.invoke(prompt).content.strip()
        try:
            # Extract first float found
            import re
            match = re.search(r"(0?\.\d+|1\.0|0|1)", response)
            return float(match.group()) if match else 0.5
        except Exception:
            return 0.5

    def faithfulness(self, answer: str, context_chunks: list[str]) -> float:
        context = "\n---\n".join(context_chunks[:3])
        prompt = f"""Rate from 0.0 to 1.0 how faithful this answer is to the context.
1.0 = every claim is directly supported. 0.0 = answer contradicts or ignores context.
Context: {context}
Answer: {answer}
Score (just the number):"""
        return self._llm_score(prompt)

    def answer_relevancy(self, question: str, answer: str) -> float:
        prompt = f"""Rate from 0.0 to 1.0 how well this answer addresses the question.
1.0 = directly and completely answers. 0.0 = off-topic or evasive.
Question: {question}
Answer: {answer}
Score (just the number):"""
        return self._llm_score(prompt)

    def context_precision(self, question: str, context_chunks: list[str]) -> float:
        scores = []
        for chunk in context_chunks[:self.config_k if hasattr(self, "config_k") else 3]:
            prompt = f"""Rate from 0.0 to 1.0 how relevant this chunk is to answering the question.
1.0 = directly relevant. 0.0 = completely irrelevant.
Question: {question}
Chunk: {chunk[:400]}
Score (just the number):"""
            scores.append(self._llm_score(prompt))
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def evaluate(self, question: str, answer: str,
                 context_chunks: list[str]) -> dict:
        faith = self.faithfulness(answer, context_chunks)
        relevancy = self.answer_relevancy(question, answer)
        precision = self.context_precision(question, context_chunks)
        return {
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "mean_score": round((faith + relevancy + precision) / 3, 3),
        }

    def log(self, result: dict, eval_scores: dict, log_path: str) -> None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        record = {**result, "eval": eval_scores, "timestamp": time.time()}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# CLI

def build_index(docs_path: str, config: RAGConfig) -> FAISS:
    print("\n── Ingesting documents ─────────────────────────────")
    ingester = DocumentIngester(config)
    docs = ingester.load_directory(docs_path)
    chunks = ingester.chunk(docs)

    print("\n── Building vector index ───────────────────────────")
    vsm = VectorStoreManager(config)
    store = vsm.build(chunks)
    vsm.save(store)
    return store


def main():
    parser = argparse.ArgumentParser(description="Financial RAG Pipeline")
    parser.add_argument("--docs",  default="data/sample_docs/",
                        help="Path to financial documents directory")
    parser.add_argument("--query", default=None,
                        help="Question to ask (interactive mode if omitted)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of FAISS index")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation suite from data/eval_questions.json")
    args = parser.parse_args()

    config = RAGConfig()
    vsm = VectorStoreManager(config)

    # Load or build index
    if args.rebuild or not vsm.index_exists():
        store = build_index(args.docs, config)
    else:
        store = vsm.load()

    rag = FinancialRAG(store, config)

    # Eval mode
    if args.eval:
        eval_path = Path("data/eval_questions.json")
        if not eval_path.exists():
            print("No eval_questions.json found. See data/README.md for format.")
            return

        evaluator = RAGEvaluator()
        questions = json.loads(eval_path.read_text())
        print(f"\n── Running eval on {len(questions)} questions ──────────")

        scores = []
        for item in questions:
            result = rag.query(item["question"])
            chunks = [doc.page_content for doc in
                      rag.retriever.invoke(item["question"])]
            evals = evaluator.evaluate(item["question"], result["answer"], chunks)
            evaluator.log(result, evals, config.eval_log)
            scores.append(evals["mean_score"])
            print(f"  Q: {item['question'][:60]}...")
            print(f"     Faithfulness={evals['faithfulness']:.2f}  "
                  f"Relevancy={evals['answer_relevancy']:.2f}  "
                  f"Precision={evals['context_precision']:.2f}")

        print(f"\n  Mean score across all questions: {sum(scores)/len(scores):.3f}")
        print(f"  Eval log saved → {config.eval_log}")
        return

    # Single query mode
    if args.query:
        result = rag.query(args.query)
        print(f"\nQ: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")
        print(f"Chunks retrieved: {result['num_chunks_retrieved']} | "
              f"Latency: {result['latency_s']}s")
        return

    # Interactive mode
    print("\n── Financial RAG — Interactive Mode ────────────────")
    print("Type your question (or 'quit' to exit)\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue
        result = rag.query(question)
        print(f"\nA: {result['answer']}")
        print(f"   Sources: {', '.join(result['sources'])}")
        print(f"   [{result['num_chunks_retrieved']} chunks | {result['latency_s']}s]\n")


if __name__ == "__main__":
    main()
