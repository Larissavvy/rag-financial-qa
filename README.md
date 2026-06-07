# 💼 Financial Document RAG — Production Q&A Pipeline

> Production-grade Retrieval-Augmented Generation system for financial document Q&A. Built with LangChain + FAISS + GPT-4o, with a full evaluation suite measuring faithfulness, answer relevancy, and context precision.

---

## What It Does

Answers natural language questions over financial documents — fund factsheets, analyst reports, compliance policies, earnings transcripts — with source attribution and zero hallucination tolerance.

```
Q: "What is the expense ratio of the Horizon Growth Fund?"
A: The direct plan expense ratio is 0.42% per annum.
   Sources: horizon_growth_fund_factsheet.txt
   [3 chunks retrieved | 1.2s]
```

---

## Architecture

```
Financial Documents (PDF / TXT)
          │
          ▼
  DocumentIngester
  ├── PyPDFLoader / TextLoader
  └── RecursiveCharacterTextSplitter
      (chunk_size=512, overlap=64)
          │
          ▼
  HuggingFace Embeddings
  (all-MiniLM-L6-v2, local — no API cost)
          │
          ▼
  FAISS Vector Store
  (local index, persisted to disk)
          │
          ▼
  Similarity Retrieval
  (top_k=5, score_threshold=0.3)
          │
          ▼
  GPT-4o (temperature=0)
  + Financial QA Prompt
          │
          ▼
  Answer + Source Attribution
          │
          ▼
  RAGEvaluator
  ├── Faithfulness score
  ├── Answer Relevancy score
  └── Context Precision score
```

---

## Production Design Decisions

| Decision | Why |
|----------|-----|
| `temperature=0` | Financial QA requires deterministic answers — no creative generation |
| Local embeddings (MiniLM) | No embedding API cost, runs offline, fast on CPU |
| `score_threshold=0.3` | Rejects low-relevance chunks — prevents garbage-in-garbage-out answers |
| Chunk overlap 64 tokens | Preserves context across table rows and multi-line financial figures |
| Source attribution | Every answer traces back to the source document — auditable |
| FAISS persisted to disk | Index survives restarts, no rebuild needed on each run |
| Eval logging to JSONL | Every query + scores logged for monitoring and regression testing |

---

## Evaluation Suite

Three metrics evaluated per query — no RAGAs dependency, runs on GPT-4o:

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Is every claim in the answer grounded in retrieved context? | > 0.85 |
| **Answer Relevancy** | Does the answer actually address the question? | > 0.80 |
| **Context Precision** | Are retrieved chunks relevant to the question? | > 0.75 |

```bash
# Run full eval suite
python rag_pipeline.py --docs data/sample_docs/ --eval

# Output
Q: What is the expense ratio of the Horizon Growth Fund direct plan?...
   Faithfulness=0.97  Relevancy=0.95  Precision=0.91

Q: What documents are required for corporate KYC onboarding?...
   Faithfulness=0.93  Relevancy=0.89  Precision=0.84

Mean score across all questions: 0.912
Eval log saved → logs/eval_results.jsonl
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set OpenAI key
export OPENAI_API_KEY="sk-..."

# 3. Generate sample financial documents
python generate_sample_docs.py

# 4. Ask a question (auto-builds FAISS index on first run)
python rag_pipeline.py --docs data/sample_docs/ \
  --query "What is the fund's expense ratio?"

# 5. Interactive mode
python rag_pipeline.py --docs data/sample_docs/

# 6. Run evaluation suite
python rag_pipeline.py --docs data/sample_docs/ --eval

# 7. Force rebuild index (after adding new docs)
python rag_pipeline.py --docs data/sample_docs/ --rebuild
```

---

## Repo Structure

```
rag-financial-qa/
│
├── rag_pipeline.py             ← main pipeline (ingest → embed → retrieve → generate → eval)
├── generate_sample_docs.py     ← creates demo financial documents + eval questions
├── requirements.txt
├── README.md
│
├── data/
│   ├── sample_docs/            ← put your PDFs/TXTs here
│   │   ├── fund_factsheet.txt
│   │   ├── analyst_report.txt
│   │   └── compliance_policy.txt
│   └── eval_questions.json     ← Q&A pairs for evaluation
│
├── faiss_index/                ← auto-generated, gitignored
│   ├── index.faiss
│   └── index.pkl
│
└── logs/
    └── eval_results.jsonl      ← per-query eval scores, gitignored
```

---

## Supported Document Types

| Type | Format | Examples |
|------|--------|---------|
| Fund factsheets | PDF, TXT | AUM, expense ratio, holdings, returns |
| Analyst reports | PDF, TXT | Target price, EPS estimates, ratings |
| Compliance docs | PDF, TXT | AML policies, KYC requirements |
| Earnings transcripts | TXT | Management commentary, guidance |

---

## Extending to Production

**Swap FAISS → Pinecone** for multi-user, cloud-hosted retrieval:
```python
from langchain_pinecone import PineconeVectorStore
store = PineconeVectorStore.from_documents(chunks, embeddings, index_name="financial-qa")
```

**Add streaming** for real-time advisor UX:
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
```

**Add re-ranking** for higher precision:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
```

---

## Tech Stack

`LangChain` · `FAISS` · `HuggingFace sentence-transformers` · `GPT-4o` · `PyPDF` · `Python 3.11+`
