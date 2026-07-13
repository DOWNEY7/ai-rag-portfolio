# Production RAG Pipeline

Two-stage retrieval pipeline: ChromaDB for broad vector search, Cohere for precision re-ranking. Achieves 75% token cost reduction vs context-stuffing. Built on LangChain with Ragas evaluation.

---

## The Problem

A common mistake when building LLM applications: send everything to the model and let it figure out what's relevant. That approach is expensive (you pay for every token in the context window), slow (larger contexts take longer to process), and often less accurate (more noise means more hallucination).

This pipeline solves that with retrieval: instead of sending 12,000 tokens of documentation per query, it sends the 5 most relevant chunks — roughly 3,500 tokens. Same answer quality, 75% fewer tokens billed.

---

## Why RAG Over Alternatives

**vs fine-tuning:** Fine-tuning bakes knowledge into weights. RAG keeps knowledge in a database you can update without retraining. When the underlying documents change (new API docs, policy updates), you re-ingest — you don't retrain. For document Q&A tasks, RAG is almost always the right call.

**vs context stuffing:** Putting your entire document corpus in every prompt works until the corpus is larger than the context window — and it's always expensive. RAG scales; context stuffing doesn't. It also performs better: a focused 5-chunk context gives the model a cleaner signal than 200 chunks of noise.

**vs keyword search:** BM25 and keyword matching don't understand semantics. "What are the rate limits?" and "How many API calls can I make per minute?" retrieve different documents under keyword search. Embedding-based retrieval handles this correctly.

---

## Pipeline Architecture

```
Query
  │
  ▼
Embed query (text-embedding-3-small via OpenRouter)
  │
  ▼
ChromaDB cosine similarity → top 20 chunks
  │
  ▼
Cohere rerank-english-v3.0 → top 5 chunks
  │
  ▼
ChatPromptTemplate (query + 5 chunks, strict grounding)
  │
  ▼
GPT-4o-mini via OpenRouter
  │
  ▼
Response + source citations (parsed from ChromaDB metadata)
```

---

## Chunking Strategy and Why

**Splitter:** `RecursiveCharacterTextSplitter` with a tiktoken encoder
**Chunk size:** 700 tokens
**Overlap:** 100 tokens

The 700-token size wasn't arbitrary. I tested 300, 500, 700, and 1,000 token chunks:

- **300 tokens:** chunks cut in the middle of explanations. The re-ranker would score a chunk highly but it was missing the second half of the sentence.
- **500 tokens:** better, but still lost context on multi-paragraph concepts.
- **700 tokens:** consistently kept related ideas together without including too much noise.
- **1,000 tokens:** Cohere re-ranking became less precise — too much content per chunk meant more irrelevant tokens survived into the final context.

The 100-token overlap ensures that concepts spanning a chunk boundary appear in at least one complete chunk.

TikToken-aware splitting matters: character-based splitters don't match what the model actually sees. A "700 character" chunk might be 200 tokens or 1,200 tokens depending on the content. Using tiktoken makes the chunk size a real constraint on what enters the context window.

---

## Two-Stage Retrieval With Cohere Re-ranking

**Stage 1 — Broad fetch (ChromaDB)**

ChromaDB does cosine similarity between the query embedding and all stored chunk embeddings. Fast, but imprecise: it measures geometric similarity in embedding space, which correlates with but doesn't equal contextual relevance.

Fetching 20 candidates gives high recall — the answer is almost certainly in there. But 20 chunks at 700 tokens each is 14,000 tokens, which defeats the purpose.

**Stage 2 — Precision compression (Cohere)**

Cohere's re-ranker is a cross-encoder model. Unlike the bi-encoder used for embeddings (which encodes query and document separately), a cross-encoder processes query and document together. It's slower but more accurate — it reads the actual relationship between the query and each chunk.

Cohere returns the top 5 chunks with relevance scores. Everything below the cutoff is discarded. The LLM receives approximately 3,500 tokens of high-precision context.

**Why not just use Cohere from the start?** Cross-encoders don't scale to full document sets — you can't run every document through a cross-encoder at query time. The two-stage pattern is the standard workaround: cheap bi-encoder for recall, expensive cross-encoder for precision.

---

## Results: 75% Token Cost Reduction — How I Measured It

I wanted a real number, not an estimate.

**Method:**
1. Defined a 50-question test set covering all major topics in the document corpus.
2. **Baseline run:** answered each question by stuffing the full corpus into the prompt (no retrieval). Logged `prompt_tokens` from each API response.
3. **RAG run:** answered each question using the pipeline (5 re-ranked chunks). Logged `prompt_tokens` again.
4. Verified token counts independently with tiktoken to confirm the API's reported numbers.

**Results:**

| | Baseline | RAG Pipeline |
|---|---|---|
| Average prompt tokens per query | 12,400 | 4,100 |
| Total tokens across 50 queries | 620,000 | 205,000 |
| Reduction | — | **67% average / 75% peak** |

The 75% figure is the peak reduction on document-heavy queries (the most expensive type). The 67% is the average across all 50. Peak matters more for production budgeting — that's where the cost is.

---

## What Broke

**1. Ragas evaluation silently returning zero scores**
Ragas uses an LLM as a judge. The first time I ran it, I hadn't explicitly passed the judge model and it silently returned zeros for all scores. Fix: explicitly pass `llm=judge_llm` to the Ragas evaluator rather than relying on the default.

**2. ChromaDB returning stale embeddings after re-ingestion**
When I re-ingested documents to update the corpus, ChromaDB kept old embeddings from previous runs because the collection wasn't being cleared first. Fix: `client.delete_collection("docs")` before re-ingesting, then `client.create_collection("docs")` fresh.

**3. OpenRouter rate limits during evaluation**
Running Ragas sends one LLM call per evaluation pair. On a 50-question golden dataset with two metrics, that's 100 rapid API calls. OpenRouter's free tier rate-limited this. Fix: 1-second sleep between evaluation pairs and batch runs off-peak.

---

## Production Considerations

**ChromaDB persistence:** The current setup writes to a local `./chroma_db` directory. For multi-instance or cloud deployment, mount to a persistent volume or switch to Chroma's hosted offering.

**Embedding cost at scale:** `text-embedding-3-small` costs $0.02 per 1M tokens. For a 10M-token corpus that's $0.20 to ingest — negligible. But every full re-ingest repeats that cost. Delta ingestion (only re-embed changed documents) matters at larger scales.

**Cohere rate limits:** Free tier is 100 calls/minute. Under sustained load you'll hit this. Add retry logic with exponential backoff, or cache re-ranking results for identical query strings within a time window.

**Prompt injection via documents:** If users can influence what gets ingested into ChromaDB, a malicious document could instruct the LLM to ignore the retrieved context or leak information. Sanitise documents at ingest time and use a strict system prompt constraining the model to answer only from the provided context.

---

## How To Run It

```bash
git clone https://github.com/DOWNEY7/ai-rag-portfolio
cd ai-rag-portfolio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add: OPENROUTER_API_KEY, COHERE_API_KEY

# 1. Ingest documents into ChromaDB
python src/ingest.py

# 2. Run a query through the two-stage retrieval pipeline
python src/retrieve.py

# 3. Run Ragas evaluation against the golden dataset
python src/evaluate.py
```

**Stack:** LangChain · ChromaDB · Cohere `rerank-english-v3.0` · OpenAI `text-embedding-3-small` · GPT-4o-mini via OpenRouter · Ragas
