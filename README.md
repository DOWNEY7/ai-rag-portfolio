# Production-Grade Anthropic RAG System

## Project Overview
This project is a high-performance **Retrieval-Augmented Generation (RAG) pipeline** designed to query technical documentation (specifically Anthropic's Claude guidelines and Cookbooks) with high accuracy. 

The system is built with a focus on mitigating hallucinations through strict prompting, improving context precision via two-stage Cohere re-ranking, and establishing measurable baseline performance through automated Ragas evaluations. Every generated answer is rigorously backed by transparent, metadata-driven citations.

## Architecture & Tech Stack
- **Framework**: [LangChain](https://python.langchain.com/) (Orchestrating vector operations, chains, and component wrappers)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Local, persistent storage of document embeddings)
- **Embeddings**: `OpenAIEmbeddings` (`text-embedding-3-small`) via [OpenRouter](https://openrouter.ai/)
- **LLM/Generator**: `ChatOpenAI` (`openai/gpt-4o-mini`) via [OpenRouter](https://openrouter.ai/) for scalable and cost-effective generation.
- **Re-ranker**: [Cohere](https://cohere.com/) (`rerank-english-v3.0`) utilized in a `ContextualCompressionRetriever`.
- **Evaluation**: [Ragas](https://docs.ragas.io/) (Automated testing for *Faithfulness* and *Answer Relevancy*).

## Key Features

### 1. Intelligent Data Ingestion & Chunking (`src/ingest.py`)
- Programmatically extracts and parses `.md` and `.pdf` files from the Anthropic cookbooks.
- Implements LangChain's `RecursiveCharacterTextSplitter` configured with a TikToken encoder for token-aware chunking (`chunk_size=700`, `overlap=100`) to preserve context boundaries.

### 2. Two-Stage Retrieval with Cohere Re-ranking (`src/retrieve.py`)
- **Stage 1 (Broad Fetch)**: Queries ChromaDB for the top `k=20` chunks based on standard cosine similarity.
- **Stage 2 (Precision Compression)**: Passes the 20 chunks to Cohere's Re-rank API to contextually score and compress the payload down to the absolute `top_n=5` most relevant chunks.
- **Strict Generation**: Uses a heavily constrained `ChatPromptTemplate` forcing the LLM to answer *only* based on the compressed context and to gracefully fall back if the answer isn't present. Citations are mathematically parsed from document metadata.

### 3. Automated Evaluation Pipeline (`src/evaluate.py`)
- Establishes a "Golden Dataset" of Claude-specific ground-truth Q&A pairs (Tool use, Prompt Caching, System Prompts).
- Programmatically generates answers via the live retrieve pipeline.
- Uses OpenRouter's LLM-as-a-judge to evaluate the pipeline on **Faithfulness** (factual consistency with context) and **Answer Relevancy** (semantic alignment with the query).

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/ai-rag-portfolio.git
cd ai-rag-portfolio
```

**2. Create a Virtual Environment**
Using conda or standard python `venv` (Python 3.11+ recommended):
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Environment Variables**
Copy the example environment file and populate it with your keys:
```bash
cp .env.example .env
```
Inside `.env`, ensure you have:
```env
OPENROUTER_API_KEY=your_openrouter_key
COHERE_API_KEY=your_cohere_key
```
*(Note: standard `OPENAI_API_KEY` can be used as a fallback if not using OpenRouter)*.

---

## Usage

### 1. Build the Vector Database
Populate the local `./chroma_db` database with your corpus files.
```bash
python src/ingest.py
```

### 2. Run the Query Pipeline
Test the two-stage retriever (Chroma + Cohere) against a default query:
```bash
python src/retrieve.py
```
*Output will display the generated answer and the Top 5 source citations.*

### 3. Run Ragas Evaluation
Automatically benchmark the system's performance on the included Golden Dataset:
```bash
python src/evaluate.py
```
*Output will display a pandas dataframe breaking down `faithfulness` and `answer_relevancy` scores.*
