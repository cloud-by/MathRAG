# MathRAG

A RAG-based mathematical question answering prototype system built with **FastAPI + FAISS + Embedding API + DeepSeek API**.

MathRAG is designed as a minimum viable demo for mathematical tutoring scenarios. It retrieves relevant knowledge from a structured math knowledge base, then uses a large language model to generate step-by-step answers, reference knowledge, and related follow-up questions.

## Features

- Mathematical question answering based on RAG
- FAISS-based vector retrieval
- Step-by-step solution generation
- Reference knowledge display
- Related follow-up question recommendation
- Simple multi-turn dialogue support
- FastAPI backend + browser frontend

## Project Structure

```text
MathRAG/
├─ README.md
├─ requirements.txt
├─ .env
├─ run.py
│
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  │
│  ├─ api/
│  │  ├─ __init__.py
│  │  └─ chat.py
│  │
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ config.py
│  │  └─ logger.py
│  │
│  ├─ schemas/
│  │  ├─ __init__.py
│  │  └─ chat.py
│  │
│  ├─ services/
│  │  ├─ __init__.py
│  │  ├─ embedding_service.py
│  │  ├─ vector_store.py
│  │  ├─ retriever.py
│  │  ├─ llm_service.py
│  │  └─ rag_pipeline.py
│  │
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ text_cleaner.py
│  │  ├─ prompt_builder.py
│  │  └─ math_postprocess.py
│  │
│  └─ frontend/
│     ├─ index.html
│     ├─ style.css
│     └─ app.js
│
├─ scripts/
│  ├─ __init__.py
│  ├─ build_kb.py
│  ├─ build_index.py
│  ├─ demo_query.py
│  └─ test_rag.py
│
├─ data/
│  ├─ raw/
│  │  └─ math_knowledge_seed.jsonl
│  ├─ processed/
│  │  └─ kb_chunks.jsonl
│  └─ index/
│     ├─ faiss.index
│     └─ id_map.json
│
└─ tests/
   ├─ __init__.py
   └─ test_chat_api.py
```

## Workflow

MathRAG works in the following steps:

1. Prepare raw math knowledge in JSONL format.
2. Run `build_kb.py` to normalize and convert raw knowledge into retrieval-ready chunks.
3. Run `build_index.py` to generate embeddings and build the FAISS index.
4. When a user asks a question:
   - the system embeds the question,
   - retrieves the most relevant knowledge from FAISS,
   - constructs a prompt with question + history + retrieved knowledge,
   - calls the LLM API to produce a structured answer.
5. The frontend displays:
   - answer,
   - steps,
   - reference knowledge,
   - related follow-up questions.

## Environment Requirements

Recommended:

- Python 3.10 / 3.11 / 3.12
- Windows / Linux / macOS
- Available Embedding API key
- Available DeepSeek API key

> Note: some third-party packages may behave inconsistently on very new Python versions. If you encounter odd dependency issues, Python 3.11 is the safest choice.

## Installation

Create and activate a virtual environment first, then install dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root.

Example:

```env
APP_HOST=127.0.0.1
APP_PORT=8000
APP_DEBUG=true

EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=10
EMBEDDING_TIMEOUT=60
EMBEDDING_NORMALIZE=true

LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-reasoner
LLM_TIMEOUT=120
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.2
LLM_RETURN_REASONING=false

TOP_K=3
USE_INNER_PRODUCT=true
```

## Data Preparation

Place the raw seed knowledge file here:

```text
data/raw/math_knowledge_seed.jsonl
```

Then run the preprocessing script:

```bash
python -m scripts.build_kb
```

After that, build the vector index:

```bash
python -m scripts.build_index
```

If successful, the following files will be generated:

```text
data/processed/kb_chunks.jsonl
data/index/faiss.index
data/index/id_map.json
```

## Retrieval Validation

Run retrieval-only testing:

```bash
python -m scripts.demo_query --question "x^2+4x+3=0 怎么解？" --show-context
```

Interactive retrieval mode:

```bash
python -m scripts.demo_query --interactive --show-context
```

## RAG Validation

Run end-to-end RAG testing:

```bash
python -m scripts.test_rag --question "x^2+4x+3=0 怎么解？" --show-references
```

This verifies the whole chain:

- knowledge retrieval
- prompt construction
- LLM generation
- structured answer output

## Run the Web App

Start the FastAPI service:

```bash
python run.py
```

or:

```bash
uvicorn app.main:app --reload
```

After startup, open:

- Home page: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API Example

### POST `/api/chat`

Request body:

```json
{
  "question": "x^2+4x+3=0 怎么解？",
  "history": [],
  "top_k": 3
}
```

Response example:

```json
{
  "answer": "解为 x=-1 或 x=-3，通过因式分解法求解。",
  "steps": [
    "方程已是标准形式 x^2+4x+3=0。",
    "因式分解为 (x+1)(x+3)=0。",
    "令每个因式等于0，得到 x=-1 或 x=-3。"
  ],
  "used_knowledge": [
    "因式分解法解一元二次方程"
  ],
  "references": [
    {
      "title": "因式分解法解一元二次方程",
      "category": "quadratic_equation",
      "score": 0.687758
    }
  ],
  "related_questions": [
    "如何用求根公式解这道题？",
    "为什么这题可以因式分解？"
  ]
}
```

## Current MVP Scope

This prototype currently focuses on:

- algebra-oriented math QA
- structured seed knowledge retrieval
- Chinese math tutoring scenarios
- simple multi-turn question answering

It is suitable for demo, coursework, and early-stage graduation project development.

## Suggested Next Steps

- expand the knowledge base from 50 items to 200 / 500 / 2000 items
- improve prompt constraints and answer formatting
- add formula rendering support
- optimize frontend interaction and history management
- add reranking for better retrieval quality
- optionally add user logging and evaluation metrics

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

Run scripts from the project root using module mode:

```bash
python -m scripts.build_index
```

### `Error code: 402 - Insufficient Balance`

Your LLM API request reached the DeepSeek server successfully, but the account balance is insufficient. Recharge the API account and retry.

### Frontend page loads but styles are missing

Check whether these files exist:

```text
app/frontend/index.html
app/frontend/style.css
app/frontend/app.js
```

## License

This repository is intended for educational and research prototype use.