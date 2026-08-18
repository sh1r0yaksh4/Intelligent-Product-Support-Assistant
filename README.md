# Intelligent Product Support Assistant 🔎

A chat-first, zero-hallucination customer support assistant for consumer products built with **Python**, **Streamlit**, **Gemini (`google-genai` SDK)**, and **Chroma DB**.

---

## 🎯 Core Operating Principles

1. **DOCUMENT-FIRST:** Uploaded manuals, FAQs, and official support URLs are chunked, embedded with Gemini, and searched before consulting external sources.
2. **VERSION-AWARE:** Extracts manufacturer, model, and hardware/revision version (e.g. Archer AX21 V1 vs V2) to prevent cross-version instruction mixing.
3. **SOURCE-GROUNDED:** Answers are strictly synthesized from retrieved evidence with full source citations (title, section, page, URL).
4. **WEB SEARCH FALLBACK:** When local documentation is missing or insufficient (similarity score $< 0.42$), Gemini Google Search Grounding researches official web sources.
5. **SAFE ESCALATION:** If no verifiable source citation can be produced, the assistant explicitly declines to answer and provides official support links instead of fabricating steps.
6. **APPROVED MEMORY:** Customer feedback ("Helpful") promotes verified Q&A interactions into an approved memory store in Chroma DB for secondary retrieval in future similar turns.

---

## 🏗️ Architecture & Module Layout

```text
app.py                       Streamlit multi-tab application & role-based routing
auth/                        Authentication & role security (ADMIN vs CUSTOMER)
database/                    SQLite database manager & table schemas
rag/                         RAG pipeline, Gemini SDK wrapper, vector retriever, OCR
├── config.py                Paths, model settings, similarity thresholds
├── models.py                Dataclasses (DocumentChunk, Citation, ChatResult, ProductEntity)
├── gemini.py                google-genai SDK wrapper (embeddings, generation, JSON extraction)
├── loader.py                PDF, TXT, Markdown reader & chunker with OCR fallback
├── indexer.py               Chroma DB indexer (product_documents & approved_interactions)
├── retriever.py             Version-aware vector similarity search
├── sources.py               File upload & webpage HTML scraper (requests + BeautifulSoup)
├── admin.py                 Admin source review queue & approval workflow (PENDING/APPROVED/REJECTED)
├── chatbot.py               ProductAssistant decision orchestrator
├── interactions.py          DB & JSONL interaction logging + memory promotion
└── ocr.py                   Scanned PDF OCR fallback module
ui/                          Streamlit UI components & tabs
├── components.py            Header, badges, chat bubbles, citations, CSS loader
├── admin_tab.py             Admin review queue, workspace creator, ingestion inputs
├── eval_tab.py              RAG benchmark evaluation dashboard
└── analytics_tab.py         Usage & retrieval performance metrics
evaluation/                  Golden question benchmark dataset & evaluator
scripts/reindex.py           CLI script to rebuild product vector indexes
tests/                       Pytest test suite (loader, db, retriever, chatbot)
static/style.css             Custom dark-mode glassmorphic CSS theme
```

---

## 🚀 Quickstart & Setup

### 1. Environment Setup

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Set your Gemini API key in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Run the Application

```bash
streamlit run app.py
```

Default Login Credentials:
- **Admin Role:** `admin` / `admin123`
- **Customer Role:** `customer` / `customer123`

---

## 🧪 Testing & Evaluation

### Run Pytest Test Suite:
```bash
pytest
```

### Run Benchmark Evaluation CLI:
```bash
python evaluation/run_eval.py
```

### Reindex Product via CLI:
```bash
python scripts/reindex.py "TP-Link Archer AX21" --manufacturer "TP-Link" --model "AX21" --version "V2"
```

---

## 🎬 Hackathon Judge Demo Flow

1. Log in as **Admin** (`admin` / `admin123`).
2. Open **⚙️ Admin Source Review** tab $\rightarrow$ Create product workspace: `TP-Link Archer AX21` (Model: `AX21`, Hardware Version: `V2`).
3. Upload an official PDF/TXT manual or paste an official support URL.
4. Source immediately enters the **PENDING** approval queue.
5. Click **✅ Approve & Index** — system chunks, embeds, and indexes the document into Chroma DB.
6. Switch to **💬 Customer Chat** tab.
7. Ask: *"How do I reset my TP-Link Archer AX21 router?"*
8. Assistant retrieves local manual chunks, synthesizes grounded instructions, displays source citations (title, section, page), and shows the **VERIFIED DOC GROUNDED** badge.
9. Click **👍 Helpful** — interaction is promoted to approved historical memory.
10. Ask a question not in the local manual: *"What is the warranty policy for TP-Link AX21?"*
11. System automatically falls back to **GOOGLE SEARCH GROUNDED** and provides official web citations.
12. Ask an unanswerable / unsupported question: *"Does it support satellite orbital messaging?"*
13. System refuses to guess and displays an explicit **ESCALATION** warning with official manufacturer support links.
14. Open **📊 Benchmark Evaluation** tab $\rightarrow$ Click **Run Live Benchmark Evaluation** to display grounding rates, citation rates, and escalation accuracy.
15. Open **📈 System Analytics** tab to view retrieval tier breakdowns and feedback ratios.
