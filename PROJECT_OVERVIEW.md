# Intelligent Product Support Assistant — System Architecture & Developer Handoff

> **Document Purpose**: This comprehensive specification is crafted for AI agents and engineering collaborators (such as Claude) to immediately grasp the system architecture, code organization, runtime data flows, design invariants, and strategic roadmap of the **Intelligent Product Support Assistant**.

---

## 1. Executive Summary & Core Objective

The **Intelligent Product Support Assistant** is a local-first, version-aware, Retrieval-Augmented Generation (RAG) system engineered for high-precision technical support of consumer electronics, IoT devices, and computer hardware.

### Key Value Proposition
1. **Zero-Hallucination Hardware Grounding**: Incorrect hardware instructions (e.g., wrong reset button sequences, mismatched voltage instructions, or invalid firmware flashes) can permanently brick customer devices. The system enforces strict citation grounding and escalates gracefully rather than guessing.
2. **Hardware Version & Revision Awareness**: A TP-Link Archer AX21 V1.0 often has completely different firmware, LEDs, and reset behaviors than V2.0 or V3.0. The retriever isolates documents by exact `hardware_version` metadata.
3. **Dual-Tier Evidence Retrieval**:
   - **Tier 1 (Local Vector Store)**: Official manuals, technical specifications, and approved historical interaction memory stored in ChromaDB.
   - **Tier 2 (Live Web Grounding Fallback)**: Google Search Grounding dynamically restricted to verified manufacturer domains when local documentation is absent.
   - **Tier 3 (Safe Escalation)**: If neither source yields verified evidence, the system triggers a safe escalation path providing manufacturer support links.
4. **Self-Improving Verified Memory**: User feedback (Helpful / Not helpful) promotes verified Q&A interactions into a secondary memory vector collection without contaminating immutable official documentation.

---

## 2. Technology Stack

| Layer | Technologies / Libraries |
|---|---|
| **Language & Environment** | Python 3.11+ / 3.12 / 3.14 (macOS/Linux), `pyproject.toml` |
| **Frontend UI** | Streamlit (Custom Dark Charcoal GPT-minimalist design, zero unnecessary chrome) |
| **LLM & Vision** | Google Gemini 2.5 Flash / Flash Lite via the official `google-genai` SDK |
| **Embeddings** | Gemini `text-embedding-004` (Task types: `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) |
| **Vector Store** | ChromaDB (`chromadb.PersistentClient`) with local SQLite persistence |
| **Document Ingestion** | `pypdf`, `pdfplumber`, `pymupdf` (fitz), Markdown/Plaintext splitters |
| **Web Scraping** | `trafilatura`, `BeautifulSoup4`, `urllib.request` |
| **Testing & Evaluation** | `pytest`, `pytest-asyncio`, custom Golden Question evaluation pipeline |

---

## 3. System Architecture & End-to-End Data Flow

```
                                      +-------------------------+
                                      |       User Prompt       |
                                      +-------------------------+
                                                   |
                                                   v
                                      +-------------------------+
                                      |  Conversation Assessor  |
                                      | (rag/gemini.py)         |
                                      +-------------------------+
                                        /                     \
                      [Need Clarification]                  [Ready to Answer]
                                    /                           \
                                   v                             v
                    +----------------------+       +---------------------------+
                    | Ask Clarification Q  |       | Standalone Query Reformer |
                    +----------------------+       +---------------------------+
                                                                 |
                                                                 v
                                                   +---------------------------+
                                                   | ChromaDB Vector Retrieval |
                                                   | - product_docs (Tier 1)   |
                                                   | - approved_memory (Memory)|
                                                   +---------------------------+
                                                                 |
                                                    [Score >= Similarity Threshold?]
                                                    /                             \
                                                 (YES)                            (NO)
                                                  /                                 \
                                                 v                                   v
                             +-----------------------+             +---------------------------+
                             | Strict Grounded Gen   |             | Google Search Grounding   |
                             | (Local Evidence Only) |             | (Manufacturer Portals)    |
                             +-----------------------+             +---------------------------+
                                         |                                       |
                               [NOT_FOUND or Success?]                 [NOT_FOUND or Success?]
                                  /            \                               /          \
                              (Fail)         (Pass)                        (Fail)        (Pass)
                                /                \                          /              \
                               v                  v                        v                v
                 +-------------------+  +-------------------+  +-------------------+  +-------------------+
                 | Safe Escalation   |  | Synthesize Answer |  | Safe Escalation   |  | Synthesize Answer |
                 | + Support Portal  |  | + Doc Citations   |  | + Support Portal  |  | + Web Citations   |
                 +-------------------+  +-------------------+  +-------------------+  +-------------------+
                                                  \                                         /
                                                   \-------------------┬-------------------/
                                                                       |
                                                                       v
                                                       +-------------------------------+
                                                       | User Feedback Capture         |
                                                       | (Helpful / Not Helpful)       |
                                                       +-------------------------------+
                                                                       |
                                                                  [Helpful?]
                                                                  /        \
                                                               (YES)       (NO)
                                                                /            \
                                                               v              v
                                                +--------------------+  +--------------------+
                                                | Save to ChromaDB   |  | Log to Analytics   |
                                                | `approved_memory`  |  | for Prompt Review  |
                                                +--------------------+  +--------------------+
```

---

## 4. Codebase Anatomy

```
Intelligent Product Support Assistant/
├── app.py                      # Streamlit GPT-minimalist chat UI, session state, auto-ingestion
├── pyproject.toml              # Build dependencies & project configuration
├── README.md                   # User documentation and setup guide
├── PROJECT_OVERVIEW.md         # This technical specification for AI agent handoff
│
├── rag/                        # Core RAG Library
│   ├── __init__.py
│   ├── config.py               # Environment configs, paths, similarity thresholds, model names
│   ├── models.py               # Dataclasses: ProductEntity, RetrievedChunk, Citation, ChatResult
│   ├── gemini.py               # Gemini SDK wrapper, prompts, embeddings, search grounding, rate-limit resilience
│   ├── retriever.py            # ChromaDB query engine with metadata filtering ($and: product_id, hw_ver)
│   ├── indexer.py              # Embedding generation & ChromaDB collection management
│   ├── loader.py               # Document loaders (PDF, TXT, MD) & text chunkers
│   ├── sources.py              # Web fetcher (Trafilatura/BS4) and upload storage manager
│   ├── interactions.py         # Logging to interactions.jsonl & promotion of helpful Q&As to memory
│   ├── chatbot.py              # ProductAssistant orchestrator (assesses, retrieves, fallbacks, escalates)
│   └── ocr.py                  # Vision & OCR utilities for hardware labels and port diagrams
│
├── data/                       # Local data persistence
│   ├── chats.json              # Persistent conversation sessions surviving app restarts
│   ├── interactions.jsonl      # Audit trail of questions, answers, citations, and user feedback
│   ├── products/               # Product manuals categorized by slugified product_id
│   └── examples/               # Reference device documentation (e.g. TP-Link Archer AX21)
│
├── chroma_store/               # ChromaDB SQLite and vector binary storage directory
│
├── evaluation/                 # Automated accuracy & safety evaluation
│   ├── golden_questions.csv    # Benchmark dataset with expected answers vs escalation checks
│   └── run_eval.py             # Evaluation runner asserting pass/fail against live LLM responses
│
└── tests/                      # Pytest unit & integration test suite
    ├── test_assistant.py
    ├── test_indexer.py
    ├── test_interactions.py
    └── test_loader.py
```

---

## 5. Detailed Component Breakdown

### 5.1 `rag/chatbot.py` (`ProductAssistant`)
The central orchestrator responsible for executing the RAG state machine:
1. **Conversation Assessment**: Analyzes message history via `assess_conversation()` in `rag/gemini.py`. If details are missing or ambiguous, it asks a targeted clarification question before retrieving.
2. **Query Formulation**: Constructs a standalone retrieval query incorporating the product name, hardware version, and conversation history.
3. **Local Vector Search**: Queries the `product_docs` collection in ChromaDB. If `hardware_version` is specified, it uses strict `$and` filtering.
4. **Relevance Threshold Gating**: Evaluates `local_chunks[0].score >= MIN_DOCUMENT_SIMILARITY` (default `0.65`). If satisfied, prompts Gemini with strict grounding. If the output returns `NOT_FOUND:`, it triggers safe escalation.
5. **Google Search Grounding Fallback**: If local similarity is below threshold, calls `search_grounded_answer()` with Gemini's native Google Search grounding tool, filtering citations.
6. **Graceful Quota Handling**: Catches 429 quota exhaustion errors during search grounding and safely escalates to official manufacturer support instead of crashing.

### 5.2 `rag/indexer.py` & `rag/retriever.py`
- Manages two distinct ChromaDB collections:
  1. `product_documents`: Immutable official documentation chunks.
  2. `approved_interactions`: User-validated historical answers promoted through positive feedback.
- Uses `gemini.embed_texts()` with explicit `task_type="RETRIEVAL_DOCUMENT"` for indexing and `task_type="RETRIEVAL_QUERY"` for querying.

### 5.3 `rag/interactions.py`
- Records every query, response, citation list, escalation status, and feedback to `data/interactions.jsonl`.
- `submit_feedback(interaction_id, helpful=True)`: Validates that the answer was not an escalation, contains valid citations, and stores the verified (Question, Answer, Source URLs) tuple into `approved_interactions` in ChromaDB.

### 5.4 `app.py` (Streamlit Frontend)
- **GPT-Minimalist Dark Charcoal Theme**: Custom CSS palette (`#16181D` background, `#0F1014` sidebar, `#1E2129` assistant card container, `#2D323E` right-aligned user bubble).
- **Persistent Chat History**: Stores active and past chats in `data/chats.json`.
- **Sidebar Organization**: Unified chat rows with left-aligned titles and compact 3-dot popovers (`⋮`) for archiving or deleting chats.
- **Seamless Attachment**: "+ Add documentation" popover accepts PDF/TXT/MD files or documentation URLs and indexes them automatically when the user sends a prompt.
- **Copy on Hover**: Embedded HTML/JS copy button on hover for all chat bubbles.

---

## 6. Critical Invariants & Rules for Future Work

When designing features or proposing code modifications, ensure adherence to these core principles:

1. **Hardware Safety First**: Never allow the model to guess pinouts, reset holding times, voltage ratings, or firmware upgrade instructions. If uncertain, escalate.
2. **Memory Isolation**: User memory (`approved_interactions`) must never overwrite or delete official documentation (`product_documents`). Memory serves solely as secondary context.
3. **No Unbounded Test Runs**: Do not execute live LLM evaluation scripts (`evaluation/run_eval.py`) during basic CSS/frontend iterations; run fast syntax/unit checks instead.
4. **Environment Conventions**: macOS environment using `python3` (not `python`), with API keys loaded via `python-dotenv` from `.env`.

---

## 7. Strategic Roadmap & Gaps to Advise On

We invite **Claude** to analyze this architecture and provide architectural recommendations and implementation strategies for the following priorities:

### Priority 1: Continuous Learning with Attack/Poisoning Defense
* **Current State**: Feedback logs to `interactions.jsonl`, and positive feedback adds the Q&A to Chroma's `approved_interactions`.
* **Challenge**: An attacker or confused user could upvote a malicious prompt injection, hallucinated instruction, or nonsense conversation, poisoning the retrieval memory for future users.
* **Goal**: Design a multi-stage validation system (e.g., automated LLM judge, semantic consistency check against official docs, citation validity verifier, toxicity/injection scanner) before promoting user chats to permanent memory.

### Priority 2: Multi-Modal Visual Troubleshooting (Gemini 2.5 Vision)
* **Goal**: Enable users to upload photos of their physical hardware (e.g., LED light patterns, damaged ports, model number stickers on back panels).
* **Goal**: Have the agent automatically extract model/hardware revision from the sticker OCR (`rag/ocr.py`), diagnose LED blink error codes, and match against diagram figures in the manual.

### Priority 3: Dynamic Agentic Router vs Sequential Fallback
* **Current State**: Static conditional branching (Local Docs $\to$ Search Grounding $\to$ Escalation).
* **Goal**: Implement a dynamic tool-calling agent (ReAct framework or native Gemini function calling) capable of deciding whether to query internal vector memory, fetch fresh web documentation, inspect image attachments, or ask clarifying questions dynamically.

### Priority 4: Token-by-Token Streaming UI
* **Current State**: `st.chat_message` waits for full response completion under `st.spinner`.
* **Goal**: Implement real-time response streaming using `st.write_stream` and Gemini's `generate_content_stream` to drastically lower perceived latency.

### Priority 5: Advanced RAG Evaluation & Hallucination Guardrails
* **Current State**: Binary pass/fail assertions on a small CSV dataset (`evaluation/golden_questions.csv`).
* **Goal**: Integrate continuous evaluation metrics (Context Precision, Context Recall, Faithfulness, Answer Relevance) using frameworks like Ragas or custom Gemini-as-a-Judge evaluators.

---

## 8. Summary for the Incoming AI Agent

You now have complete context on the codebase structure, design philosophies, data pipelines, and technical stack. Use this document as your baseline when recommending improvements, refactoring modules, or implementing new capabilities.
