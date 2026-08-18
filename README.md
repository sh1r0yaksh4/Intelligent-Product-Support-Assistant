# Product Support Assistant

A chat-first, grounded support assistant for arbitrary consumer products. Ask naturally, add manuals or official support URLs when you have them, and the assistant will search those product-specific documents before researching the web.

## Current MVP

The running application provides one chat interface. A user can describe a product and ask a question in the same message, for example: `I have a TP-Link Archer AX21. How do I reset it?`

The app then:

1. Identifies the product name when possible.
2. Searches that product's uploaded/saved documentation first.
3. Uses Gemini Google Search grounding when local documentation does not support an answer.
4. Shows citations for every non-escalated answer.
5. Says it cannot verify an answer instead of inventing one.
6. Records feedback safely; only helpful answers with sources are retained as separate interaction memory.

## Tech stack and responsibilities

| Technology | Used for | Not used for |
|---|---|---|
| Python | All application, ingestion, retrieval, and feedback code | Separate backend service |
| Streamlit | Single chat-first web interface and local document controls | User accounts or production hosting |
| Gemini via `google-genai` | Product identification, natural-language answers, embeddings, and Google Search grounding | Fine-tuning model weights from conversations |
| Chroma | Local persistent vector search, partitioned by product workspace | Primary source of truth; original documents remain authoritative |
| `pypdf` | Extracting readable text from manuals and PDF guides | OCR for scanned/image-only PDFs |
| Beautiful Soup + Requests | Saving user-supplied public documentation URLs as indexable text | Background crawling of arbitrary websites |
| JSONL | Local interaction and feedback log | Multi-user analytics or a production database |
| Pytest | Loader and chunking checks | End-to-end live Gemini API tests |

## Reliability model

- **Supplied product documents come first.** PDFs, TXT, Markdown, and supplied support URLs are chunked, embedded with Gemini, and stored in Chroma by product workspace.
- **Research is a fallback, not a guess.** When local documents do not support an answer, Gemini Google Search grounding researches the question and returns only API-provided citations.
- **No evidence means no answer.** Missing citations, weak local retrieval, or `NOT_FOUND:` produces an explicit escalation message.
- **Feedback does not silently rewrite facts.** Helpful, source-backed interactions are retained in a separate local memory collection. They do not replace official documentation; negative feedback is logged for review only.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GEMINI_API_KEY to .env
streamlit run app.py
```

The app uses the current `google-genai` Python SDK. Gemini Google Search grounding must be available to the supplied API key for live research. If it is unavailable, add a manufacturer manual or official documentation URL in the app instead.

## Use the assistant

1. Start with one message containing the product and question: `I have a Sony WH-1000XM5. How do I pair it with a second device?`
2. The assistant identifies the product and checks its local workspace. If no supporting document is present, it researches official web sources in the same response.
3. Open **Add trusted product documents or a support URL** to upload a manual or paste an official support page. The app immediately rebuilds that product's index.
4. Use **Helpful** only for an answer that is correct and well-cited. These interactions are stored separately as reusable, source-backed memory.

## Repository layout

```text
app.py                 Streamlit chat experience
rag/                   ingestion, Gemini, Chroma, retrieval, feedback logic
data/products/         ignored local product workspaces and uploaded sources
evaluation/            golden-question template
scripts/reindex.py     rebuild a product workspace from the command line
tests/                 loader behavior tests
```

## Components to add after the basic MVP

- **Source review:** show all discovered/uploaded sources and let an admin approve or remove them before indexing.
- **Better product detection:** extract manufacturer, model, and hardware/version separately, then prevent cross-version answers.
- **Evaluation dashboard:** read the golden-question CSV and report grounded-answer, citation, and escalation accuracy.
- **Interaction review queue:** require an admin approval step before helpful interactions are reused, instead of automatic promotion.
- **Multi-user storage:** replace JSONL with a proper database only when user accounts or shared history are needed.
- **Deployment:** package the single Streamlit app after local evaluation is reliable.

## Expected final product

The final demonstrable product is a reusable support chatbot: a customer mentions a product, asks a normal-language question, and receives a concise cited answer grounded in either approved product documentation or current official web research. If evidence is weak, ambiguous, or unavailable, it clearly explains that it cannot verify the answer and directs the user to official support. Over time, the system collects feedback and source-backed interaction patterns without allowing unchecked conversations to become product facts.

## Evaluation

Use `evaluation/golden_questions.csv` as the initial test log. Build a set for each demo product with direct manual questions, noisy/colloquial phrasing, follow-ups, and deliberately unanswerable questions. A passing answer needs a factual response and a real source citation. A passing escalation must avoid invented advice.

## Scope boundaries

This is a local prototype. It intentionally excludes accounts, multi-user permissions, automatic fine-tuning, background web crawling, persistent cloud deployment, and unreviewed ingestion of forum content. Use public official documentation only where possible.
