from __future__ import annotations

import streamlit as st

from rag.chatbot import ProductAssistant
from rag.config import slugify
from rag.gemini import GeminiUnavailable
from rag.indexer import ProductIndex
from rag.interactions import record_interaction, submit_feedback
from rag.loader import load_file
from rag.sources import fetch_webpage, product_directory, save_upload, source_urls

st.set_page_config(page_title="Product Support Assistant", page_icon="🔎", layout="centered")


@st.cache_resource
def assistant() -> ProductAssistant:
    return ProductAssistant()


def rebuild_product(product_name: str) -> int:
    product_id = slugify(product_name)
    directory = product_directory(product_id)
    urls = source_urls(product_id)
    files = [path for path in directory.iterdir() if path.suffix.lower() in {".pdf", ".txt", ".md"}]
    chunks = [chunk for path in files for chunk in load_file(path, product_id, urls.get(path.name, ""))]
    return ProductIndex().index_documents(chunks, product_id)


def render_answer(message: dict, index: int) -> None:
    with st.chat_message("assistant"):
        st.markdown(message["content"])
        if message.get("escalated"):
            st.caption("No verified answer was available, so the assistant did not guess.")
        if message.get("used_search"):
            st.caption("Researched with Google Search grounding.")
        citations = message.get("citations", [])
        if citations:
            st.markdown("**Sources**")
            for citation in citations:
                title = citation["title"]
                section = citation.get("section")
                label = f"{title} — {section}" if section else title
                url = citation.get("url")
                st.markdown(f"- [{label}]({url})" if url else f"- {label}")
        interaction_id = message.get("interaction_id")
        if interaction_id:
            left, right, _ = st.columns([1, 1, 6])
            if left.button("Helpful", key=f"helpful-{index}"):
                submit_feedback(interaction_id, helpful=True)
                st.toast("Saved as approved interaction memory because it has supporting sources.")
            if right.button("Not helpful", key=f"not-helpful-{index}"):
                submit_feedback(interaction_id, helpful=False)
                st.toast("Saved for review; it will not be reused as product knowledge.")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_product" not in st.session_state:
    st.session_state.active_product = None

st.title("Product Support Assistant")
st.caption("Ask naturally. The assistant checks supplied documents first, then researches official sources when needed. It escalates rather than guessing.")

with st.expander("Add trusted product documents or a support URL"):
    product_name = st.text_input(
        "Product name and model",
        value=st.session_state.active_product or "",
        placeholder="Example: TP-Link Archer AX21",
        help="Optional for chat. Required only when adding documents to a product workspace.",
    )
    uploaded = st.file_uploader("Manual, FAQ, brochure, or guide", type=["pdf", "txt", "md"])
    source_url = st.text_input("Official support/documentation URL")
    add_files, add_url, reindex = st.columns(3)
    if add_files.button("Add document", disabled=not uploaded):
        if not product_name.strip():
            st.error("Enter a product name before adding a document.")
        else:
            product_id = slugify(product_name)
            save_upload(product_id, uploaded.name, uploaded.getvalue())
            count = rebuild_product(product_name)
            st.session_state.active_product = product_name.strip()
            st.success(f"Document saved. {count} chunks are now indexed for this product.")
    if add_url.button("Add URL", disabled=not source_url):
        if not product_name.strip():
            st.error("Enter a product name before adding a URL.")
        else:
            try:
                fetch_webpage(slugify(product_name), source_url.strip())
                count = rebuild_product(product_name)
                st.session_state.active_product = product_name.strip()
                st.success(f"Official source saved. {count} chunks are now indexed for this product.")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    if reindex.button("Rebuild index", disabled=not product_name):
        try:
            st.success(f"Indexed {rebuild_product(product_name)} chunks.")
        except Exception as exc:
            st.error(f"Could not rebuild index: {exc}")

if st.session_state.active_product:
    st.info(f"Active product workspace: {st.session_state.active_product}")

for index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        render_answer(message, index)

prompt = st.chat_input("Describe the product and ask your question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[:-1]
    ]
    with st.chat_message("assistant"):
        with st.spinner("Checking available documentation and reliable web sources…"):
            try:
                result = assistant().answer(prompt, history, st.session_state.active_product)
                product_id = slugify(result.product_name or st.session_state.active_product or "general")
                interaction_id = record_interaction(product_id, prompt, result)
                result.interaction_id = interaction_id
                if result.product_name:
                    st.session_state.active_product = result.product_name
                message = {
                    "role": "assistant",
                    "content": result.answer,
                    "citations": [citation.__dict__ for citation in result.citations],
                    "escalated": result.escalated,
                    "used_search": result.used_search,
                    "interaction_id": interaction_id,
                }
                st.session_state.messages.append(message)
                render_answer(message, len(st.session_state.messages) - 1)
            except GeminiUnavailable as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error("I could not complete that research request safely. Please try again or add an official manual/URL.")
                st.exception(exc)

