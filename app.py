from __future__ import annotations

import html
import importlib
import json
import sys
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
import streamlit as st

# Ensure modified submodules are reloaded in Streamlit's long-running process
for mod_name in list(sys.modules.keys()):
    if mod_name == "rag" or mod_name.startswith("rag."):
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

from rag.chatbot import ProductAssistant
from rag.config import CHROMA_DIR, DATA_DIR, FAQ_COLLECTION, FAQ_DIR, slugify
from rag.gemini import GeminiUnavailable
from rag.indexer import ProductIndex
from rag.interactions import record_interaction, submit_feedback
from rag.loader import load_file
from rag.sources import fetch_webpage, product_directory, save_upload, source_urls

CHATS_FILE = DATA_DIR / "chats.json"

st.set_page_config(
    page_title="Product Support",
    layout="centered",
    initial_sidebar_state="expanded",
)


# --- Custom Dark Gray Charcoal UI ---
def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        /* Base typography & dark gray charcoal palette */
        html, body, [class*="css"], .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #16181D !important;
            color: #EDEDED !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6, p, label, li, 
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] div {
            color: #EDEDED !important;
        }
        small, .caption, [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {
            color: #8E95A5 !important;
        }

        /* Header & Chrome */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        footer {
            display: none !important;
        }

        /* Sidebar toggle control */
        [data-testid="stSidebarCollapsedControl"] {
            background: #1E2129 !important;
            border: 1px solid #2C303B !important;
            border-radius: 6px !important;
            padding: 4px 6px !important;
            top: 10px !important;
            left: 10px !important;
            color: #8E95A5 !important;
        }
        [data-testid="stSidebarCollapsedControl"]:hover {
            color: #EDEDED !important;
            border-color: #3B4252 !important;
        }

        /* Sidebar container */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            background-color: #0F1014 !important;
            color: #EDEDED !important;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid #22252C !important;
            padding-top: 0.5rem !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.75rem 0 !important;
            border-color: #22252C !important;
        }
        section[data-testid="stSidebar"] * {
            color: #EDEDED !important;
        }

        /* Fixed container width */
        .block-container {
            max-width: 680px !important;
            padding-top: 1.5rem !important;
            padding-bottom: 6rem !important;
        }

        /* Chat Message Layout: Right for User, Left for Assistant */
        [data-testid="stChatMessage"] {
            padding: 0.85rem 1.15rem !important;
            margin-bottom: 0.85rem !important;
            position: relative !important;
            transition: all 0.15s ease !important;
        }

        /* User Message: Right Aligned, Content-Fitting Bubble, No Avatar */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            margin-left: auto !important;
            margin-right: 0 !important;
            width: fit-content !important;
            max-width: 75% !important;
            min-width: 48px !important;
            background-color: #2D323E !important;
            border: 1px solid #3B4252 !important;
            border-radius: 16px 16px 3px 16px !important;
            padding: 0.55rem 0.95rem !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) *,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span {
            color: #FFFFFF !important;
        }
        [data-testid="stChatMessageAvatarUser"] {
            display: none !important;
        }

        /* Assistant Message: Left Aligned, Enclosing Box Card */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            margin-right: auto !important;
            margin-left: 0 !important;
            width: 100% !important;
            max-width: 90% !important;
            background-color: #1E2129 !important;
            border: 1px solid #2C303B !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) *,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) span,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div {
            color: #EDEDED !important;
        }
        [data-testid="stChatMessageAvatarAssistant"] {
            background-color: #1E2129 !important;
            border: 1px solid #2C303B !important;
            border-radius: 6px !important;
        }

        /* Copy Button Revealed on Hover at Bottom of Message */
        .copy-bar {
            display: flex;
            justify-content: flex-end;
            margin-top: 6px;
            opacity: 0;
            transition: opacity 0.15s ease;
        }
        [data-testid="stChatMessage"]:hover .copy-bar {
            opacity: 1;
        }
        .copy-btn {
            background: transparent;
            border: 1px solid #2C303B;
            color: #8E95A5 !important;
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.12s ease;
        }
        .copy-btn:hover {
            color: #EDEDED !important;
            border-color: #3B4252;
            background: #1E2129;
        }

        /* Inputs & Textareas */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div {
            background-color: #1E2129 !important;
            color: #EDEDED !important;
            border: 1px solid #303542 !important;
            border-radius: 6px !important;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #EDEDED !important;
            box-shadow: none !important;
        }

        /* Standard buttons */
        .stButton > button {
            border-radius: 6px !important;
            font-size: 0.83rem !important;
            font-weight: 500 !important;
            border: 1px solid #2C303B !important;
            background-color: #1E2129 !important;
            color: #EDEDED !important;
            transition: all 0.12s ease !important;
            white-space: nowrap !important;
            word-break: keep-all !important;
            padding: 0.35rem 0.65rem !important;
        }
        .stButton > button:hover {
            border-color: #3B4252 !important;
        }

        /* Primary New Chat button */
        .stButton > button[kind="primary"] {
            background-color: #EDEDED !important;
            color: #16181D !important;
            border: 1px solid #EDEDED !important;
            font-weight: 600 !important;
            text-align: center !important;
        }
        .stButton > button[kind="primary"] *,
        .stButton > button[kind="primary"] p {
            color: #16181D !important;
        }
        .stButton > button[kind="primary"]:hover {
            opacity: 0.9 !important;
        }

        /* Popover button in main area */
        div[data-testid="stPopover"] > button {
            background-color: #1E2129 !important;
            color: #EDEDED !important;
            border: 1px solid #2C303B !important;
            border-radius: 18px !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            padding: 0.35rem 0.9rem !important;
            text-align: center !important;
            box-shadow: none !important;
            transition: all 0.12s ease !important;
        }
        div[data-testid="stPopover"] > button:hover {
            border-color: #3B4252 !important;
        }

        /* Complete removal of caret / arrow / chevron from popovers */
        [data-testid="stPopover"] button svg,
        [data-testid="stPopover"] button i,
        [data-testid="stPopover"] button [data-testid="stIcon"],
        [data-testid="stPopover"] button [data-testid="stIconMaterial"],
        [data-testid="stPopover"] button span:has(svg),
        [data-testid="stPopover"] button span:last-child:not(:first-child),
        section[data-testid="stSidebar"] [data-testid="stPopover"] svg,
        section[data-testid="stSidebar"] [data-testid="stPopover"] span:has(svg) {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            font-size: 0 !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }

        /* Sidebar Chat Item Row: 1 Unified Box */
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0 !important;
            background-color: #1E2129 !important;
            border: 1px solid #2C303B !important;
            border-radius: 6px !important;
            margin-bottom: 0.35rem !important;
            padding: 0 4px 0 8px !important;
            transition: border-color 0.12s ease, background-color 0.12s ease !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover {
            border-color: #3B4252 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) {
            border-color: #3B4252 !important;
            background-color: #2D323E !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) * {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child {
            flex-grow: 1 !important;
            min-width: 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child {
            flex-shrink: 0 !important;
            width: auto !important;
            margin-left: auto !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button {
            border: none !important;
            background: transparent !important;
            padding: 0.4rem 0.1rem !important;
            font-size: 0.83rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
            width: 100% !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button:hover {
            background: transparent !important;
        }

        /* Borderless, flat 3 dots popover trigger */
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div[data-testid="stPopover"],
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button,
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div[data-testid="stPopover"] button {
            border: none !important;
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
            padding: 0 4px !important;
            margin: 0 !important;
            min-height: unset !important;
            height: auto !important;
            border-radius: 0 !important;
        }

        /* Sidebar Popover Dropdown (Archive/Delete): Ultra-Compact */
        section[data-testid="stSidebar"] div[data-testid="stPopoverBody"] {
            background-color: #0F1014 !important;
            border: 1px solid #2C303B !important;
            border-radius: 4px !important;
            padding: 2px !important;
            margin: 0 !important;
            min-width: unset !important;
            width: 100px !important;
            max-width: 105px !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stPopoverBody"] .stButton {
            margin: 0 !important;
            padding: 0 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stPopoverBody"] .stButton > button {
            border: none !important;
            background: transparent !important;
            color: #8E95A5 !important;
            padding: 2px 4px !important;
            margin: 0 !important;
            min-height: unset !important;
            height: auto !important;
            font-size: 0.72rem !important;
            line-height: 1.1 !important;
            font-weight: 400 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            width: 100% !important;
            border-radius: 3px !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stPopoverBody"] .stButton > button:hover {
            color: #EDEDED !important;
            background-color: #1E2129 !important;
        }

        /* Main Area Popover (Add Documentation): Generous Sizing */
        div[data-testid="stPopoverBody"]:not(section[data-testid="stSidebar"] div[data-testid="stPopoverBody"]) {
            background-color: #1E2129 !important;
            border: 1px solid #2C303B !important;
            border-radius: 12px !important;
            padding: 0.85rem !important;
            min-width: 340px !important;
            max-width: 460px !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
        }

        /* File Uploader Theming */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] section {
            background-color: #16181D !important;
            border: 1px dashed #2C303B !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"] button {
            background-color: #1E2129 !important;
            border: 1px solid #2C303B !important;
            color: #EDEDED !important;
        }

        /* Understated citations */
        .citation-item {
            display: inline-block;
            background: #1E2129;
            border: 1px solid #2C303B;
            padding: 2px 7px;
            border-radius: 4px;
            margin: 2px 4px 2px 0;
            font-size: 0.78rem;
            color: #8E95A5 !important;
        }
        .citation-item a {
            color: #EDEDED !important;
            text-decoration: underline;
        }

        /* Bottom Chat Input Bar */
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {
            background: #16181D !important;
            background-color: #16181D !important;
        }
        [data-testid="stBottom"] > div {
            max-width: 680px !important;
            width: 100% !important;
            margin: 0 auto !important;
            padding-bottom: 1.25rem !important;
        }

        /* Boxless chat input bar */
        [data-testid="stChatInput"] {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        [data-testid="stChatInput"] > div {
            background-color: #1E2129 !important;
            border: 1px solid #303542 !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] > div:focus-within {
            border-color: #EDEDED !important;
        }
        [data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: #EDEDED !important;
            border: none !important;
            resize: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Storage Helpers ---
def load_conversations() -> list[dict]:
    if not CHATS_FILE.exists():
        return []
    try:
        with CHATS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return [c for c in data if isinstance(c, dict) and c.get("messages")]
    except Exception:
        return []


def save_conversations(chats: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CHATS_FILE.open("w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)


# --- Assistant Resource ---
@st.cache_resource
def get_assistant() -> ProductAssistant:
    return ProductAssistant()


@st.cache_resource
def ensure_faq_index() -> bool:
    """Index general support FAQs into ChromaDB if the collection is empty."""
    faq_path = FAQ_DIR / "general_support_faqs.jsonl"
    if not faq_path.exists():
        return False
    try:
        import chromadb
        from chromadb.config import Settings as _Settings
        client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=_Settings(anonymized_telemetry=False))
        try:
            col = client.get_collection(FAQ_COLLECTION)
            if col.count() > 0:
                return True  # Already indexed
        except Exception:
            pass
        return ProductIndex().index_faq_pairs(faq_path) > 0
    except Exception:
        return False


def rebuild_product_index(product_name: str) -> int:
    product_id = slugify(product_name)
    directory = product_directory(product_id)
    urls = source_urls(product_id)
    files = [path for path in directory.iterdir() if path.suffix.lower() in {".pdf", ".txt", ".md"}]
    chunks = [chunk for path in files for chunk in load_file(path, product_id, urls.get(path.name, ""))]
    return ProductIndex().index_documents(chunks, product_id)


# --- Session State Setup ---
inject_custom_styles()
ensure_faq_index()

if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()

if "current_chat" not in st.session_state:
    st.session_state.current_chat = {
        "id": str(uuid.uuid4())[:8],
        "title": "New Chat",
        "active_product": None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [],
    }

current_chat = st.session_state.current_chat


# --- Sidebar ---
with st.sidebar:
    st.markdown("<div style='font-size: 0.9rem; font-weight: 600; margin-bottom: 0.6rem;'>Product Support</div>", unsafe_allow_html=True)

    # New Chat Button
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        if current_chat["messages"]:
            st.session_state.current_chat = {
                "id": str(uuid.uuid4())[:8],
                "title": "New Chat",
                "active_product": None,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": [],
            }
            st.rerun()

    st.markdown("---")

    # Past Conversations List
    active_chats = [c for c in st.session_state.conversations if not c.get("archived")]
    if active_chats:
        st.markdown("<div style='font-size: 0.72rem; font-weight: 600; opacity: 0.6; text-transform: uppercase; margin-bottom: 0.4rem;'>Recent Chats</div>", unsafe_allow_html=True)
        for conv in active_chats:
            is_active = conv["id"] == current_chat["id"]
            title = conv["title"]
            if len(title) > 24:
                title = title[:22] + "..."

            col_btn, col_menu = st.columns([9, 1], gap="small")
            with col_btn:
                btn_type = "primary" if is_active else "secondary"
                if st.button(title, key=f"chat-{conv['id']}", use_container_width=True, type=btn_type):
                    st.session_state.current_chat = conv
                    st.rerun()
            with col_menu:
                with st.popover("⋮", use_container_width=True):
                    if st.button("Archive chat", key=f"arch-{conv['id']}", use_container_width=True):
                        conv["archived"] = True
                        save_conversations(st.session_state.conversations)
                        st.toast("Chat archived")
                        st.rerun()
                    st.markdown("<div style='border-top: 1px solid rgba(128,128,128,0.2); margin: 2px 0;'></div>", unsafe_allow_html=True)
                    if st.button("Delete chat", key=f"del-{conv['id']}", use_container_width=True):
                        st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != conv["id"]]
                        save_conversations(st.session_state.conversations)
                        if is_active:
                            st.session_state.current_chat = {
                                "id": str(uuid.uuid4())[:8],
                                "title": "New Chat",
                                "active_product": None,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "messages": [],
                            }
                        st.rerun()


# --- Main Chat Area ---

# Empty state
if not current_chat["messages"]:
    st.markdown(
        """
        <div style="text-align: center; padding-top: 16vh; padding-bottom: 0.75rem;">
            <h2 style="font-weight: 500; font-size: 1.85rem; letter-spacing: -0.02em; margin-bottom: 0.25rem;">
                What product can I help you with?
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean documentation popup without any fluff
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        with st.popover("+ Add documentation", use_container_width=True):
            st.file_uploader("Upload manual", type=["pdf", "txt", "md"], key="pending_upload", label_visibility="collapsed")
            st.text_input("Documentation URL", placeholder="Paste support / FAQ URL...", key="pending_url", label_visibility="collapsed")

else:
    # Render Conversation Messages (User right, Model left in box)
    for idx, msg in enumerate(current_chat["messages"]):
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            if role == "assistant":
                if msg.get("escalated"):
                    st.caption("No verified documentation was found for this specific hardware.")
                if msg.get("used_search"):
                    st.caption("Researched using Google Search grounding.")

                citations = msg.get("citations", [])
                if citations:
                    st.markdown(
                        "<div style='margin-top: 6px; font-size: 0.78rem; font-weight:600; text-transform:uppercase;'>Sources:</div>",
                        unsafe_allow_html=True,
                    )
                    tags_html = ""
                    for cite in citations:
                        title = cite.get("title") or "Documentation"
                        sec = cite.get("section")
                        lbl = f"{title} ({sec})" if sec else title
                        url = cite.get("url")
                        if url:
                            tags_html += f"<div class='citation-item'><a href='{url}' target='_blank'>{lbl}</a></div>"
                        else:
                            tags_html += f"<div class='citation-item'>{lbl}</div>"
                    st.markdown(tags_html, unsafe_allow_html=True)

                # Inline feedback actions
                interaction_id = msg.get("interaction_id")
                if interaction_id and not msg.get("feedback_submitted"):
                    col_fb, _ = st.columns([3.5, 6.5])
                    with col_fb:
                        fb1, fb2 = st.columns(2, gap="small")
                        with fb1:
                            if st.button("Helpful", key=f"fb-pos-{idx}", use_container_width=True):
                                submit_feedback(interaction_id, helpful=True)
                                msg["feedback_submitted"] = "helpful"
                                save_conversations(st.session_state.conversations)
                                st.toast("Saved to verified memory")
                                st.rerun()
                        with fb2:
                            if st.button("Not helpful", key=f"fb-neg-{idx}", use_container_width=True):
                                submit_feedback(interaction_id, helpful=False)
                                msg["feedback_submitted"] = "not_helpful"
                                save_conversations(st.session_state.conversations)
                                st.toast("Logged for review")
                                st.rerun()

            # Copy Button on Hover (HTML clipboard copy)
            escaped_text = urllib.parse.quote(msg["content"])
            st.markdown(
                f"""
                <div class="copy-bar">
                    <button class="copy-btn" onclick="navigator.clipboard.writeText(decodeURIComponent('{escaped_text}')); this.innerText='Copied'; setTimeout(() => this.innerText='Copy', 1400);">
                        Copy
                    </button>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Chat Input
user_input = st.chat_input("Ask a product question or describe an issue...")
if user_input:
    # Auto-index documentation if uploaded or provided in the popover
    uploaded_doc = st.session_state.get("pending_upload")
    url_doc = (st.session_state.get("pending_url") or "").strip()

    if uploaded_doc or url_doc:
        doc_name = current_chat.get("active_product") or (Path(uploaded_doc.name).stem.replace("-", " ").replace("_", " ").title() if uploaded_doc else "Attached Product")
        pid = slugify(doc_name)
        if uploaded_doc:
            save_upload(pid, uploaded_doc.name, uploaded_doc.getvalue())
        if url_doc:
            try:
                fetch_webpage(pid, url_doc)
            except Exception:
                pass
        try:
            rebuild_product_index(doc_name)
            current_chat["active_product"] = doc_name
        except Exception:
            pass

    current_chat["messages"].append({"role": "user", "content": user_input})
    if current_chat["title"] == "New Chat":
        current_chat["title"] = (user_input[:28] + "...") if len(user_input) > 30 else user_input

    # Ensure current chat is in conversations list
    if not any(c["id"] == current_chat["id"] for c in st.session_state.conversations):
        st.session_state.conversations.insert(0, current_chat)
    save_conversations(st.session_state.conversations)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in current_chat["messages"][:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and preparing response..."):
            try:
                assistant_inst = get_assistant()
                result = assistant_inst.answer(
                    user_input,
                    history=history,
                    active_product=current_chat.get("active_product"),
                )

                product_id = slugify(result.product_name or current_chat.get("active_product") or "general")
                interaction_id = None
                if not result.clarification_needed:
                    interaction_id = record_interaction(product_id, user_input, result)
                    result.interaction_id = interaction_id

                if result.product_name:
                    current_chat["active_product"] = result.product_name

                assistant_msg = {
                    "role": "assistant",
                    "content": result.answer,
                    "citations": [c.__dict__ for c in result.citations],
                    "escalated": result.escalated,
                    "used_search": result.used_search,
                    "interaction_id": interaction_id,
                    "clarification_needed": result.clarification_needed,
                }
                current_chat["messages"].append(assistant_msg)
                save_conversations(st.session_state.conversations)
                st.rerun()

            except GeminiUnavailable as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not complete request: {exc}")
