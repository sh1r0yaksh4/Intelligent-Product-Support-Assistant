from __future__ import annotations

import streamlit as st

from database.db import init_db
from rag.admin import list_products
from rag.chatbot import ProductAssistant
from rag.config import slugify
from rag.gemini import GeminiUnavailable
from rag.interactions import record_interaction, submit_feedback
from ui.admin_tab import render_admin_tab
from ui.components import (
    load_css,
    render_hero_banner,
    render_source_card,
    render_standalone_login_screen,
    render_status_message,
    render_suggestion_pills,
    render_top_navbar,
)
from ui.eval_tab import render_eval_tab
from ui.knowledge_tab import render_knowledge_tab

st.set_page_config(page_title="SupportAI — Product Support Assistant", page_icon="⚡", layout="centered")

# Initialize database and default users
init_db()

# Load Custom CSS Theme
load_css()

# Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = "guest"
if "user_role" not in st.session_state:
    st.session_state.user_role = "CUSTOMER"
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_product" not in st.session_state:
    st.session_state.active_product = None
if "active_version" not in st.session_state:
    st.session_state.active_version = None


@st.cache_resource
def assistant() -> ProductAssistant:
    return ProductAssistant()


# Standalone Login Page Handler
if st.session_state.current_page == "login" and not st.session_state.authenticated:
    render_standalone_login_screen()
    st.stop()

# Compact Top Navbar (~64px)
nav_action = render_top_navbar(
    user_role=st.session_state.user_role,
    username=st.session_state.username,
)
if nav_action:
    st.session_state.current_page = nav_action
    st.rerun()

current_page = st.session_state.current_page

# Page Routing
if current_page == "chat":
    # Customer View (Centered Composition)
    if not st.session_state.messages:
        render_hero_banner()

    # Product Selector Card
    products = list_products()
    st.markdown('<div class="selector-container">', unsafe_allow_html=True)
    st.markdown('<div class="selector-label">Your Product</div>', unsafe_allow_html=True)
    prod_options = ["Select your product ▾"] + [f"{p.manufacturer} {p.name}" for p in products]
    sel_prod = st.selectbox("Product", prod_options, label_visibility="collapsed", key="hero_prod_select")

    matching_prod = None
    if sel_prod != "Select your product ▾":
        st.session_state.active_product = sel_prod
        matching_prod = next((p for p in products if f"{p.manufacturer} {p.name}" == sel_prod), None)
        if matching_prod and matching_prod.hardware_version:
            st.session_state.active_version = matching_prod.hardware_version
    else:
        st.session_state.active_product = None
        st.session_state.active_version = None

    # Show Version Selector ONLY if multiple versions or explicit version exists
    if matching_prod and matching_prod.hardware_version:
        st.markdown('<div class="selector-label" style="margin-top: 10px;">Version</div>', unsafe_allow_html=True)
        ver_options = [matching_prod.hardware_version, "V1", "V2", "Rev A"]
        sel_ver = st.selectbox("Version", ver_options, label_visibility="collapsed", key="hero_ver_select")
        if sel_ver:
            st.session_state.active_version = sel_ver

    st.markdown('</div>', unsafe_allow_html=True)

    # Suggestion Pills (Hidden once conversation starts)
    selected_suggestion = None
    if not st.session_state.messages:
        selected_suggestion = render_suggestion_pills()

    # Conversation History Area
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="chat-user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-assistant-card">', unsafe_allow_html=True)
            render_status_message(message)
            st.markdown(f'<div style="font-size: 14px; line-height: 1.6; color: #F8FAFC;">{message["content"]}</div>', unsafe_allow_html=True)

            citations = message.get("citations", [])
            for cit in citations:
                render_source_card(cit)

            interaction_id = message.get("interaction_id")
            if interaction_id:
                st.markdown('<div style="font-size: 12px; color: var(--text-secondary); margin-top: 14px; margin-bottom: 6px;">Was this helpful?</div>', unsafe_allow_html=True)
                fb1, fb2, _ = st.columns([1.5, 1.5, 7])
                if fb1.button("👍 Yes", key=f"y-{idx}"):
                    submit_feedback(interaction_id, helpful=True)
                    st.toast("Thank you for your feedback!")
                if fb2.button("👎 No", key=f"n-{idx}"):
                    submit_feedback(interaction_id, helpful=False)
                    st.toast("Thank you for your feedback.")

            st.markdown('</div>', unsafe_allow_html=True)

    # Primary Question Input Hero
    prompt = st.chat_input("Describe your problem or ask a question...")
    if selected_suggestion and not prompt:
        prompt = selected_suggestion

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

        try:
            result = assistant().answer(
                prompt,
                history,
                active_product=st.session_state.active_product,
                active_version=st.session_state.active_version,
            )
            product_id = slugify(result.product_name or st.session_state.active_product or "general")
            interaction_id = record_interaction(product_id, prompt, result, user_id=st.session_state.username)

            msg_data = {
                "role": "assistant",
                "content": result.answer,
                "citations": [c.__dict__ for c in result.citations],
                "escalated": result.escalated,
                "used_search": result.used_search,
                "used_memory": result.used_memory,
                "interaction_id": interaction_id,
            }
            st.session_state.messages.append(msg_data)

            if result.product_name:
                st.session_state.active_product = result.product_name
            if result.hardware_version:
                st.session_state.active_version = result.hardware_version

            st.rerun()

        except GeminiUnavailable as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error("I could not complete that research request. Please try again.")

elif current_page == "knowledge":
    if st.session_state.user_role == "ADMIN":
        render_knowledge_tab()
    else:
        st.warning("Admin login required.")

elif current_page == "admin":
    if st.session_state.user_role == "ADMIN":
        render_admin_tab()
    else:
        st.warning("Admin login required.")

elif current_page == "eval":
    if st.session_state.user_role == "ADMIN":
        render_eval_tab()
    else:
        st.warning("Admin login required.")
