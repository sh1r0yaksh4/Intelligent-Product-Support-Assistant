from __future__ import annotations

import streamlit as st
from rag.config import ROOT_DIR


def load_css() -> None:
    css_path = ROOT_DIR / "static" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ── Customer Top Navbar ────────────────────────────────────────────────────────

def render_top_navbar(user_role: str = "CUSTOMER", username: str = "guest") -> str | None:
    load_css()
    col_l, col_r = st.columns([6, 4])
    with col_l:
        st.markdown(
            """
            <div class="nav-brand">
                <div class="nav-logo-icon">S</div>
                <div class="nav-brand-title">SupportAI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    action_page = None
    with col_r:
        if st.session_state.get("authenticated"):
            role_badge = (
                f'<span style="font-size:11px; background:rgba(139,92,246,0.15); '
                f'color:#C084FC; padding:2px 8px; border-radius:12px; '
                f'border:1px solid rgba(139,92,246,0.3); font-weight:600;">'
                f'{user_role}</span>'
            )
            st.markdown(
                f'<div style="text-align:right; font-size:13px; margin-top:4px;">'
                f'{username}&nbsp;{role_badge}</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 1])
            if user_role == "ADMIN":
                if c1.button("Admin Panel", key="top-nav-admin", use_container_width=True):
                    action_page = "admin_dashboard"
            if c2.button("Log Out", key="top-nav-logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = "guest"
                st.session_state.user_role = "CUSTOMER"
                st.session_state.current_page = "chat"
                st.rerun()
        else:
            if st.button("Sign In", key="top-nav-signin", use_container_width=True):
                action_page = "login"

    st.markdown('<div class="navbar-divider"></div>', unsafe_allow_html=True)
    return action_page


# ── Admin Top Navigation Bar ───────────────────────────────────────────────────

def render_admin_navbar(current_admin_page: str) -> str:
    """Renders a compact horizontal admin nav. Returns the currently active admin page."""
    load_css()
    pages = [
        ("📊 Dashboard", "admin_dashboard"),
        ("🗂️ Knowledge Sources", "admin_sources"),
        ("🔬 Evaluation", "admin_eval"),
    ]

    st.markdown('<div class="admin-nav">', unsafe_allow_html=True)
    nav_cols = st.columns([2] + [2] * len(pages) + [1])

    with nav_cols[0]:
        st.markdown(
            '<div class="admin-nav-logo">'
            '<div class="nav-logo-icon">S</div>'
            '<span style="font-size:15px;font-weight:700;">SupportAI</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    selected = current_admin_page
    for i, (label, page_key) in enumerate(pages):
        with nav_cols[i + 1]:
            is_active = current_admin_page == page_key
            div_class = "admin-nav-btn-active" if is_active else ""
            st.markdown(f'<div class="{div_class}">', unsafe_allow_html=True)
            if st.button(label, key=f"adm-nav-{page_key}", use_container_width=True):
                selected = page_key
                st.session_state.current_page = page_key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    with nav_cols[-1]:
        if st.button("← Back", key="adm-nav-back", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    return selected


# ── Hero & Login ───────────────────────────────────────────────────────────────

def render_hero_banner() -> None:
    load_css()
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-badge">✦ AI PRODUCT SUPPORT</div>
            <div class="hero-heading">How can we help <span class="gradient-text">you today?</span></div>
            <div class="hero-sub">Get reliable answers backed by trusted product sources.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_standalone_login_screen() -> None:
    load_css()
    st.markdown(
        """
        <div class="centered-login-card">
            <div class="nav-logo-icon" style="margin:0 auto 16px auto; width:40px; height:40px; font-size:20px;">S</div>
            <div style="font-size:22px; font-weight:800; color:var(--text-primary); margin-bottom:6px;">Welcome back</div>
            <div style="font-size:13px; color:var(--text-secondary); margin-bottom:24px;">Sign in to access your product support assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        u_val = st.text_input("Username", key="login_u_field", placeholder="admin or customer")
        p_val = st.text_input("Password", type="password", key="login_p_field", placeholder="admin123 or customer123")

        st.markdown('<div class="btn-gradient">', unsafe_allow_html=True)
        do_login = st.button("Sign in", key="login_btn_submit", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="text-align:center; font-size:12px; color:var(--text-secondary); margin-top:16px;">'
            'Customer login · Admin login</div>',
            unsafe_allow_html=True,
        )

        if do_login:
            from auth.security import authenticate_user
            u_rec = authenticate_user(u_val, p_val)
            if u_rec:
                st.session_state.authenticated = True
                st.session_state.username = u_rec.username
                st.session_state.user_role = u_rec.role
                # Admins go straight to admin panel
                st.session_state.current_page = (
                    "admin_dashboard" if u_rec.role == "ADMIN" else "chat"
                )
                st.rerun()
            else:
                st.error("Invalid username or password.")


# ── Chat / Customer Components ─────────────────────────────────────────────────

def render_suggestion_pills() -> str | None:
    st.markdown('<div class="suggestion-header">Try asking</div>', unsafe_allow_html=True)
    examples = [
        "How do I reset my device?",
        "Why is my Wi-Fi not working?",
        "How do I update the firmware?",
    ]
    selected_example = None
    c1, c2, c3 = st.columns(3)
    for idx, (col, ex) in enumerate(zip([c1, c2, c3], examples)):
        with col:
            st.markdown('<div class="suggestion-pill">', unsafe_allow_html=True)
            if st.button(ex, key=f"sug-{idx}", use_container_width=True):
                selected_example = ex
            st.markdown('</div>', unsafe_allow_html=True)
    return selected_example


def render_status_message(message: dict) -> None:
    if message.get("escalated"):
        pill, cls = "⚠️ Unable to verify — please check official manufacturer support.", "status-escalated"
    elif message.get("used_search"):
        pill, cls = "✓ Found on official support website", "status-web"
    else:
        pill, cls = "✓ Verified from official documentation", "status-doc"
    st.markdown(f'<div class="status-pill {cls}">{pill}</div>', unsafe_allow_html=True)


def render_source_card(citation: dict) -> None:
    title = citation.get("title", "Official User Guide")
    section = citation.get("section")
    page = citation.get("page")
    url = citation.get("url")
    meta_parts = []
    if section:
        meta_parts.append(f"Section: {section}")
    if page:
        meta_parts.append(f"Page {page}")
    meta_str = " · ".join(meta_parts) if meta_parts else "Official documentation"
    link_html = (
        f'<a href="{url}" target="_blank" style="color:var(--color-purple);font-weight:600;text-decoration:none;">View →</a>'
        if url else '<span style="color:var(--text-secondary);">Local manual</span>'
    )
    st.markdown(
        f"""
        <div class="source-card">
            <div>
                <div class="source-card-title">📄 {title}</div>
                <div class="source-card-sub">{meta_str}</div>
            </div>
            <div>{link_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Compatibility stubs (used by knowledge_tab & eval_tab) ────────────────────

def render_top_header(title: str, subtitle: str = "") -> None:
    load_css()
    st.markdown(
        f'<div class="page-heading">{title}</div>'
        f'<div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def render_status_badge(status_type: str) -> str:
    s = status_type.upper()
    if s == "APPROVED":
        return '<span class="badge-approved">APPROVED</span>'
    elif s == "PENDING":
        return '<span class="badge-pending">PENDING</span>'
    elif s == "REJECTED":
        return '<span class="badge-rejected">REJECTED</span>'
    return f'<span>{status_type}</span>'


def render_empty_state(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div style="text-align:center; padding:32px; background:var(--bg-card);
             border-radius:12px; border:1px solid var(--border-color); margin:16px 0;">
            <div style="font-size:32px; margin-bottom:10px;">{icon}</div>
            <div style="font-weight:700; font-size:15px; color:var(--text-primary);">{title}</div>
            <div style="font-size:13px; color:var(--text-secondary); margin-top:6px;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simple_citation(citation: dict) -> None:
    render_source_card(citation)


def render_app_header() -> None:
    render_hero_banner()
