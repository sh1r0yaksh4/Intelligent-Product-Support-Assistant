from __future__ import annotations

import streamlit as st
from rag.admin import list_products, list_sources
from ui.components import render_empty_state, render_status_badge, render_top_header


def render_knowledge_tab() -> None:
    render_top_header("Knowledge Base", "Explore trusted product manuals, hardware specifications, and approved documentation.")

    products = list_products()
    sources = list_sources()

    if not sources:
        render_empty_state(
            "📚",
            "No Knowledge Base Sources Yet",
            "Upload official documentation or add a support URL in the Admin panel to populate the knowledge base.",
        )
        return

    # Filter Bar
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    search_query = c1.text_input("🔍 Search Sources", placeholder="Search document name...")

    prod_options = ["All Products"] + [f"{p.manufacturer} {p.name}" for p in products]
    selected_prod = c2.selectbox("Product Filter", prod_options)

    type_options = ["All Types", "pdf", "txt", "md", "url"]
    selected_type = c3.selectbox("Source Type", type_options)

    status_options = ["All Statuses", "APPROVED", "PENDING", "REJECTED"]
    selected_status = c4.selectbox("Approval Status", status_options)

    st.divider()

    # Filter Application
    filtered = sources
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = [s for s in filtered if q in s.source_name.lower() or q in s.product_id.lower()]

    if selected_prod != "All Products":
        # Extract product ID slug from list
        prod_obj = next((p for p in products if f"{p.manufacturer} {p.name}" == selected_prod), None)
        if prod_obj:
            filtered = [s for s in filtered if s.product_id == prod_obj.id]

    if selected_type != "All Types":
        filtered = [s for s in filtered if s.source_type == selected_type]

    if selected_status != "All Statuses":
        filtered = [s for s in filtered if s.status == selected_status]

    if not filtered:
        render_empty_state(
            "🔎",
            "No Matching Sources Found",
            "Try broadening your search query or adjusting your status and product filters.",
        )
        return

    # Render Source Cards Grid
    st.caption(f"Showing {len(filtered)} matching knowledge base sources")
    for src in filtered:
        badge_html = render_status_badge(src.status)
        prod_info = next((p for p in products if p.id == src.product_id), None)
        p_label = f"{prod_info.manufacturer} {prod_info.name} (Ver: {prod_info.hardware_version or 'All'})" if prod_info else src.product_id

        st.markdown(
            f"""
            <div class="saas-metric-card" style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; font-size: 15px; color: var(--text-primary);">📄 {src.source_name}</div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
                            Product: <b>{p_label}</b> · Type: <code style="color: var(--accent-color);">{src.source_type.upper()}</code> · Chunks Indexed: <b>{src.chunk_count}</b>
                        </div>
                        {"<div style='font-size: 11px; color: var(--text-secondary); margin-top: 4px;'>URL: <a href='" + src.source_url + "' target='_blank' style='color: var(--accent-color);'>" + src.source_url + "</a></div>" if src.source_url else ""}
                    </div>
                    <div>{badge_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
