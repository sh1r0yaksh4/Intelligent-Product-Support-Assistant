from __future__ import annotations

import streamlit as st
from database.db import get_db
from rag.admin import approve_source, create_product, delete_source, list_products, list_sources, reject_source
from rag.sources import fetch_webpage, save_upload


def render_admin_tab() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 24px; font-weight: 800; color: var(--text-primary);">Knowledge Sources</div>
            <div style="font-size: 14px; color: var(--text-secondary);">Review official documents before they are used by the assistant.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    products = list_products()
    sources = list_sources()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM interactions;")
        conversations_count = cursor.fetchone()[0]

    pending_sources = [s for s in sources if s.status == "PENDING"]

    # 4 Compact Metric Boxes
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="admin-metric-box"><div class="admin-metric-number">{len(products)}</div><div class="admin-metric-title">Products</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="admin-metric-box"><div class="admin-metric-number">{len(sources)}</div><div class="admin-metric-title">Sources</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="admin-metric-box"><div class="admin-metric-number" style="color: var(--color-pink);">{len(pending_sources)}</div><div class="admin-metric-title">Pending</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="admin-metric-box"><div class="admin-metric-number" style="color: var(--color-cyan);">{conversations_count}</div><div class="admin-metric-title">Conversations</div></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Section 1: Pending Approvals Queue
    st.subheader("Pending Review Queue")
    if not pending_sources:
        st.info("No pending sources waiting for review.")
    else:
        for src in pending_sources:
            prod_info = next((p for p in products if p.id == src.product_id), None)
            p_name = f"{prod_info.manufacturer} {prod_info.name}" if prod_info else src.product_id

            st.markdown(
                f"""
                <div style="background: var(--bg-card); padding: 14px; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom: 8px;">
                    <b>Source:</b> {src.source_name} &nbsp;|&nbsp; <b>Product:</b> {p_name} &nbsp;|&nbsp; <b>Type:</b> {src.source_type.upper()}
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_a, col_r, col_d, _ = st.columns([1.5, 1.5, 1.5, 5.5])
            if col_a.button("Approve", key=f"app-{src.id}", use_container_width=True):
                try:
                    approve_source(src.id)
                    st.success("Approved and indexed!")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

            if col_r.button("Reject", key=f"rej-{src.id}", use_container_width=True):
                reject_source(src.id)
                st.warning("Rejected.")
                st.rerun()

            if col_d.button("Delete", key=f"del-{src.id}", use_container_width=True):
                delete_source(src.id)
                st.info("Deleted.")
                st.rerun()
            st.markdown("<br/>", unsafe_allow_html=True)

    st.divider()

    # Section 2: Ingest & Add Product
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Upload Document / Web URL")
        if not products:
            st.warning("Create a product first.")
        else:
            prod_map = {f"{p.manufacturer} {p.name}": p.id for p in products}
            sel_p_label = st.selectbox("Select Product", list(prod_map.keys()))
            target_p_id = prod_map[sel_p_label]

            up_file = st.file_uploader("Upload Manual (PDF, TXT, MD)", type=["pdf", "txt", "md"])
            if st.button("Upload Document", disabled=not up_file, use_container_width=True):
                try:
                    save_upload(target_p_id, up_file.name, up_file.getvalue())
                    st.success("Uploaded in Pending state!")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

            src_url = st.text_input("Official Support Page URL")
            if st.button("Fetch Support Page", disabled=not src_url, use_container_width=True):
                try:
                    fetch_webpage(target_p_id, src_url.strip())
                    st.success("Page saved in Pending state!")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with c_right:
        st.subheader("Create Product")
        mfg = st.text_input("Manufacturer", placeholder="e.g. TP-Link, Sony")
        name = st.text_input("Product Name", placeholder="e.g. Archer AX21")
        model = st.text_input("Model", placeholder="e.g. AX21")
        hver = st.text_input("Hardware Version", placeholder="e.g. V2")

        st.markdown('<div class="btn-gradient">', unsafe_allow_html=True)
        do_create = st.button("Create Product Workspace", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if do_create:
            if not name.strip() or not mfg.strip():
                st.error("Manufacturer and Product Name are required.")
            else:
                create_product(mfg, name, model or name, hver)
                st.success(f"Product '{name}' created!")
                st.rerun()
