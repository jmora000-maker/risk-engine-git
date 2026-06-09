import streamlit as st
import os
import io
import contextlib
import logging
from pathlib import Path
from datetime import date

# Import your existing risk engine components
from risk_engine import (
    process_folder, 
    SimpleVectorStore, 
    RiskRegistryMatcher
)

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Next-Gen Risk Delivery Analytics", layout="wide")
st.title("🚀 Automated Risk Analyst")
st.markdown("---")

# --- STATE INITIALIZATION ---
if "report" not in st.session_state:
    st.session_state.report = None
if "logs" not in st.session_state:
    st.session_state.logs = "System idling... Enter parameters to begin."

# --- UI LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Control Center")
    target_directory = st.text_input("Project Folder Path", value="../project_folder")
    
    if st.button("Refresh File List"):
        st.rerun()

    if os.path.exists(target_directory):
        files = os.listdir(target_directory)
        st.subheader("Detected Files")
        st.write(files)
    else:
        st.error("Directory not found.")

    # Execution Button
    start_pipeline = st.button("Execute Risk Audit", type="primary", use_container_width=True)

    st.subheader("Live Operation Logs")
    console_logs = st.empty()
    console_logs.code(st.session_state.logs)

# --- PIPELINE EXECUTION ---
if start_pipeline:
    log_output = io.StringIO()
    VECTOR_STORE_PATH = "global_vector_store.json"
    
    with st.spinner("Processing risk parameters..."):
        with contextlib.redirect_stdout(log_output):
            try:
                # Setup
                register_path = Path(target_directory) / "test_risk.txt"
                matcher = RiskRegistryMatcher(register_path) if register_path.exists() else None
                store = SimpleVectorStore()
                
                # Ingestion Logic
                if os.path.exists(VECTOR_STORE_PATH):
                    print(f"Found existing store. Loading...")
                    store.load(VECTOR_STORE_PATH)
                else:
                    print("No store found. Ingesting files...")
                    data = process_folder(target_directory)
                    if data:
                        store.add_many(data)
                        store.save(VECTOR_STORE_PATH)
                
                # Audit Logic
                if store.entries:
                    audit_queries = [
                        "staffing turnover, resource departures, personnel shortages",
                        "security vulnerabilities, mTLS failures, unauthorized data access",
                        "data pipeline errors, system latency, memory leaks, parsing crashes"
                    ]
                    
                    report_lines = [f"UNREGISTERED RISK DISCOVERY REPORT - {date.today()}\n", "="*40]
                    found_any = False
                    
                    for query in audit_queries:
                        report_lines.append(f"\n--- Checking for: {query.split(',')[0]} ---")
                        for match in store.search(query, top_k=5):
                            is_new, _ = matcher.is_unregistered(match["text"]) if matcher else (True, 0)
                            if is_new:
                                found_any = True
                                report_lines.append(f"[NEW] SOURCE: {match['source']}\nDESC: {match['text']}\n" + "-"*20)
                    
                    if not found_any:
                        report_lines.append("No new unregistered risks identified.")
                    
                    st.session_state.report = "\n".join(report_lines)
                else:
                    print("No data in store.")
            
            except Exception as e:
                st.error(f"Pipeline crashed: {e}")
                logging.error(f"Execution failure: {e}", exc_info=True)

    st.session_state.logs = log_output.getvalue()
    console_logs.code(st.session_state.logs)

# --- DISPLAY OUTPUTS ---
with col2:
    st.subheader("Generated Executive Narrative")
    if st.session_state.report:
        st.markdown(
            f"""
            <div style="
                background-color: #1e293b; 
                color: #f8fafc; 
                padding: 20px; 
                border-radius: 8px; 
                height: 550px; 
                overflow-y: scroll; 
                white-space: pre-wrap; 
                font-family: monospace;
                border: 1px solid #334155;
            ">{st.session_state.report}</div>
            """, 
            unsafe_allow_html=True
        )
        st.download_button(
            label="Download Risk Report (.txt)",
            data=st.session_state.report,
            file_name="audit_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("Results will appear here once the audit is executed.")