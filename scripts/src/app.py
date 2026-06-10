import streamlit as st
import os
import io
import contextlib
import logging
from pathlib import Path
from datetime import date
from risk_engine import process_folder, SimpleVectorStore, RiskRegistryMatcher

# --- CONFIGURATION ---
st.set_page_config(page_title="Risk Detection Analytics", layout="wide")
st.title("Automated Risk Analyst")
st.markdown("---")

# --- STATE INITIALIZATION ---
if "report" not in st.session_state:
    st.session_state.report = None
if "logs" not in st.session_state:
    st.session_state.logs = "System idling... Enter parameters to begin."

# --- UI LAYOUT (Always rendered) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Control Center")
    target_directory = st.text_input("Project Folder Path", value="../project_folder")

    if os.path.exists(target_directory):
        files = os.listdir(target_directory)
        st.subheader("Detected Files")
        st.write(files)
    else:
        st.error("Directory not found.")

    start_pipeline = st.button("Execute Risk Audit", type="primary", use_container_width=True)

    st.subheader("Live Operation Logs")
    # This container holds the logs that update during execution
    console_logs_placeholder = st.empty()
    console_logs_placeholder.code(st.session_state.logs)

with col2:
    st.subheader("Generated Executive Narrative")
    # Content container that gets populated after execution
    report_container = st.empty()
    
    if st.session_state.report:
        report_container.markdown(
            f"""
            <div style="
                background-color: #1e293b; color: #f8fafc; padding: 20px; 
                border-radius: 8px; height: 550px; overflow-y: auto; 
                border: 1px solid #334155; font-family: 'Source Sans Pro', sans-serif;
            ">
                <div style="font-size: 16px; line-height: 1.6; white-space: pre-wrap;">
                    {st.session_state.report}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.download_button("Download Risk Report (.txt)", st.session_state.report, "audit_report.txt", use_container_width=True)
    else:
        report_container.info("Results will appear here once the audit is executed.")

# --- PIPELINE EXECUTION ---
if start_pipeline:
    log_output = io.StringIO()
    VECTOR_STORE_PATH = "global_vector_store.json"
    
    with st.spinner("Processing risk parameters..."):
        with contextlib.redirect_stdout(log_output):
            try:
                # ... [Your risk_engine logic] ...
                # Use console_logs_placeholder.code(log_output.getvalue()) here to update logs
                # ...
                st.session_state.report = "Your generated report text here..." 
            except Exception as e:
                st.error(f"Pipeline crashed: {e}")
                
    st.session_state.logs = log_output.getvalue()
    console_logs_placeholder.code(st.session_state.logs)
    # The UI will auto-rerun and render the report in the pre-defined col2 container
    st.rerun()