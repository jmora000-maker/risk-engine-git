import streamlit as st
import os
import io
import contextlib
import logging
import time
from pathlib import Path
from datetime import date
# Import your custom modules
from risk_engine import process_folder, SimpleVectorStore, RiskRegistryMatcher

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Risk Detection Analytics", layout="wide")
st.title("Automated Risk Audit Dashboard")
st.markdown("---")

# --- STATE INITIALIZATION ---
if "report" not in st.session_state:
    st.session_state.report = None
if "logs" not in st.session_state:
    st.session_state.logs = "Ready for execution..."

# --- UI LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    target_directory = "../project_folder"
    st.subheader("1. System Configuration")
    st.text(target_directory)

    if os.path.exists(target_directory):
        files = os.listdir(target_directory)
        st.subheader("2. Control Center")
        st.write(files)
    else:
        st.error("Directory not found.")

    start_pipeline = st.button("Execute Risk Audit", type="primary", use_container_width=True)

    st.markdown("**Live Operational Console Logs:**")
    # A persistent container to show logs
    log_placeholder = st.empty()
    log_placeholder.code(st.session_state.logs)

# --- PIPELINE EXECUTION ---
if start_pipeline:
    # 1. Initialize buffer for this specific run
    buffer = io.StringIO()
    VECTOR_STORE_PATH = "global_vector_store.json"
    
    with st.spinner("Processing risk parameters..."):
        # 2. Redirect standard output (print statements) to our buffer
        with contextlib.redirect_stdout(buffer):
            try:
                # --- START OF YOUR LOGIC ---
                print("Step 1: Ingesting files...")
                log_placeholder.code(buffer.getvalue())  # Immediate UI update
                time.sleep(1) # Simulate work

                print("Step 2: Normalizing risk data...")
                log_placeholder.code(buffer.getvalue())
                time.sleep(1)

                print("Step 3: Initializing Engine...")
                log_placeholder.code(buffer.getvalue())
                
                register_path = Path(target_directory) / "test_risk.txt"
                matcher = RiskRegistryMatcher(register_path) if register_path.exists() else None
                store = SimpleVectorStore()
                
                print("Step 4: Vector Store operations...")
                if os.path.exists(VECTOR_STORE_PATH):
                    store.load(VECTOR_STORE_PATH)
                else:
                    data = process_folder(target_directory)
                    if data:
                        store.add_many(data)
                        store.save(VECTOR_STORE_PATH)
                
                # --- AUDIT LOGIC ---
                print("Step 5: Executing audit...")
                log_placeholder.code(buffer.getvalue())
                
                if store.entries:
                    audit_queries = [
                        "staffing turnover, resource departures, personnel shortages",
                        "security vulnerabilities, mTLS failures, unauthorized data access",
                        "data pipeline errors, system latency, memory leaks, parsing crashes"
                    ]
                    
                    report_lines = [f"UNREGISTERED RISK REPORT - {date.today()}",
                        "", 
                        "" 
                    ]
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
                
                print("Pipeline completed successfully.")
                # --- END OF LOGIC ---
                
            except Exception as e:
                print(f"Error: {e}")
                logging.error(f"Execution failure: {e}", exc_info=True)
                log_placeholder.code(buffer.getvalue())

    # Save to session state to persist after rerun
    st.session_state.logs = buffer.getvalue()
    st.rerun()

# --- DISPLAY OUTPUTS ---
with col2:
    st.subheader("3. Executive Synthesis Workspace")

    # 1. Check if the report exists in session state
    if st.session_state.report:
        # 1. Clean text: Replace markdown heading prefixes to ensure it stays standard size
        clean_report = st.session_state.report
        if clean_report.startswith("#"):
            clean_report = clean_report.lstrip("#").strip()

        # 2. Render using Streamlit's native HTML container
        st.html(
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
                line-height: 1.5;
            ">
                <p style="font-size: 14px !important; margin: 0; padding: 0;">{clean_report}</p>
            </div>
            """
        )

        # 3. Add the download button below the styled container
        st.download_button(
            label="Download Risk Report (.txt)",
            data=st.session_state.report,
            file_name="audit_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("Results will appear here once the audit is executed.")