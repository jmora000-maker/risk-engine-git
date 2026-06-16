import streamlit as st
import os
import contextlib
from pathlib import Path
import json
import logging
from datetime import date
import csv
import numpy as np
import pandas as pd
from openai import OpenAI

# Define paths and global variables
today = date.today()

current_script_dir=Path(__file__).resolve().parent
project_root=current_script_dir.parent  #.parent go up one directory level from script location

log_folder = project_root / "logs"
output_folder = project_root / "outputs"
project_folder = project_root / "project_folder"

# Initialize client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# --- UTILITY TO CAPTURE STDOUT ---
# This class redirects standard output to a Streamlit text component in real-time.
class StreamlitStdoutRedirector:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.output_str = ""

    def write(self, text):
        self.output_str += text
        self.placeholder.code(self.output_str, language="text")

    def flush(self):
        pass

# --- ADD SCRIPT FUNCTIONS HERE ---

# --- EMBEDDING API CONNECTOR ---
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Fetches high-dimensional numeric coordinates for any text string from OpenAI."""
    cleaned_text = str(text).replace("\n", " ").strip()
    if not cleaned_text:  # Edge-case catch for blank data rows
        cleaned_text = "Empty row placeholder"
    response = client.embeddings.create(
        input=[cleaned_text],
        model=model
    )
    return response.data[0].embedding

# --- MATHEMATICAL SIMILARITY CALCULATIONS ---
def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Measures the mathematical vector proximity metric between coordinate sets."""
    a = np.array(v1)
    b = np.array(v2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

# --- FILE TYPE-SPECIFIC CHUNKING FUNCTIONS ---
# Each function handles a specific file format with specialized parsing logic
# Returns a list of dictionaries containing the chunked text and metadata
# Metadata includes source filename and unique chunk identifier
# These functions are called by the process_folder function based on file extension

# For plain text files, splits into word chunks with overlap
def chunk_text_file(filepath: str, chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    """Reads standard *.txt files and executes sliding window word chunking."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    chunks = []
    start = 0
    filename = os.path.basename(filepath)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_string = " ".join(words[start:end])

        chunks.append({
            "text": chunk_string,
            "source": filename,
            "chunk_id": f"{filename}_chunk_{len(chunks) + 1}"
        })

        if end >= len(words):
            break
        start = end - overlap

    return chunks

# For CSV files, reads each row as a separate chunk
def chunk_csv_file(filepath: str) -> list[dict]:
    """Reads *.csv spreadsheets and maps each line into a row dictionary asset."""
    chunks = []
    filename = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            # Flatten spreadsheet cells into a structured semantic string
            row_as_text = " | ".join([f"{col}: {val}" for col, val in row.items() if val])

            chunks.append({
                "text": row_as_text,
                "source": filename,
                "chunk_id": f"{filename}_row_{i}"
            })
    return chunks

# For Excel files, reads each row as a separate chunk
def chunk_excel_file(filepath: str) -> list[dict]:
    """Ingests *.xls and *.xlsx spreadsheets using pandas for structural parsing."""
    chunks = []
    filename = os.path.basename(filepath)

    # Read the spreadsheet (defaults to the first active worksheet)
    df = pd.read_excel(filepath)
    # Fill empty data cells (NaN) with empty string blocks to prevent parsing crashes
    df = df.fillna("")

    for index, row in df.iterrows():
        row_dict = row.to_dict()
        # Convert row map variables into a unified semantic sequence string
        row_as_text = " | ".join([f"{col}: {val}" for col, val in row_dict.items() if val != ""])

        chunks.append({
            "text": row_as_text,
            "source": filename,
            "chunk_id": f"{filename}_row_{index + 2}"  # Accounting for 1-index header maps
        })
    return chunks

# For structured issue logs, splits by unique issue markers
def chunk_issue_log(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split the file by the unique marker that starts each record
    # This creates a list where each item is one complete issue
    raw_records = text.split("Issue ID:")

    chunks = []
    filename = os.path.basename(filepath)

    for i, record in enumerate(raw_records):
        if not record.strip(): continue # Skip empty splits

        # Add the marker back so the text makes sense
        chunk_text = f"Issue ID:{record.strip()}"

        chunks.append({
            "text": chunk_text,
            "source": filename,
            "chunk_id": f"{filename}_issue_{i}"
        })
    return chunks

# For risk register files, reads each row as a separate chunk
def chunk_risk_file(filepath: str) -> list[dict]:
    """Reads the risk register file row-by-row to prevent chunking the entire file at once."""
    chunks = []
    filename = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        # Use DictReader if it's CSV-like, otherwise read lines
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            # Create a clean, semantic description for each row
            row_as_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v])

            chunks.append({
                "text": row_as_text,
                "source": filename,
                "chunk_id": f"{filename}_row_{i}"
            })
    return chunks


# --- CENTRAL DIRECTORY SCANNER FUNCTION ---

# This function scans a directory and processes each file based on its type
# It uses the appropriate chunking function for each file type
# It returns a list of all chunks from all files in the directory

def process_folder(folder_path: str, chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    """Scans a specified folder directory path and dynamically routes matched document types."""

    # Define the name of the file to ignore
    registry_filename = "test_risk.txt"

    all_chunks = []

    if not os.path.exists(folder_path):
            print(f" -> Directory target folder '{folder_path}' missing. Indexer aborted.")
            return []

    # Read items found inside directory parameters
    files_in_folder = os.listdir(folder_path)

    for file in files_in_folder:
        # EXCLUSION LOGIC: Skip the register file
        if file.lower() == registry_filename.lower():
            print(f" -> Skipping risk register file: {file}")
            continue

        full_path = os.path.join(folder_path, file)

        # Guard clause: skip items that are directories
        if os.path.isdir(full_path):
            continue

        file_lower = file.lower()
        file_chunks = []

        # Strategic Type Routing Mechanism
        if file_lower == "test_risk.txt":
            print(f" -> Chunking risk register: {file}")
            file_chunks = chunk_risk_file(full_path)
        elif file_lower == "issue_log.txt":
            print(f" -> Chunking issue log: {file}")
            file_chunks = chunk_issue_log(full_path)
        elif file_lower.endswith(".txt"):
            print(f" -> Chunking plain-text document: {file}")
            file_chunks = chunk_text_file(full_path, chunk_size, overlap)
        elif file_lower.endswith(".csv"):
            print(f" -> Chunking comma-separated spreadsheet: {file}")
            file_chunks = chunk_csv_file(full_path)
        elif file_lower.endswith(".xlsx") or file_lower.endswith(".xls"):
            print(f" -> Chunking Excel spreadsheet: {file}")
            file_chunks = chunk_excel_file(full_path)
        else:
            # Skip unmapped formats (e.g., pdf, zip, png) silently
            continue

        all_chunks.extend(file_chunks)

    return all_chunks


# --- VECTOR STORE IMPLEMENTATION ---
# This class handles the storage and retrieval of vector embeddings
# It provides methods to add chunks, search for similar chunks, and save the store
class SimpleVectorStore:
    def __init__(self):
        self.entries = []

    def load(self, filepath: str) -> None:
        """Loads the vector store entries from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
            print(f" -> Vector Store loaded {len(self.entries)} entries from {filepath}")
        else:
            raise FileNotFoundError(f" Vector store file not found at {filepath}")

    # Add a single chunk to the store
    def add_many(self, chunks: list[dict]) -> None:
        print(f" -> Sending {len(chunks)} chunks to OpenAI for vector synthesis...")
        print(f" -> This may take a few moments...")
        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            self.entries.append({**chunk, "embedding": embedding})

    # Search for similar chunks based on a query
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = get_embedding(query)
        scored = []
        for entry in self.entries:
            sim = cosine_similarity(query_embedding, entry["embedding"])
            scored.append((sim, entry))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [{**entry, "similarity": round(sim, 4)} for sim, entry in scored[:top_k]]

    # Save the store to a file
    def save(self, filepath: str) -> None:
        print(f" -> Saving vector store entries to {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=4)

# --- RISK REGISTRY MATCHER ---
# This class handles the comparison of candidate risks against the registered risks
# It uses embeddings to determine if a candidate risk is already registered
# It provides a method to check if a candidate risk is unregistered
class RiskRegistryMatcher:
    def __init__(self, registered_risks_filepath):
        # Load your master register
        self.register = self._load_register(registered_risks_filepath)

    # Load the registered risks from a file
    #TODO this should be
    def _load_register(self, filepath):
        # Assuming your register is a simple text file of known issues
        with open(filepath, "r") as f:
            lines = f.readlines()
        # Create embeddings for each known risk
        return [{"text": line.strip(), "embedding": get_embedding(line.strip())}
                for line in lines if line.strip()]

    # Check if a candidate risk is unregistered
    def is_unregistered(self, candidate_text, threshold=0.85):
        candidate_embedding = get_embedding(candidate_text)

        for known_risk in self.register:
            score = cosine_similarity(candidate_embedding, known_risk["embedding"])
            # If similarity is high, it's already registered
            if score >= threshold:
                return False, score # Found a match

        return True, 0.0 # No match found, it's unregistered


def run_risk_audit(store: SimpleVectorStore, matcher: RiskRegistryMatcher, audit_queries: list[str]) -> list[dict]:
    """
    Executes the semantic analysis against risk categories.
    Returns a structured collection of newly discovered unregistered risks.
    """
    audit_results = []

    for query in audit_queries:
        category_name = query.split(',')[0].strip()
        # Clean progress markers for StreamlitStdoutRedirector capture
        print(f" -> Scanning Vector Space for Category: '{category_name.upper()}'")

        category_data = {
            "category": category_name,
            "discovered_risks": []
        }

        # Search vector store for context matching the query parameters
        results = store.search(query, top_k=5)

        new_count = 0
        for match in results:
            # Check similarity threshold alignment
            is_new, score = matcher.is_unregistered(match["text"]) if matcher else (True, 0.0)

            if is_new:
                new_count += 1
                text = match.get("text", "")
                source_doc = match.get("source", "Unknown Source")

                # Safely parse back CSV/Excel row key-value string chunks
                fields = {part.split(": ")[0].strip(): part.split(": ")[1].strip()
                          for part in text.split(" | ") if ": " in part}

                category_data["discovered_risks"].append({
                    "source": source_doc,
                    "description": fields.get("Description", text)
                })

        print(f" -> Complete. Identified {new_count} anomalies against Master Register.")
        audit_results.append(category_data)

    print(f" -> Analysis loop finished successfully.")
    return audit_results

# TODO update report
#  --- RESTRUCTURED REPORT WRITER (SILENCED PRINTS) ---
def generate_audit_report(audit_results: list[dict], file_path: Path, today_str: str) -> str:
    """
    Takes structured audit results, writes a formatted presentation report
    to a physical text file, and returns the full report string for Streamlit display.
    """

    print(f" -> Performing automated report generation for {len(audit_results)} risk categories...")

    lines = []
    total_categories = len(audit_results)
    total_unregistered_risks = sum(len(section.get("discovered_risks", [])) for section in audit_results)

    # --- REPORT HEADER ---
    lines.append("RISK AUDIT REPORT")
    lines.append(f"Report Date: {today_str}")
    lines.append(f"Summary: There were {total_categories} risk categories and {total_unregistered_risks} new risks identified")
    lines.append("")

    unregistered_risks_found = False

    # --- CATEGORY SCANNING LOOP ---
    for section in audit_results:
        category = section.get("category", "Unspecified Category")
        discovered_risks = section.get("discovered_risks", [])

        lines.append(f" --- CATEGORY: {category.upper()} ---")

        if discovered_risks:
            for risk in discovered_risks:
                source = risk.get("source", "Unknown Source")
                description = risk.get("description", "No description narrative provided.")
                lines.append(f"SOURCE: {source}")
                lines.append(f"DESCRIPTION: {description}")
            lines.append("")  # Uniform padding after category blocks
        else:
            lines.append("  > No significant unregistered risks detected in this category.")
            lines.append("")

    # --- GLOBAL SUMMARY EVALUATION ---
    if total_unregistered_risks == 0:
        lines.append("SUMMARY EVALUATION:")
        lines.append("No new unregistered risks were identified across any analyzed categories.")
        lines.append("")

    #Combine array elements into a single uniform string object
    final_report_text = "\n".join(lines).strip()

    # Write cleanly to hard drive persistence layer
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_report_text)

    # Return the verified text variable straight to your Streamlit columns session reference
    # return final_report_text
    return final_report_text

# --- CORE PIPELINE EXECUTION WRAPPER ---
def run_automated_pipeline(log_placeholder):
    # --- 1. CONFIGURATION ---
    target_directory = project_folder
    database_file_destination = "global_vector_store.json"
    register_path = os.path.join(target_directory, "test_risk.txt")
    file_path = output_folder / "UNREGISTERED_RISK_DISCOVERY_REPORT.txt"

    # Pipeline Execution
    try:
        print("PIPELINE STARTED.")

        # --- 1. CREATING VECTOR STORE ---
        print("STEP 1: Creating Vector Store.")
        store = SimpleVectorStore()

        # Chunking documents
        compiled_data_chunks = process_folder(target_directory, chunk_size=500, overlap=80)

        # Converting chunks to embedding vectors
        store.add_many(compiled_data_chunks)

        # Save vector store to file
        store.save(database_file_destination)

        # --- 2. Identifying unregistered risks ---
        print(f"STEP 2: Starting automated risk audit.")
        matcher = RiskRegistryMatcher(register_path) if os.path.exists(register_path) else None

        if not matcher:
            print(f"Warning: Register not found at {register_path}. Skipping registration check.")

        audit_queries = [
            "staffing turnover, resource departures, personnel shortages",
            "security vulnerabilities, mTLS failures, unauthorized data access",
            "data pipeline errors, system latency, memory leaks, parsing crashes"
        ]

        # Extract data layout parameters
        discovered_risk_data = run_risk_audit(store, matcher, audit_queries)

        # 3. Generate file on disk AND extract string payload
        print("STEP 3: Generating Risk Audit Report.")
        final_report_text = generate_audit_report(discovered_risk_data, file_path, today)

        print("PIPELINE COMPLETED.")

        return final_report_text

    except Exception as e:
        print(f"Pipeline crashed with an unhandled traceback exception: {e}")

        return None

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(
    page_title="AI Risk Audit Generator",
    layout="wide"
)

# APPLICATION TITLE
st.title("Risk Audit Dashboard")
st.markdown("---")

# Split dashboard workspace view evenly into two layout control blocks
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Configuration")
    # 1. Get the directory of app.py
    SCRIPT_DIR = Path(__file__).resolve().parent
    SCRIPTS_DIR = SCRIPT_DIR.parent
    target_directory = SCRIPTS_DIR / "project_folder"

    if target_directory.exists():
        st.text("Files found in 'project_folder':")
        files = os.listdir(str(target_directory))
        st.write(files)
    else:
        st.error(f"Directory not found")
        if SCRIPTS_DIR.exists():
            st.warning("Folders found in 'scripts':")
            st.write(os.listdir(str(SCRIPTS_DIR)))

# Check for files

    # Core system action trigger interface button
    start_pipeline = st.button("Generate Risk Audit Report", use_container_width=True, type="primary")

    st.subheader("Pipeline Summary")
    # Interactive log tracing viewport block
    console_logs = st.empty()
    console_logs.info("Click 'Generate Risk Audit Report' button to begin.")

# Persistent frame layout setup for Column 2 immediately on boot
with col2:
    st.subheader("Report Workspace")
    report_placeholder = st.empty()

    # Pre-execution placeholder info state setup
    report_placeholder.info("The Risk Audit Report will populate here upon synthesis.")

# Active process handler evaluations
if start_pipeline:
    redirector = StreamlitStdoutRedirector(console_logs)

    with st.spinner("Processing risk parameters..."):
        # Wrap the stream interceptor strictly around the pipeline engine call
        with contextlib.redirect_stdout(redirector):
            final_narrative = run_automated_pipeline(console_logs)

    if final_narrative:
        # Create a container inside the placeholder to hold both the HTML and the button
        with report_placeholder.container():
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
                    font-family: inherit;
                    border: 1px solid #334155;
                    line-height: 1.5;
                ">
                    <p style="font-size: 16px !important; margin: 0; padding: 0;">{final_narrative}</p>
                </div>
                """
            )

            st.download_button(
                label="Download Risk Audit Report (.txt)",
                data=final_narrative,
                file_name="audit_report.txt",
                mime="text/plain",
                use_container_width=True
            )
