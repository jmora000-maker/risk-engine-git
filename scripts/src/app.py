import csv
import json
import os
from datetime import date
from pathlib import Path
import contextlib
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field

# Define paths and global variables
# This gives you a datetime object
today_obj = date.today()
today = today_obj.strftime("%B %d, %Y")

current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

log_folder = project_root / "logs"
output_folder = project_root / "outputs"
project_folder = project_root / "project_folder"
vector_store_folder = project_root / "vector_store"

report_path = output_folder / "UNREGISTERED_RISK_DISCOVERY_REPORT.txt"
database_file_destination = vector_store_folder / "global_vector_store.json"
register_path = project_folder / "test_risk.txt"

# Ensure they exist individually
log_folder.mkdir(parents=True, exist_ok=True)
vector_store_folder.mkdir(parents=True, exist_ok=True)

# Initialize client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT ---
class RiskCategoryReport(BaseModel):
    category_name: str = Field(description="The risk category name in ALL CAPS (e.g., STAFFING TURNOVER).")
    core_issue: str = Field(description="A summary of the core risks identified in this category in 2 - 3 sentences.")
    operational_impact: str = Field(
        description="An evaluation of the operational consequences and downstream vulnerabilities in 2 - 3 sentences.")
    recommendation: str = Field(description="A concrete, actionable mitigation recommendation in 2 - 3 sentences.")


# Top-level Pydantic model representing the executive report (summary + per-category details)
class ExecutiveRiskReport(BaseModel):
    executive_summary: str = Field(
        description="A professional summary of the findings in 2 to 3 sentences.")
    categories: list[RiskCategoryReport] = Field(
        description="The detailed findings broken down by specific audited risk categories.")

# --- UTILITY TO CAPTURE STDOUT ---
class StreamlitStdoutRedirector:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.output_str = ""

    def write(self, text):
        self.output_str += text
        self.placeholder.code(self.output_str, language="text")

    def flush(self):
        pass

# --- EMBEDDING API CONNECTOR ---
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    cleaned_text = str(text).replace("\n", " ").strip()
    if not cleaned_text:
        cleaned_text = "Empty row placeholder"
    response = client.embeddings.create(
        input=[cleaned_text],
        model=model
    )
    return response.data[0].embedding

# --- MATHEMATICAL SIMILARITY CALCULATIONS ---
def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

# --- FILE TYPE-SPECIFIC CHUNKING FUNCTIONS ---
def chunk_text_file(filepath: Path, chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    text = filepath.read_text(encoding="utf-8")
    words = text.split()
    chunks = []
    start = 0
    filename = filepath.name

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


def chunk_csv_file(filepath: Path, max_field_words: int = 100) -> list[dict]:
    chunks = []
    filename = filepath.name

    with filepath.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            issue_id = row.get("Issue ID", f"Row_{i}")
            category = row.get("Category", "General")
            base_anchor = f"Source: {filename} | ID: {issue_id} | Category: {category}"
            description_text = row.get("Description", "").strip()

            other_fields = " | ".join([f"{k}: {v}" for k, v in row.items()
                                       if k not in ["Issue ID", "Category", "Description"] and v])

            words = description_text.split()

            if len(words) <= max_field_words:
                combined_text = f"{base_anchor} | Description: {description_text}"
                if other_fields:
                    combined_text += f" | {other_fields}"

                chunks.append({
                    "text": combined_text,
                    "source": filename,
                    "chunk_id": f"{filename}_row_{i}"
                })
            else:
                start = 0
                slice_idx = 1
                overlap = 20

                while start < len(words):
                    end = min(start + max_field_words, len(words))
                    text_slice = " ".join(words[start:end])
                    sliced_payload = f"{base_anchor} | Description [Part {slice_idx}]: {text_slice}"
                    if other_fields:
                        sliced_payload += f" | {other_fields}"

                    chunks.append({
                        "text": sliced_payload,
                        "source": filename,
                        "chunk_id": f"{filename}_row_{i}_slice_{slice_idx}"
                    })

                    if end >= len(words):
                        break
                    start = end - overlap
                    slice_idx += 1

    return chunks


def chunk_excel_file(filepath: Path) -> list[dict]:
    chunks = []
    filename = filepath.name
    df = pd.read_excel(filepath)
    df = df.fillna("")

    for index, row in df.iterrows():
        row_dict = row.to_dict()
        row_as_text = " | ".join([f"{col}: {val}" for col, val in row_dict.items() if val != ""])

        chunks.append({
            "text": row_as_text,
            "source": filename,
            "chunk_id": f"{filename}_row_{index + 2}"
        })
    return chunks


def chunk_issue_log(filepath: Path) -> list[dict]:
    text = filepath.read_text(encoding="utf-8")
    raw_records = text.split("Issue ID:")
    chunks = []
    filename = filepath.name

    for i, record in enumerate(raw_records):
        if not record.strip():
            continue

        chunk_text = f"Issue ID:{record.strip()}"
        chunks.append({
            "text": chunk_text,
            "source": filename,
            "chunk_id": f"{filename}_issue_{i}"
        })
    return chunks


def chunk_risk_file(filepath: Path) -> list[dict]:
    chunks = []
    filename = filepath.name

    with filepath.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            row_as_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
            chunks.append({
                "text": row_as_text,
                "source": filename,
                "chunk_id": f"{filename}_row_{i}"
            })
    return chunks


# --- CENTRAL DIRECTORY SCANNER FUNCTION ---
def process_folder(folder_path: Path, chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    registry_filename = "test_risk.txt"
    all_chunks = []

    if not folder_path.exists():
        print(f" -> Directory target folder '{folder_path}' missing. Indexer aborted.")
        return []

    for file_path in folder_path.iterdir():
        if file_path.is_dir():
            continue

        if file_path.name.lower() == registry_filename.lower():
            print(f" -> Skipping risk register file: {file_path.name}")
            continue

        file_chunks = []
        file_name_lower = file_path.name.lower()
        file_suffix = file_path.suffix.lower()

        if file_name_lower == "issue_log.txt":
            print(f" -> Chunking issue log: {file_path.name}")
            file_chunks = chunk_issue_log(file_path)
        elif file_suffix == ".txt":
            print(f" -> Chunking plain-text document: {file_path.name}")
            file_chunks = chunk_text_file(file_path, chunk_size, overlap)
        elif file_suffix == ".csv":
            print(f" -> Chunking comma-separated spreadsheet: {file_path.name}")
            file_chunks = chunk_csv_file(file_path)
        elif file_suffix in [".xlsx", ".xls"]:
            print(f" -> Chunking Excel spreadsheet: {file_path.name}")
            file_chunks = chunk_excel_file(file_path)
        else:
            continue

        all_chunks.extend(file_chunks)

    return all_chunks


# --- VECTOR STORE IMPLEMENTATION ---
class SimpleVectorStore:
    def __init__(self):
        self.entries = []

    def load(self, filepath: Path) -> None:
        if filepath.exists():
            with filepath.open("r", encoding="utf-8") as f:
                self.entries = json.load(f)
            print(f" -> Vector Store loaded {len(self.entries)} entries.")
        else:
            raise FileNotFoundError(f" Vector store file not found.")

    def add_many(self, chunks: list[dict]) -> None:
        print(f" -> Sending {len(chunks)} chunks to OpenAI for vector synthesis...")
        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            self.entries.append({**chunk, "embedding": embedding})

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = get_embedding(query)
        scored = []
        for entry in self.entries:
            sim = cosine_similarity(query_embedding, entry["embedding"])
            scored.append((sim, entry))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [{**entry, "similarity": round(sim, 4)} for sim, entry in scored[:top_k]]

    def save(self, filepath: Path) -> None:
        print(f" -> Saving vector store entries.")
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=4)
            print(f" -> Vector Store file saved as {filepath.name}")


# --- RISK REGISTRY MATCHER ---
class RiskRegistryMatcher:
    def __init__(self, registered_risks_filepath):
        self.register = self._load_register(registered_risks_filepath)

    def _load_register(self, filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()
        return [{"text": line.strip(), "embedding": get_embedding(line.strip())}
                for line in lines if line.strip()]

    def is_unregistered(self, candidate_text, threshold=0.85):
        candidate_embedding = get_embedding(candidate_text)
        for known_risk in self.register:
            score = cosine_similarity(candidate_embedding, known_risk["embedding"])
            if score >= threshold:
                return False, score
        return True, 0.0


def run_risk_audit(store: SimpleVectorStore, matcher: RiskRegistryMatcher, audit_queries: list[str]) -> list[dict]:
    audit_results = []

    for query in audit_queries:
        category_name = query.split(',')[0].strip()
        print(f" -> Scanning Vector Space for Category: '{category_name.upper()}'")

        category_data = {
            "category": category_name,
            "discovered_risks": []
        }

        results = store.search(query, top_k=5)
        new_count = 0
        for match in results:
            is_new, score = matcher.is_unregistered(match["text"]) if matcher else (True, 0.0)

            if is_new:
                new_count += 1
                text = match.get("text", "")
                source_doc = match.get("source", "Unknown Source")

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


# --- REFACTORED REPORT GENERATION VIA PYDANTIC TARGETS ---
def generate_audit_report(structured_report: ExecutiveRiskReport, audit_results: list[dict], file_path: Path) -> str:
    """
    Accepts the validated Pydantic model directly, constructs the text artifact,
    persists it to the file path destination, and returns the final string payload.
    """
    print(f" -> Re-applying structure and saving report to disk.")

    total_categories = len(audit_results)
    total_unregistered_risks = sum(len(section.get("discovered_risks", [])) for section in audit_results)

    lines = []

    # --- REPORT HEADER ---
    lines.append("================================================================================")
    lines.append("RISK AUDIT REPORT")
    lines.append(f"Report Date: {today}")
    lines.append(
        f"Summary: Verified {total_categories} risk categories. Detected {total_unregistered_risks} unregistered anomalies.")
    lines.append("================================================================================")
    lines.append("")

    # --- PARSE STRUCTURED EXECUTIVE CONTENT ---
    lines.append("EXECUTIVE SUMMARY:")
    lines.append(structured_report.executive_summary)
    lines.append("")
    lines.append("DETAILED FINDINGS BY CATEGORY")
    lines.append("")

    for report_item in structured_report.categories:
        lines.append(f"{report_item.category_name}:")
        lines.append(f"  • Core Issue: {report_item.core_issue}")
        lines.append(f"  • Operational Impact: {report_item.operational_impact}")
        lines.append(f"  • Recommendation: {report_item.recommendation}")
        lines.append("")

    final_report_text = "\n".join(lines).strip()

    # Persist artifact
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_report_text)

    return final_report_text


# --- REFACTORED WORKFLOW ROUTER WITH PYDANTIC PARSING ---
def synthesize_report_with_llm(audit_results: list[dict]) -> ExecutiveRiskReport:
    """
    Takes raw unregistered risks and leverages OpenAI's structured output beta mechanics
    to return a strictly checked Pydantic model representation.
    """
    print(" -> Sending raw risk data to OpenAI for structural executive synthesis...")

    raw_context = json.dumps(audit_results, indent=2)

    prompt = f"""
    You are an expert Senior Project Leader. Analyze the following raw data of UNREGISTERED risks 
    discovered during the recent risk audit. 

    Raw Discovered Risk Data:
    {raw_context}
    """

    # We leverage beta.chat.completions.parse to enforce Pydantic parsing at the API wire layer
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a precise, professional Senior Project Leader. Extract risk anomalies into the required schema structure cleanly without markdown text wraps.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=ExecutiveRiskReport,
        temperature=0.3,
    )

    # This returns the initialized ExecutiveRiskReport object directly
    return response.choices[0].message.parsed


# --- CORE PIPELINE EXECUTION WRAPPER ---
def run_automated_pipeline(log_placeholder):
    try:
        print("PIPELINE STARTED.")

        # --- 1. CREATING VECTOR STORE ---
        print("STEP 1: Creating Vector Store.")
        store = SimpleVectorStore()

        # 1. Ensure the directory path exists
        target_dir = database_file_destination.parent
        if not target_dir.exists():
            print(f" -> Creating missing directory: {target_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)

        # 2. Check for the file
        if database_file_destination.exists():
            print(f" -> Found existing vector store: {database_file_destination.name}")
            store.load(database_file_destination)
        else:
            print(" -> No existing vector store found. Starting new ingestion...")
            compiled_data_chunks = process_folder(project_folder, chunk_size=500, overlap=80)

            if not compiled_data_chunks:
                print("No data found to process. Exiting.")
                return

            store.add_many(compiled_data_chunks)
            store.save(database_file_destination)

        # --- 2. Identifying unregistered risks ---
        print(f"STEP 2: Starting automated risk audit.")
        matcher = RiskRegistryMatcher(register_path) if register_path.exists() else None

        if not matcher:
            print(f"Warning: Register not found at {register_path}. Skipping registration check.")

        audit_queries = [
            "staffing turnover, resource departures, personnel shortages",
            "security vulnerabilities, mTLS failures, unauthorized data access",
            "data pipeline errors, system latency, memory leaks, parsing crashes"
        ]

        discovered_risk_data = run_risk_audit(store, matcher, audit_queries)

        # --- 3. Synthesize with LLM ---
        print("STEP 3: Synthesizing AI Report Narrative via Structured Validation.")
        # Returns an actual Pydantic ExecutiveRiskReport object instance now
        structured_report_obj = synthesize_report_with_llm(discovered_risk_data)

        # --- 4. Generate file on disk ---
        print("STEP 4: Generating Structured Risk Audit Report Artifacts.")
        final_report_text = generate_audit_report(structured_report_obj, discovered_risk_data, report_path)

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

st.title("Risk Audit Dashboard")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("System Configuration")
    target_directory = project_root / "project_folder"

    if target_directory.exists():
        st.text(f"Files found in '{target_directory.name}'")
        # iterdir() yields Path objects; we grab .name for just the filename
        files = [f.name for f in target_directory.iterdir()]
        st.write(files)
    else:
        st.error(f"Directory not found")

    start_pipeline = st.button("Generate Risk Audit Report", use_container_width=True, type="primary")

    st.subheader("Pipeline Summary")
    console_logs = st.empty()
    console_logs.info("Click 'Generate Risk Audit Report' button to begin.")

with col2:
    st.subheader("Report Workspace")
    report_placeholder = st.empty()
    report_placeholder.info("The Risk Audit Report will populate here upon synthesis.")

if start_pipeline:
    redirector = StreamlitStdoutRedirector(console_logs)

    with st.spinner("Processing risk parameters..."):
        with contextlib.redirect_stdout(redirector):
            final_narrative = run_automated_pipeline(console_logs)

    if final_narrative:
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