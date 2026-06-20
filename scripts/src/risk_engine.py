import csv
import json
import os
import logging
import numpy as np
import pandas as pd  
from openai import OpenAI
from pathlib import Path
from datetime import date

today = date.today().strftime("%B %d, %Y")

# --- PATH CONFIGURATION ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent
log_folder = project_root / "logs"
output_folder = project_root / "outputs"
output_folder.mkdir(exist_ok=True)

# Initialize client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


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

def chunk_text_file(filepath: Path | str, chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    """Reads standard *.txt files and executes sliding window word chunking."""
    # Ensure filepath is a Path object so it works even if a string is passed
    path = Path(filepath)

    # pathlib handles opening, reading, and closing the file automatically
    text = path.read_text(encoding="utf-8")

    words = text.split()
    chunks = []
    start = 0
    filename = path.name  # Replaces os.path.basename()

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
def chunk_csv_file(filepath: Path | str) -> list[dict]:
    """Reads *.csv spreadsheets and maps each line into a row dictionary asset."""
    path = Path(filepath)
    chunks = []
    filename = path.name  # Replaces os.path.basename()

    # path.open() behaves exactly like the built-in open() context manager
    with path.open("r", encoding="utf-8", newline="") as f:
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
def chunk_excel_file(filepath: Path | str) -> list[dict]:
    """Ingests *.xls and *.xlsx spreadsheets using pandas for structural parsing."""
    path = Path(filepath)
    chunks = []
    filename = path.name  # Replaces os.path.basename()

    # Pandas inherently supports pathlib.Path objects natively
    df = pd.read_excel(path)
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
def chunk_issue_log(filepath: Path | str) -> list[dict]:
    path = Path(filepath)

    # Safely opens, reads, and closes the file automatically
    text = path.read_text(encoding="utf-8")

    # Split the file by the unique marker that starts each record
    # This creates a list where each item is one complete issue
    raw_records = text.split("Issue ID:")

    chunks = []
    filename = path.name  # Replaces os.path.basename()

    for i, record in enumerate(raw_records):
        if not record.strip():
            continue  # Skip empty splits

        # Add the marker back so the text makes sense
        chunk_text = f"Issue ID:{record.strip()}"

        chunks.append({
            "text": chunk_text,
            "source": filename,
            "chunk_id": f"{filename}_issue_{i}"
        })

    return chunks

# For risk register files, reads each row as a separate chunk
def chunk_risk_file(filepath: Path | str) -> list[dict]:
    """Reads the risk register file row-by-row to prevent chunking the entire file at once."""
    path = Path(filepath)
    chunks = []
    filename = path.name  # Replaces os.path.basename()

    # path.open() streams the file row-by-row, keeping the memory footprint low
    with path.open("r", encoding="utf-8") as f:
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

def process_folder(folder_path: Path | str, chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    """Scans a specified folder directory path and dynamically routes matched document types."""
    folder = Path(folder_path)
    registry_filename = "test_risk.txt"
    all_chunks = []

    # Replaces os.path.exists()
    if not folder.exists():
        print(f"Directory target folder '{folder}' missing. Indexer aborted.")
        return []

    # iterdir() handles listing directory contents natively
    for file_path in folder.iterdir():
        # Guard clause: skip subdirectories instantly
        if not file_path.is_file():
            continue

        filename_lower = file_path.name.lower()

        # EXCLUSION LOGIC: Skip the register file
        if filename_lower == registry_filename.lower():
            print(f"-> Skipping risk register file: {file_path.name}")
            logging.info(f"Skipping risk register file: {file_path.name}")
            continue

        file_chunks = []

        # Strategic Type Routing Mechanism
        if filename_lower == "test_risk.txt":
            print(f"-> Processing risk register: {file_path.name}")
            logging.info(f"Processing risk register: {file_path.name}")
            file_chunks = chunk_risk_file(file_path)

        elif filename_lower == "issue_log.txt":
            print(f"-> Processing structured issue log: {file_path.name}")
            logging.info(f"Processing structured issue log: {file_path.name}")
            file_chunks = chunk_issue_log(file_path)

        # .suffix extracts extensions like '.txt' or '.csv' safely (includes the dot)
        elif file_path.suffix.lower() == ".txt":
            print(f"-> Processing plain-text document: {file_path.name}")
            logging.info(f"Processing plain-text document: {file_path.name}")
            file_chunks = chunk_text_file(file_path, chunk_size, overlap)

        elif file_path.suffix.lower() == ".csv":
            print(f"-> Processing comma-separated spreadsheet: {file_path.name}")
            logging.info(f"Processing comma-separated spreadsheet: {file_path.name}")
            file_chunks = chunk_csv_file(file_path)

        elif file_path.suffix.lower() in (".xlsx", ".xls"):
            print(f"-> Processing Excel spreadsheet portfolio matrix: {file_path.name}")
            logging.info(f"Processing Excel spreadsheet portfolio matrix: {file_path.name}")
            file_chunks = chunk_excel_file(file_path)

        else:
            # Skip unmapped formats (e.g., pdf, zip, png) silently
            continue

        all_chunks.extend(file_chunks)

    return all_chunks


# --- VECTOR STORE IMPLEMENTATION ---
# This class handles the storage and retrieval of vector embeddings
# It provides methods to add chunks, search for similar chunks, and save the store
Here is the
refactored
SimpleVectorStore


class using pathlib.


For
the
load
method, we
can
use
path.exists() and read
the
JSON
directly
using
json.loads(path.read_text()).For
the
save
method, we
can
write
the
serialized
JSON
data
directly
to
the
file in one
shot
using
path.write_text().

Python
import json
import logging
from pathlib import Path


class SimpleVectorStore:
    def __init__(self):
        self.entries = []

    def load(self, filepath: Path | str) -> None:
        """Loads the vector store entries from a JSON file."""
        path = Path(filepath)

        # Replaces os.path.exists()
        if path.exists():
            # read_text() loads the whole file as a string, which json.loads parses perfectly
            self.entries = json.loads(path.read_text(encoding="utf-8"))
            print(f"Loaded {len(self.entries)} entries from {path}")
        else:
            raise FileNotFoundError(f"Vector store file not found at {path}")

    # Add a single chunk to the store
    def add_many(self, chunks: list[dict]) -> None:
        print()
        print("Creating Vector Store...")
        logging.info("Creating Vector Store...")
        print(f"Indexing {len(chunks)} combined context chunks into system memory space.")
        print("This may take a few moments...")
        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            self.entries.append({**chunk, "embedding": embedding})
        print("Vector API coordination finalized.")

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
    def save(self, filepath: Path | str) -> None:
        """Saves the vector store entries to a JSON file."""
        path = Path(filepath)

        # Use json.dumps to stringify the data, then write it in a single line
        serialized_data = json.dumps(self.entries, indent=4, ensure_ascii=False)
        path.write_text(serialized_data, encoding="utf-8")

# --- RISK REGISTRY MATCHER ---
# This class handles the comparison of candidate risks against the registered risks
# It uses embeddings to determine if a candidate risk is already registered
# It provides a method to check if a candidate risk is unregistered
class RiskRegistryMatcher:
    def __init__(self, registered_risks_filepath: Path | str):
        # Load your master register
        self.register = self._load_register(registered_risks_filepath)

    # Load the registered risks from a file
    def _load_register(self, filepath: Path | str) -> list[dict]:
        path = Path(filepath)

        # read_text() loads the file, splitlines() splits it into a clean list of strings
        # This automatically strips out standard trailing system newline characters (\n or \r\n)
        lines = path.read_text(encoding="utf-8").splitlines()

        # Create embeddings for each known risk
        return [
            {"text": line.strip(), "embedding": get_embedding(line.strip())}
            for line in lines if line.strip()
        ]

    # Check if a candidate risk is unregistered
    def is_unregistered(self, candidate_text: str, threshold: float = 0.85) -> tuple[bool, float]:
        candidate_embedding = get_embedding(candidate_text)

        for known_risk in self.register:
            score = cosine_similarity(candidate_embedding, known_risk["embedding"])
            # If similarity is high, it's already registered
            if score >= threshold:
                return False, score  # Found a match

        return True, 0.0  # No match found, it's unregistered

    
# --- MAIN EXECUTION ---
# This section handles the main execution of the script
# It processes the folder, initializes the tools, and performs the audit
# It writes the results to a file and prints them to the console
# It handles the case where no data is found to process
# It handles the case where the register file is not found
# It handles the case where no unregistered risks are found
# It prints a confirmation message when the report is saved successfully


# --- MAIN EXECUTION ---
def main():
        # --- 1. CONFIGURATION ---
        target_directory = "../project_folder"
        database_file_destination = "global_vector_store.json"
        register_path = os.path.join(target_directory, "test_risk.txt")
        file_path = output_folder / "UNREGISTERED_RISK_DISCOVERY_REPORT.txt"

        # --- 2. INITIALIZE TOOLS ---
        store = SimpleVectorStore()
        matcher = RiskRegistryMatcher(register_path) if os.path.exists(register_path) else None

        # --- 3. DATA INGESTION (Optimized with Lazy Loading) ---
        if os.path.exists(database_file_destination):
            print(f"Found existing vector store: {database_file_destination}. Loading...")
            store.load(database_file_destination)
        else:
            print("No existing vector store found. Starting new ingestion...")
            compiled_data_chunks = process_folder(target_directory, chunk_size=500, overlap=80)

            if not compiled_data_chunks:
                print("No data found to process. Exiting.")
                return

            store.add_many(compiled_data_chunks)
            store.save(database_file_destination)
            print(f"Vector store created and saved to {database_file_destination}")

        if not matcher:
            print(f"Warning: Register not found at {register_path}. Skipping registration check.")
            logging.info(f"Warning: Register not found at {register_path}. Skipping registration check.")

        # --- 4. AUTOMATED AUDIT (Filter for Unregistered Risks) ---
        audit_queries = [
            "staffing turnover, resource departures, personnel shortages",
            "security vulnerabilities, mTLS failures, unauthorized data access",
            "data pipeline errors, system latency, memory leaks, parsing crashes"
        ]

        print(f"\nPerforming automated audit against {len(audit_queries)} risk categories...")
        logging.info(f"Performing automated audit against {len(audit_queries)} risk categories...")

        # Open the file for the entire duration of the audit
        with open(file_path, "w", encoding="utf-8") as f:
            header = f"{'='*80}\nUNREGISTERED RISK DISCOVERY REPORT - {today}\n{'='*80}\n"
            print(header)
            f.write(header)

            unregistered_risks_found = False

            for query in audit_queries:
                query_header = f"\n--- Checking for: {query.split(',')[0].strip()} ---\n"
                print(query_header)
                f.write(query_header)

                results = store.search(query, top_k=5)

                for match in results:
                    is_new, score = matcher.is_unregistered(match["text"]) if matcher else (True, 0.0)

                    if is_new:
                        unregistered_risks_found = True
                        text = match.get("text", "")
                        fields = {part.split(": ")[0].strip(): part.split(": ")[1].strip() 
                                  for part in text.split(" | ") if ": " in part}

                        output_lines = [
                            f"[NEW UNREGISTERED RISK]",
                            f"SOURCE: {match.get('source')}",
                            f"DESCRIPTION: {fields.get('Description', text)}",
                            "-" * 40
                        ]

                        for line in output_lines:
                            print(line)
                            f.write(line + "\n")

            # --- AUDIT COMPLETE (Still inside the 'with' block) ---
            if not unregistered_risks_found:
                msg = "No new unregistered risks were identified."
                print(msg)
                f.write(msg + "\n")

            print("\n" + "="*80)
            print("AUDIT COMPLETE")
            logging.info("AUDIT COMPLETE")
            print(f"Report saved successfully to: '{file_path.name}'")

            f.write("\n" + "="*80 + "\n")
            f.write("AUDIT COMPLETE\n")


    # --- EXECUTE MAIN ---
if __name__ == "__main__":

    # Silence HTTP-level logs from libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    # Configure logging to write to a file
        logging.basicConfig(level=logging.INFO, filename=log_folder/"app.log",
     format="%(asctime)s - %(levelname)s - %(message)s")
        main()