
'''
NOTE: THIS IS A WORK IN PROGRESS. we have not yet tested the google cloud RAG setup.
'''

import requests
# Removed matplotlib import to avoid NumPy ABI issues; not used in script
import os
import json
import numpy as np
import pandas as pd
import http.client
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import shutil

# TODO: Add Google Cloud imports
# from google.cloud import aiplatform
# from google.cloud import discoveryengine
# import vertexai
# from vertexai.generative_models import GenerativeModel

def reformat_bom_and_vin_table(bom, vin_table):
    # Convert DataFrames to the expected format and handle NaN values
    bom_columns = bom.columns.tolist()
    # Replace NaN with "N/A" and convert everything to strings
    bom_rows = bom.fillna("N/A").astype(str).to_dict(orient="records")
    
    vin_columns = vin_table.columns.tolist()
    vin_rows = vin_table.fillna("N/A").astype(str).to_dict(orient="records")
    
    return bom_columns, bom_rows, vin_columns, vin_rows

def _http_post_json_with_retries(path: str, payload: dict, retries: int = 3, base_sleep: float = 1.0):
    """POST JSON to localhost:8080 and parse JSON response with retries/backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            conn = http.client.HTTPConnection("localhost:8080", timeout=15)
            conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
            response = conn.getresponse()
            raw = response.read()
            conn.close()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} {response.reason}: {raw[:200]!r}")
            return json.loads(raw.decode())
        except Exception as e:
            last_err = e
            # exponential backoff
            time.sleep(base_sleep * (2 ** attempt))
    # If all retries exhausted, raise last error
    raise last_err if last_err else RuntimeError("Unknown HTTP error")


def llm_judge(actual_answer, expected_answer, retries: int = 3):
    """Use a cheap LLM to judge if the answer is correct"""
    
    # Add a system prompt for the judge
    system_prompt = """You are an answer validation judge. Your job is to determine if the actual answer contains the expected answer as a factual claim. You are NOT evaluating the question or context - only whether the expected information is present in the actual answer."""

    judge_prompt = f"""Expected: "{expected_answer}"
        Actual: "{actual_answer}"

        Does the actual answer contain the expected answer? 

        Rules:
        - Look for the expected answer within the actual answer
        - Ignore extra text, formatting, or context.
        - Case insensitive matching
        - Ignore punctuation differences
        - For descriptive text: allow different word orders and phrasing that convey the same meaning

        Return only: {{"correct": true/false, "reason": "found/not found"}}
        """

    # TODO: Replace with Google Generative AI call for judging
    # Keep using the hybrid agent for now, but this could also be converted
    result = _http_post_json_with_retries(
        "/boating_benchmark_agent_hybrid/v1/chat/completions",
        {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": judge_prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        },
        retries=retries,
        base_sleep=1.0,
    )
    
    try:
        # Extract the JSON from the response
        judge_response = result["choices"][0]["message"]["content"]
        # Parse the JSON response
        judge_result = json.loads(judge_response)
        return judge_result
    except:
        # Fallback if JSON parsing fails
        return {"correct": False, "reason": "Failed to parse judge response"}


# ---------- Google Cloud / Vertex AI helpers ----------

# System prompt copied from PromptsUtil.BOATING_BENCHMARK_SYSTEM_PROMPT
BOATING_BENCHMARK_SYSTEM_PROMPT = (
    "## Identity\n"
    "You are a boating agent for Equipade, answering questions about boat components and specifications.\n\n"
    "## Response Rules\n"
    "1. **Answer format**: Provide ONLY the direct answer - no extra explanation or context\n"
    "2. **Units**: Always use the most specific unit available in your data (prefer \"5 days\" over \"1 week\", prefer \"14\" over \"2 weeks\")\n"
    "4. **Precision**: Give exact values from your data - avoid generic defaults like \"2 weeks\"\n"
    "5. **Component matching**: Match the exact component name from the question to your data\n\n"
    "## About Equipade\n"
    "- We build 4 boat models, each identifiable by VIN\n"
    "- We manufacture boats but source components from suppliers. Your questions will be about the components, do not return the boat model or VIN number.\n"
    "- Each model has different component specifications and lead times\n\n"
    "## Reasoning\n"
    "- You will recieve information about a VIN and it's respective model. Example: \"VIN: 1234567890123456, Model: SeaPro 225\"\n"
    "- You will also recieve information about a component and it's respective model. Example: \"Component: Engine, SeaPro 100: N/A, Seapro 150: N/A, SeaPro 175: Yes, SeaPro 225: Yes, Price: 1000, Length: 22ft, ...\"\n"
    "- You will recieve information about multiple components and VINs. It's your job to identify the correct VIN from the user query, fromthe vin get what the model is, and based on the model and component name, identify which component the user is asking about. This is a 2 step process.\n"
    "- You MUST reason appropriately and give the correct answer. Make sure to understand that VINs map to models, and modelse tell us which components are available by appearing as \"Model_Name: Yes\". If they appear as \"Model_Name: N/A\", then the component does not exist for that model and we should disregard it.\n\n"
    "## Example\n"
    "- User: \"Context chunk 1:VIN: 123456789012, Model: SeaPro 225,\n"
    "            Context chunk 2: VIN: 671283797234, Model: SeaPro 150,\n"
    "            Context chunk 3: Component: Engine (Standard), SeaPro 100: Yes, Seapro 150: Yes, SeaPro 175: N/A, SeaPro 225: N/A, Price: 1000, Length: 22ft,\n"
    "            Context chunk 4: Component: Engine (Premium), SeaPro 100: N/A, Seapro 150: N/A, SeaPro 175: Yes, SeaPro 225: Yes, Price: 2000, Length: 22ft,\n"
    "            User query: What is the engine for the SeaPro 225?\"\n"
    "- You: \"2000\"\n"
    "From this example, we can map VIN 1234567890123456 to SeaPro 225, and then map the component Engine (Premium) to SeaPro 225. \n"
    "For the example, 1000 would have been the WRONG answer, since the SeaPro 225 does not have the engine that costs $1000, and we are referring to that SeaPro\n"
    "Assume the SeaPro 225: N/A text means the component does not exist for that model. Components apply to the models that say Yes, such as \"SeaPro 225: Yes\"\n\n"
    "Answer according to the data provided to the best of your ability. Use short answers, do not explain yourself or provide more text around other than the answer.\n"
    "Provide an answer that is copy pasted from the right part of the given context. Do not include the VIN number in the answer. When you don't know the answer for sure, give your best guess\n"
    "Give your answer in the following format:\n"
    "{\n"
    "    \"reasoning\": \"The user is asking about the engine for the SeaPro 225. We can map VIN 1234567890123456 to SeaPro 225, and then map the component Engine (Premium) to SeaPro 225 which has a price of $2000\",\n"
    "    \"answer\": \"2000\"\n"
    "}\n"
)


def _init_vertex_ai():
    """
    TODO: Initialize Vertex AI and Google Cloud clients
    
    This should:
    1. Set up authentication (service account or ADC)
    2. Initialize vertexai with project and location
    3. Create any necessary client objects for Discovery Engine/Search
    
    Example structure:
    - vertexai.init(project=PROJECT_ID, location=LOCATION)
    - return discovery_client, generative_model
    """
    pass


def reset_search_engine_and_documents():
    """
    TODO: Clean up any existing search engine/datastore
    
    This should:
    1. Delete existing search engine/datastore if it exists
    2. Clean up any uploaded documents
    3. Prepare for fresh setup
    
    Note: This might not be necessary depending on how we structure the data store
    """
    pass


def create_search_engine_with_documents(file_paths: list) -> str:
    """
    TODO: Create Vertex AI Search engine and upload documents
    
    This should:
    1. Create a new search app/datastore
    2. Upload the document files (BOM and VIN table text files)
    3. Wait for indexing to complete
    4. Return the search engine/datastore ID
    
    Args:
        file_paths: List of paths to text files to upload
    
    Returns:
        str: Search engine/datastore ID for use in queries
    """
    pass


def wait_for_search_engine_ready(search_engine_id: str, timeout_s: int = 300, poll_every_s: float = 10.0):
    """
    TODO: Wait for search engine indexing to complete
    
    This should:
    1. Poll the search engine status
    2. Wait until all documents are indexed and ready
    3. Print status updates during waiting
    4. Handle timeout gracefully
    
    Args:
        search_engine_id: The search engine ID to monitor
        timeout_s: Maximum time to wait
        poll_every_s: How often to check status
    """
    pass


def ask_vertex_ai_with_rag(question: str, search_engine_id: str):
    """
    TODO: Query Vertex AI with RAG using the search engine
    
    This should:
    1. Use Vertex AI Search to retrieve relevant context
    2. Send the question + context to Generative AI (Gemini)
    3. Parse the response and extract metadata
    4. Return structured response similar to OpenAI version
    
    Args:
        question: The user question
        search_engine_id: The search engine to query for context
    
    Returns:
        Tuple of (answer, start_ts, end_ts, latency_ms, in_tok, out_tok, retrieved_docs)
    """
    start_ts = datetime.utcnow().isoformat() + "Z"
    t0 = time.time()
    
    # TODO: Implement the actual RAG pipeline:
    # 1. Query Vertex AI Search for relevant documents
    # 2. Format the retrieved context
    # 3. Send to Generative AI with system prompt + context + question
    # 4. Parse response and extract answer
    
    # Placeholder return values
    answer = "TODO: Implement Vertex AI RAG query"
    retrieved_docs = []
    in_tok = None  # May not be available depending on API
    out_tok = None
    
    t1 = time.time()
    end_ts = datetime.utcnow().isoformat() + "Z"
    latency_ms = int((t1 - t0) * 1000)
    
    return answer, start_ts, end_ts, latency_ms, in_tok, out_tok, retrieved_docs


def prepare_row_text_files(csv_path: str, output_dir: str) -> List[str]:
    """Create one .txt file per row from a CSV. Clears output_dir first.
    Returns list of created file paths.
    
    NOTE: This function stays the same - it converts CSV rows to individual text files
    which should work well with Vertex AI Search document ingestion
    """
    # Clear and recreate folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.fillna("N/A").astype(str)
    columns = list(df.columns)

    created: List[str] = []
    for idx, row in df.iterrows():
        # Build compact row text: "col1: val1 | col2: val2 | ..."
        parts = [f"{col}: {row[col]}" for col in columns]
        content = " | ".join(parts)

        # Prefer filename with a stable identifier if present
        base_name = None
        for key in ("VIN", "vin", "id", "ID"):
            if key in df.columns:
                base_name = row[key]
                break
        if not base_name:
            base_name = f"row_{idx:06d}"

        safe_name = "".join(c if c.isalnum() or c in ("-","_") else "-" for c in str(base_name))[:64]
        file_path = os.path.join(output_dir, f"{safe_name}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        created.append(file_path)

    return created


# ---------- Persistent progress + logging ----------
# NOTE: These functions stay exactly the same

PROGRESS_FILE = Path(__file__).parent / ".vertex_ai_benchmark_progress.json"
LOGS_DIR = Path(__file__).parent / "logs"


def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_index": 0, "log_file": None}


def save_progress(current_index: int, log_file: Optional[str]):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"current_index": current_index, "log_file": log_file}, f)


def get_or_create_log_file(progress: dict) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if progress.get("log_file"):
        return Path(progress["log_file"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"vertex_ai_benchmark_{ts}.jsonl"
    # persist selection
    save_progress(progress.get("current_index", 0), str(log_path))
    return log_path


def write_log_entry(log_path: Path, entry: dict):
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    """
    Main benchmark function - most of this stays the same, just swap out the 
    knowledge base setup and query functions
    """
    # Load the data (UNCHANGED)
    data_dir = "data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    answer_key = pd.read_csv(os.path.join(data_dir, "answer_key.csv"))
    bom_csv = os.path.join(data_dir, "bom.csv")
    vin_csv = os.path.join(data_dir, "vin_table.csv")
    bom_chunks_dir = os.path.join(data_dir, "bom")
    vin_chunks_dir = os.path.join(data_dir, "vin_table")
    bom_chunk_files = prepare_row_text_files(bom_csv, bom_chunks_dir)
    vin_chunk_files = prepare_row_text_files(vin_csv, vin_chunks_dir)

    # split answer-key data into inputs and outputs (UNCHANGED)
    inputs = answer_key["Query with VIN"]
    outputs = answer_key["Answer"]

    # Test each query against the knowledge base (UNCHANGED STRUCTURE)
    progress = load_progress()
    log_path = get_or_create_log_file(progress)
    start_index = int(progress.get("current_index", 0))
    correct_count = 0
    total_questions = len(inputs)
    
    # TODO: Replace OpenAI setup with Vertex AI setup
    _init_vertex_ai()  # Initialize Vertex AI clients
    reset_search_engine_and_documents()  # Clean up existing resources
    search_engine_id = create_search_engine_with_documents(bom_chunk_files + vin_chunk_files)
    
    print("Search engine id:", search_engine_id)
    
    # Wait for search engine indexing to complete
    print("Waiting for search engine indexing to complete...")
    wait_for_search_engine_ready(search_engine_id)
    print("Search engine ready!")

    print(f"\n{'='*80}")
    print("STARTING VERTEX AI BENCHMARK TEST")
    print(f"{'='*80}")

    for abs_i, (question, expected_answer) in enumerate(zip(inputs, outputs)):
        if abs_i < start_index:
            continue
        i = abs_i
        print(f"\n--- Question {i+1}/{total_questions} ---")
        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")

        # Handle NaN values (UNCHANGED)
        if pd.isna(question):
            question = "N/A"
        else:
            question = str(question)

        if pd.isna(expected_answer):
            expected_answer = "N/A"
        else:
            expected_answer = str(expected_answer)

        # Per-question retry wrapper for completion + judging (UNCHANGED STRUCTURE)
        attempt = 0
        max_attempts = 3
        judge_result = {"correct": False, "reason": "not evaluated"}
        actual_answer = ""
        while attempt < max_attempts:
            try:
                # TODO: Replace OpenAI call with Vertex AI RAG call
                actual_answer, start_ts, end_ts, latency_ms, in_tok, out_tok, retrieved_docs = ask_vertex_ai_with_rag(question, search_engine_id)
                print(f"Actual Answer: {actual_answer}")

                # Judge the answer (with internal retries) - UNCHANGED
                judge_result = llm_judge(actual_answer, expected_answer, retries=3)
                break
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed for question {i+1}: {e}")
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)
                else:
                    judge_result = {"correct": False, "reason": f"error after retries: {e}"}
                    # write log and stop program on hard failure
                    write_log_entry(log_path, {
                        "question_number": i+1,
                        "question": question,
                        "expected_answer": expected_answer,
                        "actual_answer": actual_answer,
                        "judge_result": judge_result,
                        "correct": False,
                        "status": "failed_after_retries"
                    })
                    # do not advance index so we can resume here
                    print("Hard failure encountered, stopping.")
                    return
        print(f"Judge Result: {judge_result}")

        if judge_result["correct"]:
            correct_count += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")

        # Log everything (MOSTLY UNCHANGED - just different metadata)
        log_entry = {
            "question_number": i+1,
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "judge_result": judge_result,
            "correct": judge_result["correct"],
            "timestamps": {"start": start_ts, "end": end_ts},
            "latency_ms": latency_ms,
            "tokens": {"input": in_tok, "output": out_tok},
            "retrieved_docs": retrieved_docs,  # Changed from retrieved_files
        }

        print(f"Log Entry: {json.dumps(log_entry, indent=2)}")
        write_log_entry(log_path, log_entry)
        # advance and persist progress after each question
        save_progress(i+1, str(log_path))

        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Final statistics (UNCHANGED)
    accuracy = (correct_count / total_questions) * 100
    print(f"\n{'='*80}")
    print("VERTEX AI BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_count}")
    print(f"Incorrect Answers: {total_questions - correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}")
    # Reset progress for next clean run
    save_progress(0, None)

if __name__ == "__main__":
    main()