
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
from openai import OpenAI
import shutil

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

    # Use a cheap model for judging (route through hybrid agent to keep pipeline consistent)
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


# ---------- OpenAI Responses API helpers ----------

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


def _init_openai_client() -> OpenAI:
    api_key = os.getenv("GPT5_BENCHMARK_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("GPT5_BENCHMARK_OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def reset_vector_stores_and_files(client: OpenAI):
    # Delete vector stores
    try:
        vs_list = client.vector_stores.list()
        for vs in getattr(vs_list, "data", []) or []:
            try:
                client.vector_stores.delete(vs.id)
            except Exception:
                pass
    except Exception:
        pass
    # Delete all files
    try:
        files = client.files.list()
        for f in getattr(files, "data", []) or []:
            try:
                client.files.delete(f.id)
            except Exception:
                pass
    except Exception:
        pass


def create_vector_store_with_files(client: OpenAI, file_paths: list) -> str:
    store = client.vector_stores.create(name=f"boating_benchmark_{datetime.utcnow().isoformat()}Z")
    for path in file_paths:
        with open(path, "rb") as fh:
            up = client.files.create(file=fh, purpose="assistants")
        try:
            client.vector_stores.files.create(vector_store_id=store.id, file_id=up.id)
        except Exception:
            pass
    return store.id


def wait_for_vs_ready(client: OpenAI, vector_store_id: str, timeout_s: int = 60, poll_every_s: float = 3.0):
    import time
    start = time.time()
    seen = set()
    print(f"Waiting for vector store {vector_store_id} to be ready...")
    
    while True:
        try:
            lst = client.vector_stores.files.list(vector_store_id=vector_store_id)
            files = getattr(lst, "data", []) or []
            
            if not files:
                # No files found yet, wait a bit more
                print("No files found in vector store yet, waiting...")
                if time.time() - start > timeout_s:
                    print("Warning: No files found in vector store after timeout; continuing anyway")
                    return
                time.sleep(poll_every_s)
                continue
                
            all_ready = True
            statuses = []
            for vf in files:
                # Check vector store file status first
                status = getattr(vf, "status", None)
                file_id = getattr(vf, "file_id", None) or getattr(vf, "id", None)
                
                # If no status on vector store file, check the underlying file
                if not status and file_id:
                    try:
                        f = client.files.retrieve(file_id)
                        status = getattr(f, "status", None)
                    except Exception:
                        status = "unknown"
                
                statuses.append((file_id, status))
                # Consider "completed", "ready", "processed" as ready states
                if status not in ("completed", "ready", "processed"):
                    all_ready = False
            
            # Print status updates
            for fid, st in statuses:
                key = (fid, st)
                if key not in seen:
                    print(f"Vector store file {fid} status: {st}")
                    seen.add(key)
            
            if all_ready:
                print("All vector store files are ready!")
                return
                
        except Exception as e:
            print(f"Error checking vector store status: {e}")
            # Continue anyway after timeout
            if time.time() - start > timeout_s:
                print("Warning: Error checking vector store status; continuing anyway")
                return
        
        if time.time() - start > timeout_s:
            print("Warning: Vector store indexing timeout; continuing anyway")
            return
            
        time.sleep(poll_every_s)


# ---------- Persistent progress + logging ----------
PROGRESS_FILE = Path(__file__).parent / ".gpt4o_benchmark_progress.json"
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
    log_path = LOGS_DIR / f"gpt4o_benchmark_{ts}.jsonl"
    # persist selection
    save_progress(progress.get("current_index", 0), str(log_path))
    return log_path


def write_log_entry(log_path: Path, entry: dict):
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def ask_openai_gpt4o(client: OpenAI, question: str, vector_store_id: str):
    start_ts = datetime.utcnow().isoformat() + "Z"
    t0 = time.time()
    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": BOATING_BENCHMARK_SYSTEM_PROMPT + "\n\nYou have access to a file_search tool connected to two CSV-derived TXT files. Always use file_search to answer. Retrieve the smallest number of snippets necessary (<=3) and prefer the most relevant rows only."},
            {"role": "user", "content": question},
        ],
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "max_num_results": 4,
        }],
        tool_choice="auto",
        max_output_tokens=200,
        metadata={"benchmark": "boating"},
    )
    # usage tokens
    in_tok = getattr(getattr(resp, "usage", None), "input_tokens", None)
    out_tok = getattr(getattr(resp, "usage", None), "output_tokens", None)

    # extract answer text
    answer = getattr(resp, "output_text", None)
    if not answer:
        try:
            chunks = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    text = getattr(c, "text", None)
                    if text:
                        chunks.append(text)
            answer = "".join(chunks)
        except Exception:
            answer = ""

    # extract retrieved filenames from annotations (if present)
    retrieved_files = []
    try:
        # Find any content entries that include annotations
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                anns = getattr(c, "annotations", None)
                if anns:
                    for ann in anns:
                        fname = getattr(ann, "filename", None)
                        if fname:
                            retrieved_files.append(fname)
    except Exception:
        pass

    # Fallback: look for 'citations' on any content item or top-level output entries
    try:
        for item in getattr(resp, "output", []) or []:
            cits = getattr(item, "citations", None)
            if cits:
                for cit in cits:
                    fname = getattr(cit, "filename", None) or getattr(cit, "file_name", None)
                    if fname:
                        retrieved_files.append(fname)
    except Exception:
        pass

    t1 = time.time()
    end_ts = datetime.utcnow().isoformat() + "Z"
    latency_ms = int((t1 - t0) * 1000)
    return answer, start_ts, end_ts, latency_ms, in_tok, out_tok, retrieved_files

def prepare_row_text_files(csv_path: str, output_dir: str) -> List[str]:
    """Create one .txt file per row from a CSV. Clears output_dir first.
    Returns list of created file paths.
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

def main():
    # Load the data
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

    # split answer-key data into inputs and outputs
    inputs = answer_key["Query with VIN"]
    outputs = answer_key["Answer"]

    # Test each query against the knowledge base
    progress = load_progress()
    log_path = get_or_create_log_file(progress)
    start_index = int(progress.get("current_index", 0))
    correct_count = 0
    total_questions = len(inputs)
    # OpenAI setup; reset and upload files into a fresh vector store
    client = _init_openai_client()
    reset_vector_stores_and_files(client)
    vs_id = create_vector_store_with_files(client, bom_chunk_files + vin_chunk_files)
    
    # Print files.list() and vector store id for confirmation
    files_list = client.files.list()
    print("OpenAI files:", [getattr(f, "filename", None) or getattr(f, "name", None) for f in getattr(files_list, "data", [])])
    print("Vector store id:", vs_id)
    
    # Wait for vector store indexing to complete
    print("Waiting for vector store indexing to complete...")
    wait_for_vs_ready(client, vs_id)
    
    # Print vector store files to verify what the model can see
    vs_files_list = client.vector_stores.files.list(vector_store_id=vs_id)
    print("Vector store files:", [getattr(f, "id", None) for f in getattr(vs_files_list, "data", []) or []])
    print("Vector store ready!")

    print(f"\n{'='*80}")
    print("STARTING BENCHMARK TEST")
    print(f"{'='*80}")

    for abs_i, (question, expected_answer) in enumerate(zip(inputs, outputs)):
        if abs_i < start_index:
            continue
        i = abs_i
        print(f"\n--- Question {i+1}/{total_questions} ---")
        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")

        # Handle NaN values
        if pd.isna(question):
            question = "N/A"
        else:
            question = str(question)

        if pd.isna(expected_answer):
            expected_answer = "N/A"
        else:
            expected_answer = str(expected_answer)

        # Per-question retry wrapper for completion + judging
        attempt = 0
        max_attempts = 3
        judge_result = {"correct": False, "reason": "not evaluated"}
        actual_answer = ""
        while attempt < max_attempts:
            try:
                # Call Responses API with file_search over vector store
                actual_answer, start_ts, end_ts, latency_ms, in_tok, out_tok, retrieved_files = ask_openai_gpt4o(client, question, vs_id)
                print(f"Actual Answer: {actual_answer}")

                # Judge the answer (with internal retries)
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

        # Log everything, including tokens and retrieved filenames
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
            "retrieved_files": retrieved_files,
        }

        print(f"Log Entry: {json.dumps(log_entry, indent=2)}")
        write_log_entry(log_path, log_entry)
        # advance and persist progress after each question
        save_progress(i+1, str(log_path))

        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Final statistics
    accuracy = (correct_count / total_questions) * 100
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
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
    