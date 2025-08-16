
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
from typing import Optional

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

        Return only: {{"correct": true/false, "reason": "found/not found"}}"""

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


# ---------- Persistent progress + logging ----------
PROGRESS_FILE = Path(__file__).parent / ".benchmark_progress.json"
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
    log_path = LOGS_DIR / f"hybrid_benchmark_{ts}.jsonl"
    # persist selection
    save_progress(progress.get("current_index", 0), str(log_path))
    return log_path


def write_log_entry(log_path: Path, entry: dict):
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    # Load the data
    data_dir = "data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    answer_key = pd.read_csv(os.path.join(data_dir, "answer_key.csv"))
    bom = pd.read_csv(os.path.join(data_dir, "bom.csv"))
    vin_table = pd.read_csv(os.path.join(data_dir, "vin_table.csv"))

    # split answer-key data into inputs and outputs
    inputs = answer_key["Query with VIN"]
    outputs = answer_key["Answer"]

    # reformat bom and vin table into expected table dto format
    bom_columns, bom_rows, vin_columns, vin_rows = reformat_bom_and_vin_table(bom, vin_table)

    #### SECTION: Refreshes collection ####

    # BOM collection
    # # Delete old collection if it exists
    # conn = http.client.HTTPConnection("localhost:8080")
    # conn.request("DELETE", "/collections/boating_benchmark_collection_hybrid_bom")
    # response = conn.getresponse()
    # print("Delete response:", response.read().decode())
    # conn.close()

    # # Create collection (hybrid)
    # conn = http.client.HTTPConnection("localhost:8080")
    # conn.request("POST", "/collections", json.dumps({
    #     "name": "boating_benchmark_collection_hybrid_bom",
    #     "dimension": 1536
    # }), {"Content-Type": "application/json"})
    # response = conn.getresponse()
    # print("Create collection response:", response.read().decode())
    # conn.close()

    # # Add BOM table (hybrid insert)
    # conn = http.client.HTTPConnection("localhost:8080")
    # conn.request("POST", "/collections/boating_benchmark_collection_hybrid_bom/documents/hybrid", json.dumps({
    #     "type": "table",
    #     "columnNames": bom_columns,
    #     "rows": bom_rows,
    #     "chunkSize": 200,
    #     "overlap": 50
    # }), {"Content-Type": "application/json"})
    # response = conn.getresponse()
    # print("BOM table response:", response.read().decode())
    # conn.close()

    # ## VIN collection
    # # Delete old collection if it exists
    # conn = http.client.HTTPConnection("localhost:8080")
    # conn.request("DELETE", "/collections/boating_benchmark_collection_hybrid_vin")
    # response = conn.getresponse()
    # print("Delete response:", response.read().decode())
    # conn.close()

    # # Create collection (hybrid)
    # conn = http.client.HTTPConnection("localhost:8080")
    # conn.request("POST", "/collections", json.dumps({
    #     "name": "boating_benchmark_collection_hybrid_vin",
    #     "dimension": 1536
    # }), {"Content-Type": "application/json"})
    # response = conn.getresponse()
    # print("Create collection response:", response.read().decode())
    # conn.close()

    # # Add VIN table (hybrid insert)
    # conn = http.client.HTTPConnection("localhost:8080")
    # conn.request("POST", "/collections/boating_benchmark_collection_hybrid_vin/documents/hybrid", json.dumps({
    #     "type": "table",
    #     "columnNames": vin_columns,
    #     "rows": vin_rows,
    #     "chunkSize": 200,
    #     "overlap": 50
    # }), {"Content-Type": "application/json"})
    # response = conn.getresponse()
    # print("VIN table response:", response.read().decode())
    # conn.close()

    #### SECTION END: Refreshes collection ####

    # Test each query against the knowledge base
    progress = load_progress()
    log_path = get_or_create_log_file(progress)
    start_index = int(progress.get("current_index", 0))
    correct_count = 0
    total_questions = len(inputs)
    
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
                # Get response from chat completion (hybrid agent)
                result = _http_post_json_with_retries(
                    "/boating_benchmark_agent_hybrid/v1/chat/completions-with-retrieved-chunks",
                    {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 200,
                        "temperature": 0.0
                    },
                    retries=3,
                    base_sleep=1.0,
                )
                completion = result.get("completion", {})
                actual_answer = completion.get("choices", [{}])[0].get("message", {}).get("content", "")
                retrieved_chunks = result.get("retrievedChunks", [])
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
        
        # Log everything
        log_entry = {
            "question_number": i+1,
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "judge_result": judge_result,
            "correct": judge_result["correct"],
            "retrieved_chunks": retrieved_chunks,
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
    









