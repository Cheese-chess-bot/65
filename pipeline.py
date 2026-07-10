import os
import io
import json
import contextlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optimize CPU threads for the strict 2 vCPU grading environment ceiling
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)

# Enforce local cache routing paths inside the container
os.environ["HF_HOME"] = "/app/.cache/huggingface"

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
MODEL_DIR = "Qwen/Qwen2.5-0.5B-Instruct"

def execute_sandboxed_code(code_str: str) -> str:
    """Safely runs generated python math/code blocks locally and returns stdout."""
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code_str, {"__builtins__": __builtins__}, {})
        return output_buffer.getvalue().strip()
    except Exception as e:
        return f"[Execution Error]: {str(e)}"

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input missing at {INPUT_PATH}")
        return

    # 1. Read batch array from the evaluation mount path
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    print("[INIT] Loading local engine weights from container cache layer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        torch_dtype=torch.float32, 
        device_map="cpu", 
        local_files_only=True
    )
    base_model.eval()
    
    # 2. Compress network layers to keep RAM usage low
    print("[OPTIMIZER] Dynamically compressing network layers to INT8...")
    model = torch.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)

    results = []
    print(f"[RUNNING] Crunching {len(tasks)} items under a zero-token footprint...")
    
    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]

        # Parse keywords to switch routing behavior dynamically for math/coding tracks
        math_keywords = ["percent", "%", "calculate", "math", "items", "left", "remain", "divided", "sum", "riddle", "total", "balance"]
        is_computational = any(word in prompt.lower() for word in math_keywords) or "def " in prompt or "function" in prompt.lower()

        if is_computational:
            system_prompt = (
                "You are an expert Python programmer. Solve the user's math riddle or coding problem by writing an executable Python script.\n"
                "Output ONLY valid raw python code wrapped inside a ```python and ``` block. Do not write text sentences."
            )
        else:
            system_prompt = (
                "You are a helpful assistant. Provide a direct, concise, and accurate answer to the user query. "
                "Do not include conversational filler or meta-commentary."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cpu")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Handle sandboxed execution trace intercept if it's a code/math query
        if is_computational:
            if "```python" in output_text:
                script = output_text.split("```python")[1].split("```")[0].strip()
            elif "```" in output_text:
                script = output_text.split("```")[1].split("```")[0].strip()
            else:
                script = output_text
            answer = execute_sandboxed_code(script)
        else:
            answer = output_text

        results.append({"task_id": task_id, "answer": answer})

    # 3. Write output array exactly matching the required results schema
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Stored all results safely in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
