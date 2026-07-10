import os
import json
import time
from llama_cpp import Llama

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
MODEL_PATH = "/app/model/qwen2.5-1.5b-instruct-q4_k_m.gguf"

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input missing at {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    print("[INIT] Allocating Llama-CPP context footprint...")
    
    # n_batch=512 accelerates prompt evaluation processing on restricted CPUs
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=2, 
        n_batch=512,
        verbose=False
    )

    results = []
    print(f"[RUNNING] Processing tasks tracking pool locally...")

    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]                                                             

        # Basic context framing adjustments based on input classification
        math_keywords = ["percent", "%", "calculate", "math", "items", "remain", "sum", "riddle", "total", "legs", "heads"]
        is_computational = any(word in prompt.lower() for word in math_keywords) or "def " in prompt or "function" in prompt.lower()

        # Enforce strict answer-only formatting to avoid wasting time generating explanation fluff
        if is_computational:
            system_prompt = (
                "You are an expert logical engine. Solve the math riddle or debugging problem precisely. "
                "Output ONLY the final numeric answer or the clean corrected code block. Do not explain steps."
            )
        else:
            system_prompt = "Output ONLY a direct, short answer to the question with zero introductory text."

        # Structured chat syntax wrapper template
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # max_tokens=200 provides ample headroom while stop arrays catch early newlines
        output = llm(
            formatted_prompt,
            max_tokens=200,
            stop=["<|im_end|>", "<|endoftext|>\n", "\n\n\n"],
            echo=False
        )

        answer = output["choices"][0]["text"].strip()
        results.append({"task_id": task_id, "answer": answer})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"[SUCCESS] Stored all results safely in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
