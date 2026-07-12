import os
import json
import time
from llama_cpp import Llama

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
MODEL_PATH = "/app/model/gemma-2-2b-it-Q3_K_L.gguf"

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input missing at {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    print("[INIT] Allocating Llama-CPP context footprint...")
    
    # Lowered n_batch to 32 for optimal dual-core CPU processing latency
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=2, 
        n_batch=32,
        verbose=False
    )

    results = []
    print(f"[RUNNING] Processing tasks tracking pool locally...")

    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]                                                                                                                                                                                                                                
        prompt_lower = prompt.lower()

        # FIXED: Refined math keyword search to prevent misclassifying paragraphs containing "remain"
        is_math = any(word in prompt_lower for word in ["calculate", "sugar", "total cost", "units remain", "how many units"])
        is_sentiment = "sentiment" in prompt_lower or "classify" in prompt_lower
        is_summary = "summarize" in prompt_lower
        is_ner = "extract" in prompt_lower or "entities" in prompt_lower

        # Config baseline defaults
        max_tk = 150
        stop_tokens = ["<end_of_turn>", "<eos>"]
        temp = 0.1

        if is_math:
            # FIXED: Few-shot context tells Gemma exactly how to respond without formatting or math trail leaks
            system_prompt = (
                "You are an exact numerical calculator. Solve the problem step-by-step internally, "
                "but your final output must ONLY contain the raw numbers or values requested. "
                "Completely omit all equations, thought scratchpads, variables, and explanations.\n\n"
                "Example 1:\n"
                "Task: A store has 100 apples. It sells 20%. It gets 50 more. How many remain?\n"
                "Output: 130\n\n"
                "Example 2:\n"
                "Task: A recipe needs 2 cups of flour for 10 pancakes. How much for 25 pancakes? If flour costs $1 per cup, what is the cost?\n"
                "Output: 5 cups, $5.00 total cost\n\n"
                "Follow this exact numerical output behavior pattern for the task below."
            )
            max_tk = 30  
            temp = 0.0   
            stop_tokens = ["<end_of_turn>", "<eos>", "\n"]
            
        elif is_sentiment:
            system_prompt = (
                "Classify the sentiment as Positive, Negative, or Neutral. You MUST include a single-sentence reason "
                "that explicitly acknowledges BOTH sides of the review (e.g., mention shipping/packaging issues AND "
                "the functionality of the device). Do not include any markdown bold formatting like '**Sentiment:**'. "
                "Output format must strictly be: [Sentiment Value] - [One sentence reason]"
            )
            max_tk = 65
            temp = 0.1
            
        elif is_summary:
            if "bullet" in prompt_lower:
                system_prompt = (
                    "Summarize the text. You must output EXACTLY THREE raw markdown bullet points starting directly with '*'. "
                    "Do not provide introductory text, greeting headers, or concluding remarks. "
                    "Each bullet point must contain fewer than 15 words."
                )
                max_tk = 90
            else:
                system_prompt = (
                    "Summarize the text in exactly two sentences. Do not generate a third sentence. "
                    "Start printing the summary sentences directly with no introductory fluff or metadata headers."
                )
                max_tk = 80
                
        elif is_ner:
            system_prompt = (
                "Extract all named entities. Output ONLY the raw list points directly, matching this layout style: "
                "* Sundar Pichai (PERSON)\n"
                "* March 15 2023 (DATE)\n"
                "Do not write any introductory or conversational text before the list items."
            )
            max_tk = 90
        else:
            system_prompt = "Provide a direct, short answer with zero introductory fluff."

        formatted_prompt = (
            f"<start_of_turn>user\n{system_prompt}\n\nTask: {prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        output = llm(
            formatted_prompt,
            max_tokens=max_tk,
            temperature=temp,
            stop=stop_tokens,
            echo=False
        )

        answer = output["choices"][0]["text"].strip()
        
        # Clean up any rare trailing text loops
        if "Here is" in answer or "Here are" in answer:
            lines = answer.split('\n')
            cleaned_lines = [line for line in lines if not (line.lower().startswith("here is") or line.lower().startswith("here are"))]
            answer = '\n'.join(cleaned_lines).strip()

        results.append({"task_id": task_id, "answer": answer})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"[SUCCESS] Stored all results safely in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
