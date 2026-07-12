import os
import re
import json
from llama_cpp import Llama, LlamaGrammar

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
MODEL_PATH = "/app/model/gemma-2-2b-it-Q3_K_L.gguf"

# Strict GBNF Grammar to force structured JSON outputs on math paths
MATH_GRAMMAR = """
root   ::= object
object ::= "{\\n" space "\\"thought\\":\\"" text "\\",\\n" space "\\"expression\\":\\"" expr "\\"\\n}"
expr   ::= [0-9.+\\-*/() ]+
text   ::= [a-zA-Z0-9 .,!?()_\\-]+
space  ::= "  "
"""

def mock_calculator_finisher(expression_str):
    """
    Evaluates the math string deterministically at Python speed layer.
    """
    try:
        clean_expr = re.sub(r'[^0-9.+\-*/() ]', '', expression_str).strip()
        result = eval(clean_expr, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Execution Error: {str(e)}"

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input missing at {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    print("[INIT] Allocating maximized 900-token footprint for strict 4GB limits...")
    
    # FIXED: Scaled up safely to 900 tokens to increase text input processing space.
    # Added type_k and type_v quantization parameters to force 8-bit KV cache structures,
    # cutting cache overhead down significantly and maintaining an absolute safety buffer for Python JSON memory spikes.
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=900, 
        n_threads=2, # Pinned perfectly to align with the 2 vCPU environment layout
        n_batch=32,
        type_k=2,    # Quantize Key cache cells to drop RAM usage
        type_v=2,    # Quantize Value cache cells to drop RAM usage
        verbose=False
    )

    results = []
    print(f"[RUNNING] Processing secure task stream sequentially...")

    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]                                                                                                                                                                                                                                
        prompt_lower = prompt.lower()

        is_math = any(re.search(rf"\b{word}\b", prompt_lower) for word in ["calculate", "total cost", "units remain", "how many units", "equation"])
        
        if is_math:
            is_sentiment = False
            is_summary = False
            is_ner = False
        else:
            is_sentiment = "sentiment" in prompt_lower or "classify" in prompt_lower
            is_summary = "summarize" in prompt_lower
            is_ner = "extract" in prompt_lower or "entities" in prompt_lower

        max_tk = 150
        stop_tokens = ["<end_of_turn>", "<eos>"]
        temp = 0.1
        grammar_argument = None  

        if is_math:
            system_prompt = (
                "You are an algebraic translator engine. Your job is to convert natural language word problems "
                "into structured tool call payloads containing an internal thought scratchpad and a raw mathematical expression.\n"
                "You must strictly output a valid JSON object matching the provided grammar constraints.\n\n"
                "Example Format:\n"
                "{\n"
                "  \"thought\": \"Convert 100 apples minus 20 percent plus 50 into expressions\",\n"
                "  \"expression\": \"(100 * 0.8) + 50\"\n"
                "}"
            )
            max_tk = 180  
            temp = 0.0   
            grammar_argument = LlamaGrammar.from_string(MATH_GRAMMAR)
            
        elif is_sentiment:
            system_prompt = (
                "Classify the sentiment as Positive, Negative, or Neutral. You MUST include a single-sentence reason "
                "that explicitly acknowledges BOTH sides of the review. Do not include any markdown bold formatting like '**Sentiment:**'. "
                "Output format must strictly be: [Sentiment Value] - [One sentence reason]"
            )
            max_tk = 65
            
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
                "Extract all named entities from the text. Output ONLY the raw list points directly, matching this layout style:\n"
                "* [Name of Entity] (PERSON)\n"
                "* [Name of Organization] (ORGANIZATION)\n"
                "* [Name of Location] (LOCATION)\n"
                "* [Date Value] (DATE)\n"
                "Do not write any introductory, greeting, or conversational text before the list items."
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
            echo=False,
            cache_prompt=True,
            grammar=grammar_argument
        )

        answer = output["choices"][0]["text"].strip()
        
        if is_math:
            try:
                math_payload = json.loads(answer)
                extracted_expression = math_payload.get("expression", "")
                final_numeric_calculation = mock_calculator_finisher(extracted_expression)
                answer = final_numeric_calculation
            except Exception:
                answer = "Error parsing structured tool call payload."

        if "Here is" in answer or "Here are" in answer:
            lines = answer.split('\n')
            cleaned_lines = [line for line in lines if not (line.lower().startswith("here is") or line.lower().startswith("here are"))]
            answer = '\n'.join(cleaned_lines).strip()

        results.append({"task_id": task_id, "answer": answer})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"[SUCCESS] Process complete. Output stored in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
