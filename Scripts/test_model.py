from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
from evaluate import load
from tqdm import tqdm

model_path = "saved_models/epochs_100/final_checkpoint"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")

# === Loading test set ===
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

test_data = load_jsonl("final_test_diff_based.jsonl")

# === Setup pipeline ===
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, truncation=True)

# === Predictions ===
results = []
for idx,item in tqdm(enumerate(test_data)):
    input_code = item["src_code"]
    reference_code = item["target"]
    
    prediction = pipe(input_code, max_new_tokens=512)[0]["generated_text"]
    
    results.append({
        "input": input_code,
        "reference": reference_code,
        "prediction": prediction,
        "problem": item["problem_id"]
    })

# === Valutation: Exact Match ===
def exact_match(pred, ref):
    return pred.strip() == ref.strip()

accuracy = sum(exact_match(r["prediction"], r["reference"]) for r in results) / len(results)
print(f"\nExact Match Accuracy: {accuracy:.2%}")

# === Save the results ===
with open("comp_code_opt_100ep_diff_based.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)