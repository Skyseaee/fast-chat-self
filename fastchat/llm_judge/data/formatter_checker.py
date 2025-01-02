import json

def validate_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            try:
                json_obj = json.loads(line)
                print(f"Line {i} is valid: {json_obj}")
            except json.JSONDecodeError as e:
                print(f"Line {i} is invalid: {e}")

validate_jsonl("judge_prompts.jsonl")

