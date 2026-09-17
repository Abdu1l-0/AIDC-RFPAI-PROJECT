import json
import hashlib
from typing import Dict, Any

def assemble_prompt(prepared_text: str, field_set_path: str) -> Dict[str, Any]:
    with open(field_set_path, "r", encoding="utf-8") as f:
        field_set = json.load(f)
    fields_guide = []
    for fid, fdef in field_set["fields"].items():
        fields_guide.append(f"{fid}: {fdef['name']} - {fdef['description']} (Type: {fdef['type']})")
    system_inst = (
        "You are an expert procurement and contract analyst. Extract specified fields from the RFP.\n"
        "RULES:\n"
        "1. Return EXACTLY ONE JSON object with keys F01 through F17.\n"
        "2. For each field: {\"status\": \"found\" | \"not_stated\", \"value\": <extracted_val_or_null>, \"evidence\": [{\"quote\": \"exact text sentence\", \"page\": <int_or_null>}]}.\n"
        "3. If a field is absent, set \"status\": \"not_stated\", \"value\": null, \"evidence\": []. Do NOT fabricate.\n"
        "4. Strict JSON output only. No markdown formatting, no commentary."
    )
    user_prompt = (
        "FIELDS TO EXTRACT:\n" + "\n".join(fields_guide) + "\n\n"
        "RFP TEXT:\n" + prepared_text + "\n\n"
        "Output extracted JSON object:"
    )
    messages = [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": user_prompt}
    ]
    rendered_str = json.dumps(messages, sort_keys=True)
    sha = hashlib.sha256(rendered_str.encode("utf-8")).hexdigest()
    return {"messages": messages, "prompt_sha256": sha}
