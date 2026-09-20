import json
import hashlib
from pathlib import Path

from fields import FIELDS

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "extraction_prompt.txt"


def build_messages(doc_text: str) -> dict:
    """Build the chat messages for extraction.

    Loads the fixed system instructions from prompts/extraction_prompt.txt,
    combines them with the field list (from fields.py, which re-keys
    config/field_set@1.0.json by readable name) and the document text,
    to build the messages sent to each model.

    Returns: {"messages": [...], "prompt_sha256": "..."}
    """
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_inst = f.read().strip()

    fields_guide = []
    for name, fdef in FIELDS.items():
        fields_guide.append(f"{name}: {fdef['name']} - {fdef['description']} (Type: {fdef['type']})")

    user_prompt = (
        "FIELDS TO EXTRACT:\n" + "\n".join(fields_guide) + "\n\n"
        "RFP TEXT:\n" + doc_text + "\n\n"
        "Output extracted JSON object:"
    )

    messages = [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": user_prompt},
    ]

    rendered_str = json.dumps(messages, sort_keys=True)
    prompt_sha256 = hashlib.sha256(rendered_str.encode("utf-8")).hexdigest()

    return {"messages": messages, "prompt_sha256": prompt_sha256}