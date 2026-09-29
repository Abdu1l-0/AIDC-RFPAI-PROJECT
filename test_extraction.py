#!/usr/bin/env python3
"""
Send one RFP PDF to a deployed model and extract structured fields.

Usage:
    python3 test_extraction.py --pdf /path/to/rfp.pdf --port 8031 --model "Qwen/Qwen2.5-7B-Instruct-AWQ"

Requires: pip install pypdf requests --break-system-packages
"""

import argparse
import json
import re
import sys

import requests

try:
    from pypdf import PdfReader
except ImportError:
    print("Missing dependency. Run:")
    print("  pip install pypdf --break-system-packages")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# The 17-field extraction schema and prompt
# --------------------------------------------------------------------------- #

SCHEMA = """{
  "submission_deadline": "string (ISO 8601 date+time) or null",
  "inquiry_deadline": "string (ISO 8601 date) or null",
  "rfp_contact": {"name": "string or null", "email": "string or null"},
  "submission_method": "string or null",
  "contract_term": "string or null",
  "scope_of_deliverables": "string or null",
  "mandatory_submission_requirements": ["array of strings"],
  "mandatory_technical_requirements": ["array of strings"],
  "evaluation_criteria": [{"category": "string", "points": "number or null"}],
  "minimum_score_threshold": "string or null",
  "pricing_structure_requirements": "string or null",
  "minimum_insurance_requirements": [{"type": "string", "amount": "string or null"}],
  "required_vendor_experience": "string or null",
  "number_of_references_required": "integer or null",
  "data_security_requirements": "string or null",
  "data_hosting_requirements": "string or null",
  "vendor_demonstration_required": "boolean or null"
}"""

SYSTEM_PROMPT = f"""You are an information-extraction system. You will be given the full text of a
Request for Proposal (RFP) document. Extract exactly the fields defined in the
JSON schema below. Return ONLY valid JSON matching this schema — no
markdown fences, no explanation, no extra text before or after.

Rules:
- If a field is not mentioned anywhere in the document, use null (or an empty
  array [] for list fields with no items) — do not guess or infer.
- Use the meaning of each field, not exact keyword matching. RFPs use
  different terminology for the same concept (e.g. "Closing Date" and
  "Proposals due" both mean submission_deadline).
- Dates must be normalized to ISO 8601 (YYYY-MM-DD, or with time as
  YYYY-MM-DDTHH:MM if a time is specified).
- Keep free-text fields concise — one to three sentences, not a copy-paste of
  the source section.

Schema:
{SCHEMA}"""


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #

def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    if not text.strip():
        print("WARNING: extracted text is empty. This PDF may be scanned/image-based;"
              " OCR would be needed (not handled by this script).")
    return text


# --------------------------------------------------------------------------- #
# Model call
# --------------------------------------------------------------------------- #

def call_model(base_url: str, api_key: str, model: str, rfp_text: str) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"RFP DOCUMENT:\n{rfp_text}\n\nExtract the fields now. Return only the JSON object."},
        ],
        "temperature": 0,
        "max_tokens": 1500,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code != 200:
        print(f'HTTP {resp.status_code} error. Response body:')
        print(resp.text)
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {"raw_content": content, "usage": usage}


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences defensively before parsing."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Test RFP extraction against a live model")
    parser.add_argument("--pdf", required=True, help="Path to the RFP PDF")
    parser.add_argument("--host", default="localhost", help="Model host (default: localhost)")
    parser.add_argument("--port", type=int, default=8031, help="Port-forwarded port (8031=Qwen, 8032=Llama)")
    parser.add_argument("--model", required=True, help="Exact model id, e.g. Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--api-key", help="Bearer key")
    parser.add_argument("--save-output", default=None, help="Optional path to save parsed JSON output")
    args = parser.parse_args()

    base_url = args.host if args.host.startswith("http") else f"http://{args.host}:{args.port}"

    print(f"Extracting text from {args.pdf} ...")
    rfp_text = extract_pdf_text(args.pdf)
    print(f"Extracted {len(rfp_text)} characters.")

    print(f"\nSending to {args.model} at {base_url} ...")
    result = call_model(base_url, args.api_key, args.model, rfp_text)

    print(f"\nToken usage: {result['usage']}")

    try:
        model_output = parse_json_response(result["raw_content"])
    except json.JSONDecodeError as e:
        print(f"\n❌ FAILED TO PARSE MODEL OUTPUT AS JSON: {e}")
        print("Raw content:")
        print(result["raw_content"])
        sys.exit(1)

    print("\n" + "=" * 70)
    print("MODEL OUTPUT")
    print("=" * 70)
    print(json.dumps(model_output, indent=2))

    if args.save_output:
        with open(args.save_output, "w") as f:
            json.dump(model_output, f, indent=2)
        print(f"\nSaved output to {args.save_output}")


if __name__ == "__main__":
    main()
