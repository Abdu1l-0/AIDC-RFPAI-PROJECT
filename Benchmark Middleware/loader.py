"""PDF to text.

prepare_document is our own copy of harness/stages/s1_prep.py, so this folder
does not depend on anything outside it.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import pymupdf

DATA_DIR = Path(__file__).resolve().parent / "data"
TEXT_DIR = DATA_DIR / "text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

_ARABIC = re.compile(r"[؀-ۿ]")


def detect_language(text: str) -> str:
    """Rough flag: 'ar' when the text contains Arabic letters, else 'en'."""
    return "ar" if _ARABIC.search(text[:20000]) else "en"


def ensure_text(doc_id: str, pdf_path: str) -> str:
    """Return the document's text, extracting it once and reusing it after."""
    text_path = TEXT_DIR / f"{doc_id}.txt"
    if not text_path.exists():
        prepare_document(pdf_path, str(text_path))
    return text_path.read_text(encoding="utf-8")


def prepare_document(pdf_path: str, output_text_path: str) -> Dict[str, Any]:
    """Extract every page's text into one file, with a '--- Page N ---' header
    before each page, and record where each page starts.

    Returns: {char_count, page_count, offsets: [{page, start_char, length}]}
    """
    doc = pymupdf.open(pdf_path)
    full_text = []
    offsets = []
    curr = 0
    for page_num in range(len(doc)):
        header = f"--- Page {page_num + 1} ---\n"
        text = doc[page_num].get_text() + "\n"
        chunk = header + text
        offsets.append({"page": page_num + 1, "start_char": curr, "length": len(chunk)})
        curr += len(chunk)
        full_text.append(chunk)
    doc.close()

    merged = "".join(full_text)
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(merged)

    return {"char_count": len(merged), "page_count": len(offsets), "offsets": offsets}


def load_pdf(pdf_path: str, doc_id: str) -> dict:
    """Extract text from a PDF, save it to data/text/<doc_id>.txt,
    and flag any pages that came back empty (possible scan/OCR need).

    Returns: {doc_id, pages, chars, warnings}
    """
    text_path = TEXT_DIR / f"{doc_id}.txt"
    result = prepare_document(pdf_path, str(text_path))

    warnings = []
    offsets = result.get("offsets", [])
    with open(text_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    for page_info in offsets:
        page_num = page_info["page"]
        start = page_info["start_char"]
        length = page_info["length"]
        page_text = full_text[start:start + length]
        # drop only the "--- Page N ---" header line, then check what is left
        content_only = page_text.split("\n", 1)[-1].strip()
        if len(content_only) < 10:  # near-empty page: likely scanned or blank
            warnings.append(f"Page {page_num} has little or no extractable text (possible scan)")

    return {
        "doc_id": doc_id,
        "pages": result["page_count"],
        "chars": result["char_count"],
        "warnings": warnings,
    }
