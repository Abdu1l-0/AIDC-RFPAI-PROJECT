import os
import pymupdf as fitz
from typing import Dict, Any

def prepare_document(pdf_path: str, output_text_path: str) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
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
