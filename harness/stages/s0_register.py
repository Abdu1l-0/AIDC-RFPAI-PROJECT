import os
import hashlib
import json
import fitz
from typing import Dict, Any

def register_corpus(corpus_dir: str, output_manifest_path: str) -> Dict[str, Any]:
    manifest = {"version": "1.0.0", "documents": {}}
    files = sorted([f for f in os.listdir(corpus_dir) if f.lower().endswith(".pdf")])
    for idx, fname in enumerate(files, 1):
        doc_id = f"DOC_{idx:02d}"
        fpath = os.path.join(corpus_dir, fname)
        with open(fpath, "rb") as f:
            data = f.read()
            sha256 = hashlib.sha256(data).hexdigest()
            byte_size = len(data)
        doc = fitz.open(fpath)
        page_count = len(doc)
        doc.close()
        manifest["documents"][doc_id] = {
            "doc_id": doc_id,
            "filename": fname,
            "sha256": sha256,
            "byte_size": byte_size,
            "page_count": page_count
        }
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
