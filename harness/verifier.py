import os
import json

def verify_run_manifest(run_dir: str) -> bool:
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"FAILED: manifest.json missing at {manifest_path}")
        return False
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    required = ["run_id", "models", "corpus_manifest_hash", "prompt_sha256"]
    missing = [k for k in required if k not in m]
    if missing:
        print(f"FAILED: manifest missing keys: {missing}")
        return False
    print("bench verify: Manifest is valid and complete.")
    return True
