import argparse
import os
import sys
import json
import time
import hashlib
from pathlib import Path
from harness.stages.s0_register import register_corpus
from harness.stages.s1_prep import prepare_document
from harness.stages.s2_prompt import assemble_prompt
from harness.stages.s4_parse import parse_and_validate
from harness.stages.s5_repair import bounded_reask
from harness.stages.s6_normalize import Normalizer
from harness.verifier import verify_run_manifest
from harness.adapters.openai_adapter import OpenAIAdapter
from harness.adapters.vllm_adapter import VLLMAdapter
def get_default_config_paths():
    repo_root = Path(__file__).resolve().parent.parent
    field_set = repo_root / "config" / "field_set@1.0.json"
    normalizers = repo_root / "config" / "normalizers.json"
    return str(field_set), str(normalizers)
def run_command(args):
    model = args.model
    adapter_name = args.adapter.lower() if args.adapter else ("openai" if ("gpt" in model or "o1" in model or "o3" in model) else "vllm")
    input_path = args.input or args.corpus
    out_dir = args.output or args.out
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    field_set_path, normalizers_path = get_default_config_paths()
    if not os.path.exists(field_set_path):
        print(f"Error: field set not found at {field_set_path}. Create config/field_set@1.0.json first.")
        sys.exit(1)
    if not os.path.exists(normalizers_path):
        print(f"Error: normalizers not found at {normalizers_path}. Create config/normalizers.json first.")
        sys.exit(1)
    normalizer = Normalizer(normalizers_path)
    if adapter_name == "openai":
        if not api_key:
            print("=" * 60)
            print("ERROR: OPENAI_API_KEY is not set.")
            print("Please export your OpenAI API key before running:")
            print("  export OPENAI_API_KEY='sk-...'")
            print("Or pass it via the CLI: --api-key <YOUR_KEY>")
            print("=" * 60)
            sys.exit(1)
        adapter = OpenAIAdapter(model_key=model, api_key=api_key, model_name=model, base_url=args.base_url)
    elif adapter_name in ["vllm", "local"]:
        adapter = VLLMAdapter(model_key=model, endpoint_url=base_url, model_name=model)
    else:
        print(f"Error: Unknown adapter '{adapter_name}'.")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    attempts_dir = os.path.join(out_dir, "attempts")
    os.makedirs(attempts_dir, exist_ok=True)
    prep_dir = os.path.join(out_dir, "prepared")
    os.makedirs(prep_dir, exist_ok=True)
    docs_to_process = []
    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        doc_id = Path(input_path).stem.replace(" ", "_")
        docs_to_process.append((doc_id, input_path))
        manifest = {"documents": {doc_id: {"filename": os.path.basename(input_path)}}}
    elif os.path.isdir(input_path):
        manifest_file = os.path.join(out_dir, "corpus_manifest.json")
        manifest = register_corpus(input_path, manifest_file)
        for doc_id, meta in manifest["documents"].items():
            in_pdf = os.path.join(input_path, meta["filename"])
            docs_to_process.append((doc_id, in_pdf))
    else:
        print(f"Error: Input '{input_path}' not found.")
        sys.exit(1)
    print("=" * 60)
    print(f"Running Benchmark Extraction: {model} ({adapter_name})")
    print(f"Found {len(docs_to_process)} document(s). Output: {out_dir}")
    print("=" * 60)
    run_id = f"run_{int(time.time())}_{model.replace('/', '_')}"
    last_prompt_sha = ""
    total_p_tok = 0
    total_c_tok = 0
    success_count = 0
    decoding_params = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    for idx, (doc_id, pdf_path) in enumerate(docs_to_process, 1):
        txt_path = os.path.join(prep_dir, f"{doc_id}.txt")
        if not os.path.exists(txt_path):
            print(f"[{idx}/{len(docs_to_process)}] Extracting text from {os.path.basename(pdf_path)}...")
            prepare_document(pdf_path, txt_path)
        with open(txt_path, "r", encoding="utf-8") as f:
            doc_text = f.read()
        prompt_res = assemble_prompt(doc_text, field_set_path)
        last_prompt_sha = prompt_res["prompt_sha256"]
        print(f"[{idx}/{len(docs_to_process)}] Calling OpenAI on {doc_id}...", end="", flush=True)
        res = adapter.invoke(prompt_res["messages"], decoding_params)
        raw_text = res.get("response", {}).get("raw_text", "")
        usage = res.get("response", {}).get("usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        total_p_tok += p_tok
        total_c_tok += c_tok
        repair_level, parsed_json, parse_err = parse_and_validate(raw_text)
        if repair_level == "failed" and res["status"] == "ok":
            l2_res = bounded_reask(adapter, prompt_res["messages"], raw_text, parse_err, decoding_params)
            r2, p2, e2 = parse_and_validate(l2_res.get("response", {}).get("raw_text", ""))
            if r2 != "failed":
                repair_level = "L2"
                parsed_json = p2
        norm = normalizer.normalize_record(parsed_json) if repair_level != "failed" else {}
        attempt = {
            "attempt_id": f"att_{doc_id}",
            "run_id": run_id,
            "model_key": model,
            "doc_id": doc_id,
            "repeat_idx": 0,
            "status": res["status"],
            "request": {"prompt_sha256": prompt_res["prompt_sha256"], "decoding": decoding_params},
            "response": res.get("response", {}),
            "timings": res.get("timings", {}),
            "parsed": {"repair_level": repair_level, "extracted_fields": norm}
        }
        with open(os.path.join(attempts_dir, f"{doc_id}.json"), "w", encoding="utf-8") as af:
            json.dump(attempt, af, indent=2)
        ms = res.get("timings", {}).get("ms_total", 0.0)
        if res["status"] == "ok":
            success_count += 1
            print(f" Done ({ms:.0f}ms | in: {p_tok} | out: {c_tok} | repair: {repair_level})")
        else:
            print(f" FAILED ({res['status']}: {raw_text[:60]})")
    manifest_data = {
        "run_id": run_id,
        "models": [model],
        "corpus_manifest_hash": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "prompt_sha256": last_prompt_sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)
    print("=" * 60)
    print(f"Run Complete: {success_count}/{len(docs_to_process)} successful.")
    print(f"Tokens: Prompt={total_p_tok:,}, Completion={total_c_tok:,}")
    print(f"Saved: {attempts_dir}")
    print("=" * 60)
def main():
    parser = argparse.ArgumentParser(description="RFP Benchmark Runner CLI")
    subparsers = parser.add_subparsers(dest="command")
    prep_parser = subparsers.add_parser("prepare")
    prep_parser.add_argument("--corpus", "--input", required=True)
    prep_parser.add_argument("--out", "--output", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--adapter", choices=["openai", "vllm", "local"], default=None)
    run_parser.add_argument("--input", "--corpus", required=True)
    run_parser.add_argument("--output", "--out", required=True)
    run_parser.add_argument("--api-key", default=None)
    run_parser.add_argument("--base-url", default=None)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--max-tokens", type=int, default=2048)
    run_parser.add_argument("--repeats", type=int, default=1)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_command(args)
    elif args.command == "verify":
        if not verify_run_manifest(args.run_dir):
            sys.exit(1)
    else:
        parser.print_help()
if __name__ == "__main__":
    main()