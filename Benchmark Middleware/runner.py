import asyncio
from llm_client import call_model, MODEL_CONFIG
from validate import parse_model_output
from storage import save_result
from scoring import load_ground_truth, score_document


async def _run_one_model(doc_id: str, model_key: str, text: str) -> dict:
    """Call one model, validate its output, score it if ground truth
    exists, save the result, return it."""
    raw_result = await asyncio.to_thread(call_model, model_key, text)

    if raw_result["status"] == "ok":
        parsed = parse_model_output(raw_result["raw_text"])
    else:
        parsed = {"valid": False, "fields": None, "error": raw_result["error"]}

    score = None
    if parsed["valid"] and parsed["fields"] is not None:
        gold = load_ground_truth(doc_id)
        if gold is not None:
            predicted_dict = parsed["fields"].model_dump()
            gold_fields = gold.get("fields", gold)
            score_result = score_document(predicted_dict, gold_fields, text)
            score = score_result["macro_accuracy"]

    result = {
        "doc_id": doc_id,
        "model": model_key,
        "status": raw_result["status"],
        "latency_s": raw_result["latency_s"],
        "input_tokens": raw_result["input_tokens"],
        "output_tokens": raw_result["output_tokens"],
        "valid_json": parsed["valid"],
        "fields": parsed["fields"].model_dump() if parsed["fields"] else None,
        "score": score,
        "error": raw_result["error"] or parsed.get("error"),
    }

    save_result(doc_id, model_key, result)
    return result


async def run_document(doc_id: str, text: str, models: list[str] = None) -> dict:
    """Run one document across the chosen models (default: all 3), in parallel.

    Returns: {"doc_id": doc_id, "results": [result, result, result]}
    """
    if models is None:
        models = list(MODEL_CONFIG.keys())

    tasks = [_run_one_model(doc_id, model_key, text) for model_key in models]
    results = await asyncio.gather(*tasks)

    return {"doc_id": doc_id, "results": list(results)}