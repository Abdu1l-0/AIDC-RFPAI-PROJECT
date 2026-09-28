import json
import hashlib
import re
from pathlib import Path

from fields import FIELDS, value_shape

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PATH = PROMPTS_DIR / "extraction_prompt.txt"   # base rules, same for every model
SHAPE_RULES_PATH = PROMPTS_DIR / "shape_rules.txt"   # extra rules, only when shapes are in the prompt

# Recorded with every result, so the report can say which form each model got.
VARIANT_ENFORCED = "schema_enforced"      # answer shape enforced by the endpoint
VARIANT_SHAPES = "shapes_in_prompt"       # answer shape written into the prompt


def _read(path: Path) -> str:
    """Read a prompt file with line endings normalized to plain newlines.

    Git can check files out with Windows (CRLF) or Unix (LF) line endings
    depending on the machine. Without this, the same prompt would differ -
    and get a different prompt_sha256 - between two team members' laptops.
    """
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()


ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")


def _compact(text: str) -> str:
    """Squeeze runs of whitespace out of the document text.

    PDF extraction leaves long runs of spaces and blank lines, and whitespace
    tokenizes badly - about 850 characters per Arabic RFP, roughly 340 tokens.
    Safe for scoring: check_grounding() normalizes whitespace on both the quote
    and the source before comparing, so collapsing it here cannot break a match.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _language_note(doc_text: str) -> str:
    """A concrete language instruction, placed LAST in the user message.

    Rule 5 alone ("answer in the language of the RFP") did not work on the 8B
    model: every rule, field name and shape around it is English, and the last
    line before generation is English too, so it answered in English and even
    quoted the English field descriptions back as evidence. Naming the language
    outright, immediately before generation, is what the smaller model follows.
    """
    if len(ARABIC_RANGE.findall(doc_text[:20000])) < 100:
        return ""
    return (
        "IMPORTANT - LANGUAGE: the RFP TEXT above is written in Arabic.\n"
        "Write every text value and every \"quote\" in ARABIC, copied from the\n"
        "RFP TEXT above. Do not translate anything into English. Never quote\n"
        "these instructions or the field descriptions - every quote must come\n"
        "from the RFP TEXT itself.\n\n"
    )


def build_messages(doc_text: str, shapes_in_prompt: bool = False) -> dict:
    """Build the chat messages for extraction.

    Every model gets the same base rules (prompts/extraction_prompt.txt), the
    same 17 field names and descriptions, and the same document text.

    The only difference is how the exact answer SHAPE reaches the model:
      shapes_in_prompt=False  the endpoint enforces our JSON schema itself
                              (OpenAI json_schema), so the prompt stays short.
      shapes_in_prompt=True   the endpoint cannot enforce it (our vLLM), so the
                              shape of every field is written into the prompt,
                              plus prompts/shape_rules.txt.

    Tested 2026-09-22 on the ERP RFP: shapes in the prompt are what make the
    vLLM model answer in the right format, but they cost gpt-4o-mini ~0.16
    accuracy (0.80 -> 0.64). No single prompt suited both, so each model gets
    the shape through the channel its endpoint supports.

    Returns: {"messages": [...], "prompt_sha256": "...", "prompt_variant": "..."}
    """
    system_inst = _read(PROMPT_PATH)

    if shapes_in_prompt:
        system_inst += "\n" + _read(SHAPE_RULES_PATH)
        # Every field shows the COMPLETE answer, wrapper included. Showing only
        # the inner value made the smaller model return a bare list for
        # list-valued fields and drop the {status, value, evidence} wrapper.
        fields_guide = [
            f"{name}: {fdef['name']} - {fdef['description']}\n"
            f'  answer as: {{"status":"found"|"not_stated",'
            f'"value":{value_shape(name)}|null,'
            f'"evidence":[{{"quote":"exact text","page":int}}]}}'
            for name, fdef in FIELDS.items()
        ]
    else:
        fields_guide = [
            f"{name}: {fdef['name']} - {fdef['description']} (Type: {fdef['type']})"
            for name, fdef in FIELDS.items()
        ]

    user_prompt = (
        "FIELDS TO EXTRACT:\n" + "\n".join(fields_guide) + "\n\n"
        "RFP TEXT:\n" + _compact(doc_text) + "\n\n"
        + _language_note(doc_text) +
        "Output extracted JSON object:"
    )

    messages = [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": user_prompt},
    ]

    rendered_str = json.dumps(messages, sort_keys=True)
    prompt_sha256 = hashlib.sha256(rendered_str.encode("utf-8")).hexdigest()

    return {
        "messages": messages,
        "prompt_sha256": prompt_sha256,
        "prompt_variant": VARIANT_SHAPES if shapes_in_prompt else VARIANT_ENFORCED,
    }
