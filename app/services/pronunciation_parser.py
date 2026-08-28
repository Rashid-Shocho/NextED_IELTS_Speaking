"""
Receiver/parser for GOP (Goodness of Pronunciation) output -- decoupled
from RunPod entirely so it can be built and tested independently of
network/deployment issues.

Input can be ANY of:
  - a raw text blob (the notebook's printed output: lines of
    "{'phoneme': 'x', 'score': -0.726}", optionally with a
    "Utterance score: ..." line and/or "Question N:" headers)
  - an already-parsed list[dict] of {"phoneme":..., "score":...}
  - a dict/JSON blob with the phoneme list nested somewhere inside it
    (for whatever shape the real RunPod handler eventually returns)

Output is a single compact dict ready to hand to llm_scorer.py:
  - total_phonemes / utterance_avg: overall stats
  - distribution: excellent/good/moderate/severe counts
  - severe_flags: DEDUPED list of severely mispronounced phonemes,
    worst-first, each with how many times it recurred
  - worst_phoneme: the single worst-scoring phoneme in the whole
    utterance, regardless of the severe threshold -- always populated
    if there's any data, so the LLM can always call out one specific
    mistake even in an otherwise decent answer.
"""

import re

SEVERE_THRESHOLD = -3.0
MODERATE_THRESHOLD = -1.5
GOOD_THRESHOLD = -0.5

_PHONEME_LINE_RE = re.compile(
    r"\{'phoneme':\s*'([^']*)',\s*'score':\s*(-?\d+\.?\d*)\}"
)


def _parse_raw_text(text: str) -> list[dict]:
    """Extracts {'phoneme': ..., 'score': ...} lines from raw notebook-
    style text output. Ignores everything else (warnings, headers,
    'Utterance score:' summary lines) automatically since only matching
    lines are captured."""
    matches = _PHONEME_LINE_RE.findall(text)
    return [{"phoneme": p, "score": float(s)} for p, s in matches]


def _find_phoneme_list(obj) -> list | None:
    """Recursively searches a dict/list structure for the first list of
    {'phoneme':..., 'score':...} dicts -- used when input is JSON rather
    than raw text, since the real RunPod handler's exact key names aren't
    confirmed yet."""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "phoneme" in obj[0] and "score" in obj[0]:
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_phoneme_list(value)
            if found is not None:
                return found
    return None


def _normalize_input(raw) -> list[dict]:
    """Accepts str, list, or dict and always returns a flat list of
    {'phoneme': str, 'score': float} dicts."""
    if isinstance(raw, str):
        return _parse_raw_text(raw)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        found = _find_phoneme_list(raw)
        return found or []
    return []


def analyze_gop_result(raw) -> dict:
    """
    The main receiver function. Takes GOP output in any of the shapes
    described above and returns a compact, deduped, LLM-ready summary.
    """
    phoneme_data = _normalize_input(raw)
    total = len(phoneme_data)

    if total == 0:
        return {
            "utterance_avg": None,
            "total_phonemes": 0,
            "distribution": {"excellent": 0, "good": 0, "moderate": 0, "severe": 0},
            "severe_flags": [],
            "worst_phoneme": None,
        }

    scores = [item["score"] for item in phoneme_data]
    avg_score = sum(scores) / total

    excellent = sum(1 for s in scores if s >= GOOD_THRESHOLD)
    good = sum(1 for s in scores if MODERATE_THRESHOLD <= s < GOOD_THRESHOLD)
    moderate = sum(1 for s in scores if SEVERE_THRESHOLD <= s < MODERATE_THRESHOLD)
    severe_items = [item for item in phoneme_data if item["score"] < SEVERE_THRESHOLD]

    # --- Dedupe severe phonemes: keep worst score + how many times each recurred ---
    by_phoneme: dict[str, dict] = {}
    for item in severe_items:
        p = item["phoneme"]
        if p not in by_phoneme or item["score"] < by_phoneme[p]["worst_score"]:
            by_phoneme.setdefault(p, {"worst_score": item["score"], "count": 0})
            by_phoneme[p]["worst_score"] = min(by_phoneme[p]["worst_score"], item["score"])
        by_phoneme[p]["count"] += 1

    severe_flags = sorted(
        [{"phoneme": p, "worst_score": round(v["worst_score"], 3), "count": v["count"]}
         for p, v in by_phoneme.items()],
        key=lambda x: x["worst_score"],
    )

    # --- The single worst phoneme in the entire utterance (always populated) ---
    worst_item = min(phoneme_data, key=lambda item: item["score"])
    worst_phoneme = {"phoneme": worst_item["phoneme"], "score": round(worst_item["score"], 3)}

    return {
        "utterance_avg": round(avg_score, 3),
        "total_phonemes": total,
        "distribution": {
            "excellent": excellent,
            "good": good,
            "moderate": moderate,
            "severe": len(severe_items),
        },
        "severe_flags": severe_flags,
        "worst_phoneme": worst_phoneme,
    }