import asyncio
import json
import re
import httpx
from app.core.config import get_settings
from app.services.band_descriptors import format_descriptor_block

settings = get_settings()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _close_open_brackets(candidate: str) -> str:
    """Walks a possibly-truncated/malformed JSON fragment (respecting
    string/escape state so brackets inside quoted text aren't counted) and
    appends whatever closing quote/]/} characters are needed to balance
    everything still open. Used to salvage a response that got cut off, or
    that went wrong partway through (e.g. the model closed an object with
    ']' instead of '}'), rather than losing the whole response to one bad
    character."""
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    repaired = candidate
    if in_string:
        repaired += '"'
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def extract_json(text: str) -> dict:
    """Very tolerant JSON extractor.

    gpt-oss-20b via Groq occasionally emits JSON that's syntactically
    broken in ways that aren't just "got cut off at the end" -- e.g. a
    nested object closed with ']' instead of '}' partway through a long
    array (seen in production on the text_analysis pass), or the whole
    response missing its final closing '}' despite finish_reason not
    being "length" (seen on final_scoring). A single-shot direct parse
    fails on both, so on failure this walks the JSONDecodeError's exact
    error position, cuts the string there, and re-closes whatever
    strings/arrays/objects were still open at that point -- recovering
    everything up to the break instead of discarding the whole response.
    """
    text = text.strip()

    # Remove thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove markdown
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first '{' onward (handles leading prose before the JSON),
    # up through the largest possible closing '}' if there is one.
    match = re.search(r"\{[\s\S]*\}", text)
    candidate = match.group(0) if match else text[text.find("{"):] if "{" in text else text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Cut at the exact point parsing broke, trim any dangling trailing
        # comma/colon left right at that cut, then close whatever was
        # still open. Covers both "ran out of tokens mid-value" (cut
        # point = end of string, nothing lost) and "malformed/wrong token
        # mid-structure" (cut point = the bad token, so only content after
        # it is lost, not the whole response).
        truncated = re.sub(r'[,:\s]+$', "", candidate[: e.pos])
        repaired = _close_open_brackets(truncated)
        try:
            return json.loads(repaired)
        except Exception:
            pass

    raise ValueError("Could not extract valid JSON from model response")


async def _call_groq_json(
    system_prompt: str, user_prompt: str, label: str, max_tokens: int = 2000,
    response_schema: dict | None = None, _is_retry: bool = False,
) -> dict:
    """
    Shared plumbing for every scoring pass: call Groq, log input/output,
    parse JSON. Raises on any failure (after one retry for 429s) --
    callers decide how to degrade.

    response_schema, when given, is sent as a strict-mode JSON Schema via
    `response_format` -- Groq's docs confirm openai/gpt-oss-20b (our
    GROQ_LLM_MODEL) supports strict-mode structured outputs, which use
    constrained decoding to guarantee the response matches the schema
    exactly. This is the real fix for the malformed-JSON failures seen in
    production (a nested object closed with ']' instead of '}', a
    response missing its final '}') -- those were the model free-forming
    JSON as text, not a parsing bug on our end. extract_json's repair
    logic stays as a defense-in-depth fallback in case a future model
    swap loses strict-mode support or Groq's guarantee has an edge case.
    """
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.GROQ_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        # gpt-oss models "think" internally before producing visible output,
        # and that reasoning consumes completion tokens from the same
        # max_tokens budget as the final JSON -- at the default "medium"
        # effort this can burn the whole budget on reasoning and truncate
        # before any JSON is ever emitted (seen in production: empty
        # response + finish_reason=length). "low" leaves enough budget for
        # actual output on tasks this size, and meaningfully cuts total
        # token usage across the 3 sequential passes per session.
        "reasoning_effort": "low",
    }
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": label,
                "strict": True,
                "schema": response_schema,
            },
        }

    async with httpx.AsyncClient(timeout=90.0) as client:
        print(f"[LLM:{label}] Calling model: {settings.GROQ_LLM_MODEL}")
        response = await client.post(GROQ_URL, headers=headers, json=payload)

        if response.status_code == 429 and not _is_retry:
            wait_s = 10.0
            try:
                msg = response.json().get("error", {}).get("message", "")
                m = re.search(r"try again in ([\d.]+)s", msg)
                if m:
                    wait_s = float(m.group(1)) + 1.0
            except Exception:
                pass
            print(f"[LLM:{label}] Rate limited, waiting {wait_s:.1f}s and retrying once")
            await asyncio.sleep(wait_s)
            return await _call_groq_json(
                system_prompt, user_prompt, label, max_tokens,
                response_schema=response_schema, _is_retry=True,
            )

        if response.status_code != 200:
            print(f"[LLM:{label}] Error {response.status_code}: {response.text}")
            response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason")
        if finish_reason == "length":
            print(f"[LLM:{label}] WARNING: response truncated by max_tokens.")

        print(f"\n--- [LLM:{label}] raw response ---\n{content}\n--- end {label} ---\n")
        return extract_json(content)


def _format_pronunciation_evidence(pron: dict) -> str:
    """
    Formats the receiver's output (pronunciation_parser.analyze_gop_result)
    for the LLM prompt.

    `pron` shape: {utterance_avg, total_phonemes, distribution
    {excellent/good/moderate/severe}, severe_flags (deduped, worst-first),
    worst_phoneme (single worst score in the whole utterance, always
    populated when there's data)}. Scores are natural-log probabilities:
    0 = perfect, more negative = worse. We deliberately do NOT pass raw
    per-phoneme lists here -- 500-850+ phonemes per answer carries no
    extra signal over the aggregated distribution, and would blow the
    token budget.
    """
    if not pron or not isinstance(pron, dict) or not pron.get("total_phonemes"):
        return "No pronunciation data available for this part."

    dist = pron.get("distribution", {})
    total = pron.get("total_phonemes", 0)
    avg = pron.get("utterance_avg", "N/A")

    excellent = dist.get("excellent", 0)
    severe = dist.get("severe", 0)
    excellent_pct = round(100 * excellent / total, 1) if total else 0
    severe_pct = round(100 * severe / total, 1) if total else 0

    severe_flags = pron.get("severe_flags", [])
    flags_str = ", ".join(
        f"/{f['phoneme']}/ (worst {f['worst_score']:.2f}, seen {f['count']}x)"
        for f in severe_flags[:8]
    ) if severe_flags else "none"

    worst = pron.get("worst_phoneme")
    worst_str = f"/{worst['phoneme']}/ (score {worst['score']:.2f})" if worst else "N/A"

    return (
        f"GOP log-prob scale: 0=perfect, more negative=worse, below -3.0=severe. "
        f"Utterance average={avg}. {total} phonemes scored: "
        f"{excellent_pct}% excellent, {severe_pct}% severe. "
        f"Severely mispronounced sounds: {flags_str}. "
        f"Single worst phoneme in this answer: {worst_str}."
    )


# ---------- Strict-mode JSON schemas ----------
# Groq's strict structured-output mode requires every property to be
# listed in "required" (an empty list/string is how the model expresses
# "found none/nothing to add" -- these fields were never truly optional
# in the prompts above either) and every object to set
# additionalProperties: False.

_QUOTE_NOTE_SCHEMA = {
    "type": "object",
    "properties": {"quote": {"type": "string"}, "note": {"type": "string"}},
    "required": ["quote", "note"],
    "additionalProperties": False,
}
_QUOTE_ISSUE_SCHEMA = {
    "type": "object",
    "properties": {"quote": {"type": "string"}, "issue": {"type": "string"}},
    "required": ["quote", "issue"],
    "additionalProperties": False,
}

_TEXT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "grammar_errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "issue": {"type": "string"},
                    "correction": {"type": "string"},
                },
                "required": ["quote", "issue", "correction"],
                "additionalProperties": False,
            },
        },
        "grammar_strengths": {"type": "array", "items": _QUOTE_NOTE_SCHEMA},
        "vocabulary_strengths": {"type": "array", "items": _QUOTE_NOTE_SCHEMA},
        "vocabulary_issues": {"type": "array", "items": _QUOTE_ISSUE_SCHEMA},
        "fluency_observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "quote": {"type": "string"},
                    "pattern": {
                        "type": "string",
                        "enum": ["filler", "run-on", "self-correction", "repetition", "incomplete-thought"],
                    },
                },
                "required": ["label", "quote", "pattern"],
                "additionalProperties": False,
            },
        },
        "quantitative_note": {"type": "string"},
    },
    "required": [
        "grammar_errors", "grammar_strengths", "vocabulary_strengths",
        "vocabulary_issues", "fluency_observations", "quantitative_note",
    ],
    "additionalProperties": False,
}

_PRONUNCIATION_SCHEMA = {
    "type": "object",
    "properties": {
        "genuine_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phoneme": {"type": "string"},
                    "total_occurrences_flagged": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["phoneme", "total_occurrences_flagged", "note"],
                "additionalProperties": False,
            },
        },
        "excluded_as_likely_artifacts": {"type": "array", "items": {"type": "string"}},
        "overall_note": {"type": "string"},
    },
    "required": ["genuine_issues", "excluded_as_likely_artifacts", "overall_note"],
    "additionalProperties": False,
}

_FINAL_SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "fluency": {"type": "number"},
        "lexical": {"type": "number"},
        "grammar": {"type": "number"},
        "pronunciation": {"type": "number"},
        "overall": {"type": "number"},
        "generalSummary": {"type": "string"},
        "keyImprovements": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "object",
            "properties": {
                "fluency": {"type": "string"},
                "lexical": {"type": "string"},
                "grammar": {"type": "string"},
                "pronunciation": {"type": "string"},
                "per_part_feedback": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["fluency", "lexical", "grammar", "pronunciation", "per_part_feedback"],
            "additionalProperties": False,
        },
    },
    "required": [
        "fluency", "lexical", "grammar", "pronunciation", "overall",
        "generalSummary", "keyImprovements", "evidence",
    ],
    "additionalProperties": False,
}


def _flatten_segments(parts_grouped: dict) -> list:
    flat = []
    for part_no in sorted(parts_grouped.keys()):
        for seg in parts_grouped[part_no]:
            flat.append({**seg, "part_number": part_no})
    return flat


# ---------- Pass A: Grammar, Lexis & Fluency (must quote the transcript) ----------
# Merged into one call: both need the same transcript+timing input, and
# sending transcripts twice (once per pass) was a meaningful chunk of the
# token usage causing Groq TPM rate-limit failures on longer sessions.

async def _analyze_text(segments: list) -> dict:
    blocks = []
    for seg in segments:
        if not seg.get("transcript"):
            continue
        blocks.append(
            f"[{seg.get('label', seg.get('segment_id',''))}] (Part {seg['part_number']})\n"
            f"Transcript: {seg['transcript']}\n"
            f"Timing: {json.dumps(seg.get('fluency_features'))}"
        )
    combined = "\n\n".join(blocks)

    system_prompt = """You analyze IELTS Speaking transcripts for grammar, vocabulary, fluency and
coherence evidence, using both the transcript text and the timing data provided (speech_rate_wpm,
articulation_rate_wpm, phonation_time_ratio, pause_count, pause_total_sec).

CRITICAL RULE: every entry you list MUST include an exact quote copied from the transcript below --
not a paraphrase, not a made-up example. If you cannot find a real example in the text, do not
invent one; simply list fewer entries.

Reply with ONLY this JSON structure, no markdown, no extra text:
{
  "grammar_errors": [{"quote": "exact words from transcript", "issue": "what's wrong", "correction": "corrected version"}],
  "grammar_strengths": [{"quote": "exact words", "note": "why this shows good control, e.g. accurate complex conditional"}],
  "vocabulary_strengths": [{"quote": "exact words", "note": "why this is a strong lexical choice"}],
  "vocabulary_issues": [{"quote": "exact words", "issue": "e.g. repetition, wrong register, imprecise word choice"}],
  "fluency_observations": [{"label": "segment label", "quote": "exact words", "pattern": "filler | run-on | self-correction | repetition | incomplete-thought"}],
  "quantitative_note": "1-2 sentences synthesizing the timing numbers across segments (e.g. consistent ~115wpm, moderate pausing, one notably slower segment)"
}
List up to 6 grammar_errors, 3 grammar_strengths, 3 vocabulary_strengths, 4 vocabulary_issues,
6 fluency_observations. Fewer is fine if the transcript doesn't support more genuine examples."""

    user_prompt = f"Transcripts:\n\n{combined}\n\nReturn only the JSON object."

    return await _call_groq_json(
        system_prompt, user_prompt, label="text_analysis", max_tokens=2500,
        response_schema=_TEXT_ANALYSIS_SCHEMA,
    )


# ---------- Pass B: Pronunciation (filter artifacts from genuine issues) ----------

async def _analyze_pronunciation(segments: list) -> dict:
    blocks = []
    for seg in segments:
        pron_summary = _format_pronunciation_evidence(seg.get("pronunciation_result") or {})
        blocks.append(f"[{seg.get('label', seg.get('segment_id',''))}] (Part {seg['part_number']})\n{pron_summary}")
    combined = "\n\n".join(blocks)

    system_prompt = """You review phoneme-level GOP (Goodness of Pronunciation) evidence across
several IELTS Speaking segments. GOP scores are the recognizer's confidence, not a direct IELTS
pronunciation judgment -- some flagged phonemes are forced-alignment artifacts, not genuine errors.

Treat these as LIKELY ARTIFACTS and exclude them unless they recur heavily across MANY segments:
- Glottal stops (\u0294) and syllabic consonants (n\u0329, l\u0329, m\u0329, \u014b\u0329) -- these are frequently alignment noise.
- Any phoneme flagged only once (seen 1x) with a borderline score just past the severe threshold.

Treat these as GENUINE signal worth reporting:
- Any phoneme flagged 2+ times within a segment, or appearing as severe across multiple segments.
- Classic ESL-difficulty sounds: /th/ sounds, /r/ /l/, /v/ /w/, vowel-length pairs, diphthongs.

Reply with ONLY this JSON structure, no markdown, no extra text:
{
  "genuine_issues": [{"phoneme": "th-sound", "total_occurrences_flagged": 7, "note": "brief description of the pattern, e.g. recurs across 3 of 7 segments"}],
  "excluded_as_likely_artifacts": ["glottal stop", "syllabic n"],
  "overall_note": "1-2 sentences: overall intelligibility impression given the % excellent/severe across segments and which sounds are the real recurring pattern, if any"
}
List up to 6 genuine_issues, ranked by how consistently they recur."""

    user_prompt = f"Pronunciation evidence per segment:\n\n{combined}\n\nReturn only the JSON object."

    return await _call_groq_json(
        system_prompt, user_prompt, label="pronunciation",
        response_schema=_PRONUNCIATION_SCHEMA,
    )


# ---------- Pass D: Final scoring, anchored to real band descriptors ----------

async def _final_scoring(text_analysis: dict, pronunciation: dict, missing_note: str) -> dict:
    descriptor_block = format_descriptor_block([4, 5, 6, 7, 8])

    system_prompt = f"""You are an expert IELTS Speaking examiner assigning final band scores.

You are given ANALYST FINDINGS below (not raw transcripts) -- specific cited grammar/vocabulary
examples, fluency observations, and a filtered pronunciation summary. Use these findings, anchored
against the official band descriptors below, to assign scores.

{descriptor_block}

CRITICAL RULE: each "evidence" string below MUST reference a SPECIFIC finding from the analyst
findings (quote or closely paraphrase one) -- never write a generic statement like "some errors
occur" or "moderate pace" with nothing concrete behind it. If the findings for a criterion are
thin, say so plainly rather than inventing detail.

Reply with ONLY this JSON structure, no markdown, no extra text:
{{
  "fluency": 5.5,
  "lexical": 5.0,
  "grammar": 5.0,
  "pronunciation": 6.5,
  "overall": 5.5,
  "generalSummary": "2-3 sentences, holistic, mention any part not recorded",
  "keyImprovements": ["specific tip referencing an actual finding", "..."],
  "evidence": {{
    "fluency": "must cite a specific fluency_observations entry or the quantitative_note",
    "lexical": "must cite a specific vocabulary finding",
    "grammar": "must cite a specific grammar finding",
    "pronunciation": "must cite a specific genuine_issues entry or overall_note",
    "per_part_feedback": ["short feedback 1", "short feedback 2", "short feedback 3"]
  }}
}}

Rules:
- Scores 0-9, use .5 steps. overall = average of the 4 scores, rounded to nearest 0.5.
- keyImprovements: 2-4 tips, each grounded in a specific cited finding, not generic advice.
- Keep evidence strings under 30 words each (must fit a quote/citation).
"""

    user_prompt = f"""ANALYST FINDINGS:

--- Grammar, Lexical Resource, Fluency & Coherence ---
{json.dumps(text_analysis, indent=2)}

--- Pronunciation ---
{json.dumps(pronunciation, indent=2)}
{missing_note}

Return only the JSON object."""

    print("\n" + "=" * 70)
    print("[LLM:final_scoring] Analyst findings being sent to the model:")
    print("=" * 70)
    print(user_prompt)
    print("=" * 70 + "\n")

    return await _call_groq_json(
        system_prompt, user_prompt, label="final_scoring", max_tokens=1500,
        response_schema=_FINAL_SCORING_SCHEMA,
    )


_FALLBACK_ANALYSIS = {"note": "Analysis pass failed for this run; see logs."}


async def score_session(parts_grouped: dict) -> dict:
    """
    parts_grouped: {1: [segment_result, ...], 2: [...], 3: [...]}, where each
    segment_result has label/question_text/transcript/fluency_features/
    pronunciation_result (see evaluate.run_part's return shape).

    Four-pass pipeline instead of one holistic call: a single small model
    asked to juggle grammar+lexis+fluency+pronunciation+banding all at once
    tends to produce generic, uncited evidence (verified against real
    session logs). Splitting into focused passes that must quote specific
    transcript/phoneme evidence, then a final pass that scores against real
    IELTS band descriptor language referencing those findings, produces
    evidence that's actually checkable against the source instead of being
    a plausible-sounding guess.
    """
    segments = _flatten_segments(parts_grouped)

    missing_parts = sorted(set([1, 2, 3]) - set(parts_grouped.keys()))
    missing_note = (
        f"\nNOTE: Part(s) {', '.join(str(p) for p in missing_parts)} were not recorded/submitted -- "
        f"grade holistically on whatever was provided and say so plainly in generalSummary."
        if missing_parts else ""
    )

    try:
        text_analysis = await _analyze_text(segments)
    except Exception as e:
        print(f"[LLM Scorer] Text analysis pass failed: {type(e).__name__}: {e}")
        text_analysis = _FALLBACK_ANALYSIS

    try:
        pronunciation = await _analyze_pronunciation(segments)
    except Exception as e:
        print(f"[LLM Scorer] Pronunciation pass failed: {type(e).__name__}: {e}")
        pronunciation = _FALLBACK_ANALYSIS

    try:
        result = await _final_scoring(text_analysis, pronunciation, missing_note)

        if "overall" not in result:
            scores = [
                float(result.get("fluency", 6.0)),
                float(result.get("lexical", 6.0)),
                float(result.get("grammar", 6.0)),
                float(result.get("pronunciation", 6.0)),
            ]
            result["overall"] = round((sum(scores) / 4) * 2) / 2

        # Keep the full per-pass findings alongside the final scores -- not
        # shown in the app today, but valuable for debugging score
        # justifications and for the future training dataset.
        result.setdefault("evidence", {})
        result["evidence"]["detailedAnalysis"] = {
            "textAnalysis": text_analysis,
            "pronunciation": pronunciation,
        }

        print("\n[LLM FINAL SCORES]")
        print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=2))
        print("=" * 70 + "\n")

        return result

    except Exception as e:
        print(f"[LLM Scorer] Final scoring pass failed: {type(e).__name__}: {e}")
        return {
            "fluency": 6.5,
            "lexical": 6.0,
            "grammar": 6.5,
            "pronunciation": 6.5,
            "overall": 6.5,
            "generalSummary": "Automated scoring failed for this submission. These are placeholder scores -- please retry.",
            "keyImprovements": [],
            "evidence": {
                "fluency": "Fallback due to LLM error",
                "lexical": "Fallback due to LLM error",
                "grammar": "Fallback due to LLM error",
                "pronunciation": "Fallback due to LLM error",
                "per_part_feedback": [],
            },
        }
