import json
import re
import httpx
from app.core.config import get_settings

settings = get_settings()


def extract_json(text: str) -> dict:
    """Very tolerant JSON extractor."""
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

    # Find the largest possible {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            candidate = candidate.rstrip(", \n\r\t")
            if not candidate.endswith("}"):
                candidate += '"}]}'
            try:
                return json.loads(candidate)
            except Exception:
                pass

    raise ValueError("Could not extract valid JSON from model response")


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


async def score_session(
    questions: list[str],
    transcripts: list[str],
    fluency_features: list[dict],
    pronunciation_evidence: list[dict],
) -> dict:
    parts_text = ""
    for i, (q, t, flu, pron) in enumerate(
        zip(questions, transcripts, fluency_features, pronunciation_evidence), 1
    ):
        pron_summary = _format_pronunciation_evidence(pron or {})

        parts_text += f"""
### Part {i}
Question: {q}
Transcript: {t}
Fluency features: {json.dumps(flu)}
Pronunciation: {pron_summary}
"""

    print("\n" + "=" * 70)
    print("[LLM INPUT] Data being sent to the model:")
    print("=" * 70)
    print(parts_text)
    print("=" * 70 + "\n")

    system_prompt = """You are an expert IELTS Speaking examiner.
Score the candidate holistically across all three parts.

Reply with ONLY a valid JSON object. No thinking, no markdown, no extra text.

Required structure (keep every string short):
{
  "fluency": 5.5,
  "lexical": 5.0,
  "grammar": 5.0,
  "pronunciation": 6.5,
  "overall": 5.5,
  "evidence": {
    "fluency": "short reason",
    "lexical": "short reason",
    "grammar": "short reason",
    "pronunciation": "short reason, cite the single worst phoneme if one is given",
    "per_part_feedback": ["short feedback 1", "short feedback 2", "short feedback 3"]
  }
}

Rules:
- Scores 0-9, use .5 steps
- overall = average of 4 scores, rounded to nearest 0.5
- For pronunciation: cite the single worst phoneme explicitly in your evidence if one is given. Base your score primarily on the % severe and the severe-mispronunciation list, not the raw average number alone. A high % excellent with few/no severe flags means strong pronunciation (band 7+); many severe flags means weak pronunciation (band 5 or below).
- Keep all text under 15 words
"""

    user_prompt = f"""Score this IELTS Speaking test:

{parts_text}

Return only the JSON object."""

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
        "max_tokens": 3000,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            print(f"[LLM] Calling model: {settings.GROQ_LLM_MODEL}")
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                print(f"[LLM] Error {response.status_code}: {response.text}")
                response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")

            if finish_reason == "length":
                print("[LLM] WARNING: response truncated by max_tokens (finish_reason=length).")

            print("\n" + "=" * 70)
            print("[LLM] Full raw response:")
            print("=" * 70)
            print(content)
            print("=" * 70 + "\n")

            result = extract_json(content)

            if "overall" not in result:
                scores = [
                    float(result.get("fluency", 6.0)),
                    float(result.get("lexical", 6.0)),
                    float(result.get("grammar", 6.0)),
                    float(result.get("pronunciation", 6.0)),
                ]
                avg = sum(scores) / 4
                result["overall"] = round(avg * 2) / 2

            print("\n[LLM FINAL SCORES]")
            print(json.dumps(result, indent=2))
            print("=" * 70 + "\n")

            return result

    except Exception as e:
        print(f"[LLM Scorer] Error: {type(e).__name__}: {e}")
        return {
            "fluency": 6.5,
            "lexical": 6.0,
            "grammar": 6.5,
            "pronunciation": 6.5,
            "overall": 6.5,
            "evidence": {
                "fluency": "Fallback due to LLM error",
                "lexical": "Fallback due to LLM error",
                "grammar": "Fallback due to LLM error",
                "pronunciation": "Fallback due to LLM error",
                "per_part_feedback": ["Error", "Error", "Error"],
            },
        }