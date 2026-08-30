"""
Condensed, paraphrased IELTS Speaking band descriptors (source: the public
Cambridge/British Council band descriptor table). Paraphrased into our own
words rather than reproduced verbatim -- this exists to give the scoring
LLM a real anchor to compare evidence against, instead of guessing what
"IELTS-style" scoring means from general training data.

Only bands 4-9 are included: bands 0-3 describe near-total communication
breakdown, which isn't a realistic outcome for anyone who completed a full
test with substantive answers, and including them just dilutes the prompt.
"""

FLUENCY_COHERENCE = {
    9: "Fluent, no effort; hesitation only to plan content, never to search for words/grammar; fully coherent.",
    8: "Fluent with only occasional self-correction; hesitation is rare and content-related; coherent development.",
    7: "Speaks at length without noticeable effort; some language-related hesitation, repetition, or self-correction; uses a range of discourse markers with some flexibility.",
    6: "Willing to speak at length but coherence may slip due to hesitation/repetition/self-correction; uses discourse markers but not always appropriately.",
    5: "Keeps going but leans on repetition, self-correction, or slowing down to maintain flow; simple speech is fluent but complex ideas cause breakdowns; may over-use certain connectives.",
    4: "Frequent noticeable pauses, slow speech, frequent repetition/self-correction; basic sentences linked repetitively; some breakdowns in coherence.",
}

LEXICAL_RESOURCE = {
    9: "Full flexibility and precision in all topics; uses idiomatic language naturally and accurately.",
    8: "Wide vocabulary used flexibly and precisely; uses less common/idiomatic vocabulary skilfully with only occasional inaccuracy; paraphrases effectively.",
    7: "Flexible vocabulary across topics; some less common/idiomatic vocabulary with some awareness of style/collocation, occasional inappropriate choices; paraphrases effectively.",
    6: "Wide enough vocabulary to discuss topics at length and make meaning clear despite inappropriacies; generally paraphrases successfully.",
    5: "Can discuss familiar and unfamiliar topics but with limited flexibility; attempts paraphrase with mixed success.",
    4: "Can discuss familiar topics but only conveys basic meaning on unfamiliar ones; frequent word-choice errors; rarely attempts paraphrase.",
}

GRAMMAR = {
    9: "Full range of structures used naturally; consistently accurate apart from native-speaker-like slips.",
    8: "Wide range of structures used flexibly; majority of sentences error-free, only occasional inaccuracies.",
    7: "Range of complex structures with some flexibility; frequently produces error-free sentences though some mistakes persist.",
    6: "Mix of simple and complex structures with limited flexibility; frequent mistakes in complex structures, but these rarely block understanding.",
    5: "Produces basic sentence forms with reasonable accuracy; limited range of more complex structures, usually containing errors that may cause some comprehension issues.",
    4: "Basic sentence forms and some correct simple sentences, but subordinate structures are rare; errors are frequent and may cause misunderstanding.",
}

PRONUNCIATION = {
    9: "Full range of phonological features used with precision; effortless to understand throughout.",
    8: "Wide range of pronunciation features, sustained with only occasional lapses; easy to understand throughout, L1 accent has minimal effect.",
    7: "Shows positive features of band 6 and some (not all) of band 8 -- a step up in control/range without being fully sustained.",
    6: "Range of pronunciation features with mixed control; some effective use but not sustained; generally understood throughout, though individual word/sound mispronunciations reduce clarity at times.",
    5: "Shows positive features of band 4 and some (not all) of band 6.",
    4: "Limited range of pronunciation features; frequent lapses in control; mispronunciations are frequent and cause some difficulty for the listener.",
}


def format_descriptor_block(bands: list[int] = [4, 5, 6, 7, 8]) -> str:
    lines = []
    for b in bands:
        lines.append(
            f"Band {b}:\n"
            f"  Fluency & Coherence: {FLUENCY_COHERENCE.get(b, '')}\n"
            f"  Lexical Resource: {LEXICAL_RESOURCE.get(b, '')}\n"
            f"  Grammar: {GRAMMAR.get(b, '')}\n"
            f"  Pronunciation: {PRONUNCIATION.get(b, '')}"
        )
    return "\n\n".join(lines)
