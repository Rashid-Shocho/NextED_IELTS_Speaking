"""
LangGraph wiring. Pronunciation now needs reference_text (the transcript),
so it runs AFTER transcribe, not in parallel with it:

    vad --route_after_vad-->
        "mark_no_speech"    (no speech detected)
        "transcribe"        (has speech)

    mark_no_speech --> END
    transcribe --> pronunciation --> finalize --> END
"""

from langgraph.graph import StateGraph, START, END

from app.graph.nodes import (
    PartState,
    vad_node,
    route_after_vad,
    mark_no_speech_node,
    transcribe_node,
    pronunciation_node,
    finalize_node,
)


def build_part_workflow():
    graph = StateGraph(PartState)

    graph.add_node("vad", vad_node)
    graph.add_node("mark_no_speech", mark_no_speech_node)
    graph.add_node("transcribe", transcribe_node)
    graph.add_node("pronunciation", pronunciation_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "vad")

    graph.add_conditional_edges(
        "vad",
        route_after_vad,
        {
            "mark_no_speech": "mark_no_speech",
            "transcribe": "transcribe",
        },
    )

    graph.add_edge("mark_no_speech", END)
    graph.add_edge("transcribe", "pronunciation")
    graph.add_edge("pronunciation", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


part_workflow = build_part_workflow()