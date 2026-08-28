from fastapi import FastAPI

from app.api.sessions import router as sessions_router   # adjust path if needed



app = FastAPI(
    title="NextED IELTS Speaking AI Evaluator",
    description="Architecture: No fine-tuning · pgvector · single LLM call per session",
    version="0.1.0",
)

app.include_router(sessions_router)


@app.get("/health")
def health():
    return {"status": "ok"}