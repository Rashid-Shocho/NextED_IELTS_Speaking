from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.sessions import router as sessions_router


app = FastAPI(
    title="NextED IELTS Speaking AI Evaluator",
    description="Architecture: No fine-tuning · pgvector · single LLM call per session",
    version="0.1.0",
)

# Frontend runs on a different origin (Next.js dev server / deployed domain),
# so the browser blocks requests without this. TODO: replace "*" with your
# actual frontend domain(s) before production (e.g. "https://yourapp.com").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)


@app.get("/health")
def health():
    return {"status": "ok"}