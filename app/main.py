from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.api.sessions import router as sessions_router
from app.core.config import get_settings

settings = get_settings()


async def require_internal_api_key(x_internal_api_key: str | None = Header(default=None)):
    """
    This service is called server-to-server from the Next.js app, not
    directly from the browser -- so this is a plain shared-secret header
    check, not user auth. Every /sessions/* route depends on this.
    """
    if not x_internal_api_key or x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


app = FastAPI(
    title="NextED IELTS Speaking AI Evaluator",
    description="Architecture: No fine-tuning · pgvector · single LLM call per session",
    version="0.1.0",
)

# Only the Next.js server calls this service directly (server-to-server),
# so CORS doesn't need to be wide open. Restrict to known origins; add your
# deployed Next.js domain here when you deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router, dependencies=[Depends(require_internal_api_key)])


@app.get("/health")
def health():
    return {"status": "ok"}