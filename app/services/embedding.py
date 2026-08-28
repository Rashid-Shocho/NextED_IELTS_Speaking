import httpx
from app.core.config import get_settings

settings = get_settings()


async def embed_text(text: str) -> list[float]:
    """
    Create embedding using Hugging Face Inference API (free).
    Model: BAAI/bge-base-en-v1.5 → 768 dimensions
    """
    if not text or not text.strip():
        return [0.0] * settings.EMBEDDING_DIM

    headers = {
        "Authorization": f"Bearer {settings.HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": text[:8000],  # safety limit
        "options": {"wait_for_model": True}
    }

    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{settings.HF_EMBEDDING_MODEL}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            embedding = response.json()

            # HF sometimes returns nested list
            if isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = embedding[0]

            if len(embedding) != settings.EMBEDDING_DIM:
                print(f"[Embedding] Warning: got {len(embedding)} dims, expected {settings.EMBEDDING_DIM}")
                # Pad or truncate
                if len(embedding) < settings.EMBEDDING_DIM:
                    embedding = embedding + [0.0] * (settings.EMBEDDING_DIM - len(embedding))
                else:
                    embedding = embedding[:settings.EMBEDDING_DIM]

            return embedding

    except Exception as e:
        print(f"[Embedding] Error: {e}")
        return [0.0] * settings.EMBEDDING_DIM