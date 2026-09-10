"""
End-to-end smoke test against a running server (uvicorn app.main:app).
Exercises: create session -> trigger evaluate -> poll -> report.
Run: python test_pipeline.py

Requires the server's .env INTERNAL_API_KEY to also be set in THIS
environment (or just paste it into API_KEY below) -- every /sessions/*
route now requires it as the X-Internal-Api-Key header.
"""

import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("INTERNAL_API_KEY")

if not API_KEY:
    raise SystemExit(
        "INTERNAL_API_KEY not found in .env. Add it (same value the server "
        "is running with) before running this test."
    )

HEADERS = {"X-Internal-Api-Key": API_KEY}

payload = {
    "user_id": "test_user_1",
    "parts": [
        {
            "part": 1,
            "question_text": "Do you prefer living in a house or an apartment, and why?",
            "audio_url": "audio_samples/nibir1.m4a",
        },
        {
            "part": 2,
            "question_text": "Describe a memorable trip you have taken. You should say: where you went, who you traveled with, what you saw, and explain why it was memorable.",
            "audio_url": "audio_samples/nibir2.m4a",
        },
        {
            "part": 3,
            "question_text": "How do you think international travel will change in the next twenty or thirty years?",
            "audio_url": "audio_samples/nibir3.m4a",
        },
    ],
}


def main():
    with httpx.Client(timeout=30, headers=HEADERS) as client:
        print("1) Creating session...")
        resp = client.post(f"{BASE_URL}/sessions", json=payload)
        resp.raise_for_status()
        session = resp.json()
        session_id = session["id"]
        print(f"   session_id = {session_id}")

        print("2) Triggering evaluation...")
        resp = client.post(f"{BASE_URL}/sessions/{session_id}/evaluate")
        resp.raise_for_status()
        print(f"   {resp.json()}")

        print("3) Polling session status...")
        for i in range(80):          # 80 * 3s = 240s, enough for RunPod cold start
            time.sleep(3)
            resp = client.get(f"{BASE_URL}/sessions/{session_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data["status"]
            part_statuses = [(p["part_number"], p["status"]) for p in data["parts"]]
            print(f"   [{i}] session={status}  parts={part_statuses}  error={data.get('error_message')}")

            if status in ("completed", "failed", "needs_rerecording"):
                break

        print("\nFinal session state:")
        print(data)

        if status == "completed":
            print("\n4) Fetching report...")
            resp = client.get(f"{BASE_URL}/sessions/{session_id}/report")
            resp.raise_for_status()
            print(resp.json())
        elif status == "needs_rerecording":
            print(f"\nSession needs re-recording: {data.get('parts')}")


if __name__ == "__main__":
    main()
    