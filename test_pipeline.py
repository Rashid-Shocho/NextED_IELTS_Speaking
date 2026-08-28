"""
End-to-end smoke test against a running server (uvicorn app.main:app).
Exercises: create session -> trigger evaluate -> poll -> report.
Run: python tests/test_pipeline.py
"""

import time
import httpx

BASE_URL = "http://127.0.0.1:8000"

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
    with httpx.Client(timeout=30) as client:
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
        for i in range(80):          # was 30 -- 80 * 3s = 240s, enough for RunPod cold start
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