import httpx

session_id = "67c5bf99-dfe3-4c60-a38b-a60e6f147381"

print("--- Session status ---")
resp = httpx.get(f"http://127.0.0.1:8000/sessions/{session_id}")
print(resp.status_code)
print(resp.json())

print("\n--- Report ---")
try:
    resp2 = httpx.get(f"http://127.0.0.1:8000/sessions/{session_id}/report")
    print(resp2.status_code)
    print(resp2.json())
except Exception as e:
    print(f"Report not available yet: {e}")