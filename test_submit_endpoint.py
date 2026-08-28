import httpx

files = {
    "part1_audio": open("audio_samples/nibir1.m4a", "rb"),
    "part2_audio": open("audio_samples/nibir2.m4a", "rb"),
    "part3_audio": open("audio_samples/nibir3.m4a", "rb"),
}
data = {
    "part1_question": "Do you work or study?",
    "part2_question": "Describe a trip.",
    "part3_question": "How will travel change?",
}

resp = httpx.post("http://127.0.0.1:8000/sessions/submit", data=data, files=files, timeout=30)
print(resp.status_code)
print(resp.json())