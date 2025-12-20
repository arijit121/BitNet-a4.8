import requests
import json

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Health Check
    try:
        resp = requests.get(f"{base_url}/")
        print("Health Check:", resp.json())
        assert resp.status_code == 200
    except Exception as e:
        print("Health check failed:", e)
        return

    # 2. Chat
    payload = {
        "message": "Hello BitNet!",
        "max_length": 50,
        "temperature": 0.8
    }
    
    try:
        resp = requests.post(f"{base_url}/chat", json=payload)
        print("Chat Status:", resp.status_code)
        if resp.status_code == 200:
            print("Response:", resp.json())
        else:
            print("Error:", resp.text)
    except Exception as e:
        print("Chat request failed:", e)

if __name__ == "__main__":
    test_api()
