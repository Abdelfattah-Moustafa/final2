import requests
import numpy as np
import random

BASE_URL = "http://localhost:8000"

def run_test():
    # 1. Generate unique email
    email = f"test_{random.randint(1000, 9999)}@example.com"
    password = "secure_password_123"
    
    print(f"Signing up new user: {email}...")
    signup_res = requests.post(
        f"{BASE_URL}/api/auth/signup",
        json={
            "email": email,
            "password": password,
            "full_name": "Test User",
            "role": "Speaker"
        }
    )
    if signup_res.status_code != 201:
        print(f"Signup failed: {signup_res.status_code} - {signup_res.text}")
        return
        
    print("Logging in to get JWT token...")
    login_res = requests.post(
        f"{BASE_URL}/api/auth/login",
        data={
            "username": email,
            "password": password
        }
    )
    if login_res.status_code != 200:
        print(f"Login failed: {login_res.status_code} - {login_res.text}")
        return
        
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("Authentication successful.")

    # 2. Create mock sequence of 30 frames, each with 543 landmarks of (x,y,z)
    print("Creating mock landmark frames...")
    mock_frames = []
    for _ in range(30):
        frame = []
        for _ in range(543):
            # Send small random floats
            frame.append([random.uniform(0.4, 0.6), random.uniform(0.4, 0.6), 0.0])
        mock_frames.append(frame)

    # 3. Test English (ASL) inference
    print("\nTesting English (ASL) model endpoint...")
    asl_res = requests.post(
        f"{BASE_URL}/api/translate",
        json={
            "frames": mock_frames,
            "language": "english",
            "w": 640,
            "h": 480
        },
        headers=headers
    )
    print(f"ASL Result: {asl_res.status_code}")
    print(asl_res.json())

    # 4. Test Arabic (ArSL) inference
    print("\nTesting Arabic (ArSL) model endpoint...")
    ar_res = requests.post(
        f"{BASE_URL}/api/translate",
        json={
            "frames": mock_frames,
            "language": "arabic",
            "w": 640,
            "h": 480
        },
        headers=headers
    )
    print(f"Arabic Result: {ar_res.status_code}")
    print(ar_res.json())

if __name__ == "__main__":
    run_test()
