#!/usr/bin/env python3
"""Strict API checklist for translation modules + meeting/auth behavior.

Covers modules:
1) Sign to Text
2) Sign to Speech
3) Text to Sign
4) Speech to Sign
5) Live Meeting (API/auth preconditions)

Usage (PowerShell):
  $env:E2E_EMAIL='user@example.com'
  $env:E2E_PASSWORD='password'
  C:/Users/abody/AppData/Local/Python/pythoncore-3.11-64/python.exe scripts/deep_module_checklist.py

Optional env vars:
  BASE_URL (default: http://localhost:8000)
  WARM_HEALTH (default: 0; set to 1 to invoke /api/health?warm=1)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
EMAIL = os.getenv("E2E_EMAIL", "").strip()
PASSWORD = os.getenv("E2E_PASSWORD", "").strip()
WARM_HEALTH = os.getenv("WARM_HEALTH", "0") == "1"


@dataclass
class Result:
    name: str
    passed: bool
    detail: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_dummy_frames(num_frames: int = 30) -> List[List[List[float | None]]]:
    # 543 landmarks x xyz values; use None so server maps to NaN-compatible path.
    frame = [[None, None, None] for _ in range(543)]
    return [frame for _ in range(num_frames)]


def run_case(name: str, fn: Callable[[], str], out: List[Result]) -> None:
    try:
        detail = fn()
        out.append(Result(name=name, passed=True, detail=detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as exc:
        out.append(Result(name=name, passed=False, detail=str(exc)))
        print(f"[FAIL] {name}: {exc}")


def main() -> int:
    if not EMAIL or not PASSWORD:
        print("E2E_EMAIL and E2E_PASSWORD are required.")
        return 2

    results: List[Result] = []

    def login() -> str:
        r = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": EMAIL, "password": PASSWORD},
            timeout=20,
        )
        require(r.status_code == 200, f"expected 200, got {r.status_code}")
        data = r.json()
        require("access_token" in data and data["access_token"], "missing access_token")
        token = data["access_token"]
        headers["Authorization"] = f"Bearer {token}"
        return f"token_len={len(token)}"

    headers: Dict[str, str] = {}

    run_case("Auth/Login", login, results)

    def health_default() -> str:
        r = requests.get(f"{BASE_URL}/api/health", timeout=20)
        require(r.status_code == 200, f"expected 200, got {r.status_code}")
        data = r.json()
        require(data.get("status") == "ok", f"health not ok: {data}")
        require("models_ready" in data, "missing models_ready")
        return json.dumps({
            "status": data.get("status"),
            "models_ready": data.get("models_ready"),
            "db_status": data.get("db_status"),
            "engine_status": data.get("engine_status"),
        })

    run_case("Health/Default", health_default, results)

    if WARM_HEALTH:
        def health_warm() -> str:
            r = requests.get(f"{BASE_URL}/api/health", params={"warm": 1}, timeout=60)
            require(r.status_code == 200, f"expected 200, got {r.status_code}")
            data = r.json()
            require(data.get("status") in {"ok", "degraded"}, f"bad warm health: {data}")
            return json.dumps({
                "status": data.get("status"),
                "db_status": data.get("db_status"),
                "engine_status": data.get("engine_status"),
                "error": data.get("error"),
            })

        run_case("Health/Warm", health_warm, results)

    def sign_to_text_endpoint() -> str:
        payload = {"frames": make_dummy_frames(30)}
        r = requests.post(f"{BASE_URL}/api/translate", headers=headers, json=payload, timeout=30)
        require(r.status_code == 200, f"expected 200, got {r.status_code}")
        data = r.json()
        require("confidence" in data, f"missing confidence: {data}")
        require(any(k in data for k in ("text", "status")), f"missing text/status: {data}")
        return json.dumps({"text": data.get("text", ""), "status": data.get("status"), "confidence": data.get("confidence")})

    run_case("Module1/SignToText", sign_to_text_endpoint, results)

    def sign_to_speech_sentence() -> str:
        payload = {"gloss": ["hello", "world"]}
        r = requests.post(f"{BASE_URL}/api/translate/sentence", headers=headers, json=payload, timeout=30)
        require(r.status_code == 200, f"expected 200, got {r.status_code}")
        data = r.json()
        require(isinstance(data.get("sentence"), str), f"sentence missing: {data}")
        require(data.get("source") in {"llm", "fallback"}, f"unexpected source: {data}")
        return json.dumps({"source": data.get("source"), "sentence": data.get("sentence")})

    run_case("Module2/SignToSpeechSentence", sign_to_speech_sentence, results)

    def text_to_sign_batch() -> str:
        payload = {"words": ["hello", "world"]}
        r = requests.post(f"{BASE_URL}/api/signs/batch", headers=headers, json=payload, timeout=30)
        require(r.status_code == 200, f"expected 200, got {r.status_code}")
        data = r.json()
        require("found" in data and "missing" in data, f"bad schema: {data}")
        require(isinstance(data["found"], list), f"found not list: {data}")
        return json.dumps({"found_count": len(data["found"]), "missing": data.get("missing", [])})

    run_case("Module3/TextToSign", text_to_sign_batch, results)

    def speech_to_sign_batch() -> str:
        payload = {"words": ["how", "are", "you"]}
        r = requests.post(f"{BASE_URL}/api/signs/batch", headers=headers, json=payload, timeout=30)
        require(r.status_code == 200, f"expected 200, got {r.status_code}")
        data = r.json()
        require("found" in data and "missing" in data, f"bad schema: {data}")
        return json.dumps({"found_count": len(data["found"]), "missing_count": len(data["missing"])})

    run_case("Module4/SpeechToSign", speech_to_sign_batch, results)

    def meeting_auth_guard() -> str:
        payload = {"words": ["meeting", "test"]}
        r = requests.post(f"{BASE_URL}/api/signs/batch", json=payload, timeout=20)
        require(r.status_code in {401, 403}, f"expected 401/403 unauth, got {r.status_code}")
        return f"unauth_status={r.status_code}"

    run_case("Module5/MeetingAuthGuard", meeting_auth_guard, results)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print("\n=== Strict Checklist Summary ===")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        for idx, item in enumerate(failed, start=1):
            print(f"{idx}. {item.name} -> {item.detail}")

    report = {
        "base_url": BASE_URL,
        "passed": [r.__dict__ for r in passed],
        "failed": [r.__dict__ for r in failed],
    }
    report_path = os.path.join("scripts", "deep_module_checklist_report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"Report written to {report_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
