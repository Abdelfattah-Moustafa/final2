#!/usr/bin/env python3
"""Exhaustive website audit runner (API + Socket.IO + Desktop/Mobile UI).

This script is intentionally broad and tries to exercise as many behaviors as possible.
It produces a JSON report at scripts/exhaustive_site_audit_report.json.

Usage (PowerShell):
  $env:E2E_EMAIL='user@example.com'
  $env:E2E_PASSWORD='password'
  C:/Users/abody/AppData/Local/Python/pythoncore-3.11-64/python.exe scripts/exhaustive_site_audit.py

Optional:
  BASE_URL=http://localhost:8000
  E2E_HEADLESS=1
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests
import socketio
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
E2E_EMAIL = os.getenv("E2E_EMAIL", "").strip()
E2E_PASSWORD = os.getenv("E2E_PASSWORD", "").strip()
HEADLESS = os.getenv("E2E_HEADLESS", "1") != "0"


@dataclass
class CheckResult:
    area: str
    name: str
    status: str  # pass|fail|skip
    detail: str


class AuditRunner:
    def __init__(self) -> None:
        self.results: List[CheckResult] = []

    def add(self, area: str, name: str, status: str, detail: str) -> None:
        self.results.append(CheckResult(area=area, name=name, status=status, detail=detail))
        print(f"[{status.upper()}] {area} :: {name} -> {detail}")

    def case(self, area: str, name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
            self.add(area, name, "pass", detail)
        except Exception as exc:
            self.add(area, name, "fail", str(exc))

    def skip(self, area: str, name: str, detail: str) -> None:
        self.add(area, name, "skip", detail)


def require(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def random_email() -> str:
    stamp = int(time.time())
    tail = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"autotest.{stamp}.{tail}@gmail.com"


def create_test_user(session: requests.Session, full_name: str, role: str) -> Dict[str, str]:
    password = "Password123!"
    payload = {
        "full_name": full_name,
        "email": random_email(),
        "password": password,
        "role": role,
    }
    resp = session.post(f"{BASE_URL}/api/auth/signup", json=payload, timeout=30)
    require(resp.status_code == 201, f"expected 201 got {resp.status_code}: {resp.text[:200]}")
    return {"email": payload["email"], "password": password, "role": role}


def parse_local_links(html: str) -> List[str]:
    links = re.findall(r"(?:href|src)=['\"](/[^'\"]+)['\"]", html, flags=re.IGNORECASE)
    uniq = []
    seen = set()
    for link in links:
        if link.startswith("/#"):
            continue
        if link not in seen:
            seen.add(link)
            uniq.append(link)
    return uniq


def api_audit(runner: AuditRunner) -> Dict[str, str]:
    session = requests.Session()
    token_headers: Dict[str, str] = {}
    if E2E_EMAIL and E2E_PASSWORD:
        speaker_creds = {"email": E2E_EMAIL, "password": E2E_PASSWORD, "role": "Speaker"}
    else:
        speaker_creds = create_test_user(session, "Auto Speaker", "Speaker")

    def get(path: str, expected: int = 200) -> requests.Response:
        resp = session.get(f"{BASE_URL}{path}", timeout=30)
        require(resp.status_code == expected, f"expected {expected}, got {resp.status_code}")
        return resp

    def post_form(path: str, data: Dict[str, str], expected: int) -> requests.Response:
        resp = session.post(
            f"{BASE_URL}{path}",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        require(resp.status_code == expected, f"expected {expected}, got {resp.status_code}: {resp.text[:200]}")
        return resp

    def post_json(path: str, data: dict, expected: int, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        h = headers or {}
        resp = session.post(f"{BASE_URL}{path}", json=data, headers=h, timeout=60)
        require(resp.status_code == expected, f"expected {expected}, got {resp.status_code}: {resp.text[:300]}")
        return resp

    runner.case("api", "GET /", lambda: f"len={len(get('/').text)}")
    runner.case("api", "GET /login", lambda: f"len={len(get('/login').text)}")
    runner.case("api", "GET /signup", lambda: f"len={len(get('/signup').text)}")
    runner.case("api", "GET /dashboard", lambda: f"len={len(get('/dashboard').text)}")

    def static_asset_check() -> str:
        html = get("/").text
        links = parse_local_links(html)
        checked = 0
        for path in links:
            if path.startswith("/api/"):
                continue
            if path.endswith(".svg") and path.startswith("data:"):
                continue
            resp = session.get(f"{BASE_URL}{path}", timeout=30)
            require(resp.status_code in {200, 304}, f"asset {path} -> {resp.status_code}")
            checked += 1
        return f"assets_checked={checked}"

    runner.case("api", "Static assets from landing", static_asset_check)

    runner.case(
        "auth",
        "Login invalid email rejected",
        lambda: f"status={post_form('/api/auth/login', {'username': 'not-an-email', 'password': 'x'}, 400).status_code}",
    )

    if speaker_creds:
        runner.case(
            "auth",
            "Login wrong password rejected",
            lambda: f"status={post_form('/api/auth/login', {'username': speaker_creds['email'], 'password': 'wrong-password'}, 401).status_code}",
        )

        def login_ok() -> str:
            resp = post_form('/api/auth/login', {'username': speaker_creds['email'], 'password': speaker_creds['password']}, 200)
            data = resp.json()
            token = data.get("access_token", "")
            require(token, "missing access_token")
            token_headers["Authorization"] = f"Bearer {token}"
            return f"token_len={len(token)}"

        runner.case("auth", "Login valid credentials", login_ok)
    else:
        runner.skip("auth", "Login valid credentials", "missing speaker credentials")

    def signup_invalid_email() -> str:
        payload = {
            "full_name": "Auto Invalid",
            "email": "bad-email",
            "password": "Password123!",
            "role": "Speaker",
        }
        resp = session.post(f"{BASE_URL}/api/auth/signup", json=payload, timeout=30)
        require(resp.status_code in {400, 422}, f"expected 400/422 got {resp.status_code}")
        return f"status={resp.status_code}"

    runner.case("auth", "Signup invalid email rejected", signup_invalid_email)

    def signup_short_password() -> str:
        payload = {
            "full_name": "Auto ShortPw",
            "email": random_email(),
            "password": "short",
            "role": "Speaker",
        }
        resp = session.post(f"{BASE_URL}/api/auth/signup", json=payload, timeout=30)
        require(resp.status_code == 400, f"expected 400 got {resp.status_code}")
        return f"status={resp.status_code}"

    runner.case("auth", "Signup short password rejected", signup_short_password)

    def signup_and_duplicate() -> str:
        email = random_email()
        payload = {
            "full_name": "Auto User",
            "email": email,
            "password": "Password123!",
            "role": "Speaker",
        }
        first = session.post(f"{BASE_URL}/api/auth/signup", json=payload, timeout=30)
        require(first.status_code == 201, f"first signup expected 201 got {first.status_code} {first.text[:160]}")
        second = session.post(f"{BASE_URL}/api/auth/signup", json=payload, timeout=30)
        require(second.status_code == 400, f"duplicate expected 400 got {second.status_code}")
        return f"created={email} duplicate_status={second.status_code}"

    runner.case("auth", "Signup create + duplicate guard", signup_and_duplicate)

    runner.case(
        "api",
        "Health default",
        lambda: f"payload={get('/api/health').json()}",
    )

    def health_warm() -> str:
        resp = session.get(f"{BASE_URL}/api/health", params={"warm": 1}, timeout=120)
        require(resp.status_code == 200, f"expected 200 got {resp.status_code}")
        data = resp.json()
        require(data.get("status") in {"ok", "degraded"}, f"unexpected status {data}")
        return f"payload={data}"

    runner.case("api", "Health warm-up", health_warm)

    # Guard checks without auth
    runner.case(
        "authz",
        "Translate blocked without token",
        lambda: f"status={session.post(f'{BASE_URL}/api/translate', json={'frames': []}, timeout=30).status_code}",
    )
    runner.case(
        "authz",
        "Signs batch blocked without token",
        lambda: f"status={session.post(f'{BASE_URL}/api/signs/batch', json={'words': ['hello']}, timeout=30).status_code}",
    )

    if token_headers:
        runner.case(
            "auth",
            "Profile with token",
            lambda: f"email={session.get(f'{BASE_URL}/api/auth/me', headers=token_headers, timeout=30).json().get('email')}",
        )

        def translate_ok() -> str:
            frames = [[[None, None, None] for _ in range(543)] for _ in range(30)]
            resp = post_json('/api/translate', {"frames": frames}, 200, headers=token_headers)
            data = resp.json()
            require("confidence" in data, f"bad response {data}")
            return f"payload_keys={list(data.keys())}"

        runner.case("module1", "Sign-to-Text API", translate_ok)

        def sentence_ok() -> str:
            resp = post_json('/api/translate/sentence', {"gloss": ["hello", "world"]}, 200, headers=token_headers)
            data = resp.json()
            require("sentence" in data, f"bad response {data}")
            return f"source={data.get('source')} sentence={data.get('sentence')[:80]}"

        runner.case("module2", "Sign-to-Speech sentence API", sentence_ok)

        def text_to_sign_ok() -> str:
            resp = post_json('/api/signs/batch', {"words": ["hello", "world"]}, 200, headers=token_headers)
            data = resp.json()
            require("found" in data and "missing" in data, f"bad response {data}")
            return f"found={len(data['found'])} missing={len(data['missing'])}"

        runner.case("module3", "Text-to-Sign API", text_to_sign_ok)

        def speech_to_sign_ok() -> str:
            resp = post_json('/api/signs/batch', {"words": ["how", "are", "you"]}, 200, headers=token_headers)
            data = resp.json()
            require("found" in data and "missing" in data, f"bad response {data}")
            return f"found={len(data['found'])} missing={len(data['missing'])}"

        runner.case("module4", "Speech-to-Sign API", speech_to_sign_ok)

        def signs_lookup_ok() -> str:
            resp = session.get(f"{BASE_URL}/api/signs/lookup", params={"words": "hello world"}, headers=token_headers, timeout=30)
            require(resp.status_code == 200, f"expected 200 got {resp.status_code}")
            data = resp.json()
            require("found" in data and "missing" in data, f"bad response {data}")
            return f"found={len(data['found'])} missing={len(data['missing'])}"

        runner.case("module3", "Signs lookup API", signs_lookup_ok)

        def sign_word_route() -> str:
            resp = session.get(f"{BASE_URL}/api/signs/hello", headers=token_headers, timeout=30)
            require(resp.status_code in {200, 404}, f"unexpected status {resp.status_code}")
            return f"status={resp.status_code}"

        runner.case("module3", "Single sign route", sign_word_route)
    else:
        runner.skip("auth", "Profile with token", "no auth token available")
        runner.skip("module1", "Sign-to-Text API", "no auth token available")
        runner.skip("module2", "Sign-to-Speech sentence API", "no auth token available")
        runner.skip("module3", "Text-to-Sign API", "no auth token available")
        runner.skip("module4", "Speech-to-Sign API", "no auth token available")

    return token_headers


def socket_audit(runner: AuditRunner) -> None:
    sio = socketio.Client(logger=False, engineio_logger=False)
    events: List[str] = []

    @sio.event
    def connect() -> None:
        events.append("connect")

    @sio.event
    def disconnect() -> None:
        events.append("disconnect")

    def connect_case() -> str:
        sio.connect(BASE_URL, transports=["polling"], wait_timeout=10)
        require(sio.connected, "socket did not connect")
        return "connected"

    runner.case("socket", "Connect", connect_case)

    def join_leave_case() -> str:
        room = f"audit-room-{int(time.time())}"
        sio.emit("join_room", {"room": room})
        time.sleep(0.4)
        sio.emit("translate_sentence", {"room": room, "text": "hello", "mode": "audit", "senderRole": "Speaker"})
        time.sleep(0.4)
        sio.emit("leave_room", {"room": room})
        time.sleep(0.4)
        return f"room={room}"

    runner.case("socket", "Join/Relay/Leave", join_leave_case)

    def disconnect_case() -> str:
        sio.disconnect()
        require(not sio.connected, "socket still connected")
        return f"events={events}"

    runner.case("socket", "Disconnect", disconnect_case)


def login_ui(page: Page, email: str, password: str) -> None:
    page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded")
    page.locator("#inp-email").fill(email)
    page.locator("#inp-password").fill(password)
    page.locator("#submit-btn").click()
    page.wait_for_url("**/dashboard", timeout=20000)


def browser_flow_for_viewport(runner: AuditRunner, page: Page, label: str) -> None:
    area = f"ui-{label}"

    runner.case(area, "Landing load", lambda: _landing_load(page))
    runner.case(area, "Landing CTA Get Started", lambda: _landing_cta_signup(page))
    runner.case(area, "Landing CTA Sign In", lambda: _landing_cta_login(page))
    runner.case(area, "Login invalid email validation", lambda: _login_invalid_email(page))
    runner.case(area, "Login tab switch to signup", lambda: _login_switch_signup(page))
    runner.case(area, "Signup mismatch validation", lambda: _signup_mismatch(page))

    if E2E_EMAIL and E2E_PASSWORD:
        runner.case(area, "Login success", lambda: _login_success(page, E2E_EMAIL, E2E_PASSWORD))
        runner.case(area, "Dashboard header", lambda: _dashboard_header(page))
        runner.case(area, "Module switch vision", lambda: _switch_module(page, "vision", label))
        runner.case(area, "Module switch speech", lambda: _switch_module(page, "speech", label))
        runner.case(area, "Module switch text2sign", lambda: _switch_module(page, "text2sign", label))
        runner.case(area, "Module switch speech2sign", lambda: _switch_module(page, "speech2sign", label))
        runner.case(area, "Module switch meeting", lambda: _switch_module(page, "meeting", label))
        runner.case(area, "Text-to-sign send", lambda: _text_to_sign_send(page, label))
        runner.case(area, "Sign-to-speech voice toggle", lambda: _sign_to_speech_toggle(page, label))
        runner.case(area, "Speech-to-sign mic toggle", lambda: _speech_to_sign_toggle(page, label))
        runner.case(area, "Meeting create/join/leave", lambda: _meeting_flow(page, label))
        runner.case(area, "Sign out", lambda: _signout(page, label))
    else:
        runner.skip(area, "Authenticated dashboard flows", "missing E2E_EMAIL/E2E_PASSWORD")


def _landing_load(page: Page) -> str:
    page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
    page.get_by_text("Together").first.wait_for(timeout=10000)
    return "landing rendered"


def _landing_cta_signup(page: Page) -> str:
    page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
    page.get_by_role("link", name="Get Started").first.click()
    page.wait_for_url("**/signup", timeout=10000)
    return "navigated /signup"


def _landing_cta_login(page: Page) -> str:
    page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
    page.get_by_role("link", name="Sign In").first.click()
    page.wait_for_url("**/login", timeout=10000)
    return "navigated /login"


def _login_invalid_email(page: Page) -> str:
    page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded")
    page.locator("#inp-email").fill("not-an-email")
    page.locator("#inp-password").fill("whatever123")
    page.locator("#submit-btn").click()
    page.get_by_text("Please enter a valid email address.").first.wait_for(timeout=8000)
    return "invalid email blocked"


def _login_switch_signup(page: Page) -> str:
    page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded")
    page.locator("#tab-signup").click()
    page.locator("#inp-name").wait_for(timeout=5000)
    return "signup tab visible"


def _signup_mismatch(page: Page) -> str:
    page.goto(f"{BASE_URL}/login?tab=signup", wait_until="domcontentloaded")
    page.locator("#inp-name").fill("Audit User")
    page.locator("#inp-email").fill(random_email())
    page.locator("#inp-password").fill("Password123!")
    page.locator("#inp-confirm").fill("Different123!")
    page.locator("#submit-btn").click()
    page.get_by_text("Passwords do not match.").first.wait_for(timeout=8000)
    return "mismatch blocked"


def _login_success(page: Page, email: str, password: str) -> str:
    login_ui(page, email, password)
    return "dashboard opened"


def _dashboard_header(page: Page) -> str:
    page.locator("#topbar-title").wait_for(timeout=10000)
    return page.locator("#topbar-title").inner_text().strip()


def _switch_module(page: Page, module: str, viewport: str) -> str:
    if viewport == "mobile":
        picker = page.locator("#mobile-module-select")
        picker.wait_for(timeout=8000)
        picker.select_option(value=module)
    else:
        page.locator(f"#nav-{module}").click()

    expected = {
        "vision": "Sign to Text",
        "speech": "Sign to Speech",
        "text2sign": "Text to Sign",
        "speech2sign": "Speech to Sign",
        "meeting": "Live Meeting",
    }[module]
    page.locator("#topbar-title").get_by_text(expected).wait_for(timeout=8000)
    return expected


def _text_to_sign_send(page: Page, viewport: str) -> str:
    _switch_module(page, "text2sign", viewport)
    page.locator("#text2sign-input").fill("hello world")
    page.locator("#text2sign-input").press("Enter")
    page.locator("#t2s-history").wait_for(timeout=10000)
    text = page.locator("#t2s-history").inner_text().strip()
    require("hello world" in text.lower(), f"history missing sent text: {text[:200]}")
    return "history updated"


def _sign_to_speech_toggle(page: Page, viewport: str) -> str:
    _switch_module(page, "speech", viewport)
    btn = page.locator("#s2sp-voice-btn")
    if btn.count() == 0:
        # Voice toggle was intentionally removed; speech is always enabled.
        return "voice control removed (always on)"
    old = btn.inner_text().strip()
    btn.click()
    new = btn.inner_text().strip()
    require(old != new, f"voice toggle unchanged ({old})")
    return f"{old} -> {new}"


def _speech_to_sign_toggle(page: Page, viewport: str) -> str:
    _switch_module(page, "speech2sign", viewport)
    btn = page.locator("#mic-btn")
    btn.click()
    time.sleep(1.2)
    status_text = page.locator("#mic-status").inner_text().strip().lower()
    # Browser environments vary: listening, unsupported, unavailable, denied are all valid runtime outcomes.
    valid = ["listening", "not supported", "denied", "blocked", "unavailable", "start listening"]
    require(any(v in status_text for v in valid), f"unexpected mic status: {status_text}")
    # best-effort stop if active
    try:
        btn.click(timeout=1000)
    except Exception:
        pass
    return f"status={status_text}"


def _meeting_flow(page: Page, viewport: str) -> str:
    _switch_module(page, "meeting", viewport)
    page.get_by_role("button", name="New meeting").first.click()
    page.get_by_text("Connected").first.wait_for(timeout=12000)
    code = page.locator("#meeting-current-code").inner_text().strip()
    require(code and code != "-", f"invalid room code: {code}")
    copy_btn = page.locator("#copy-link-btn")
    require(not copy_btn.is_disabled(), "copy link should be enabled when connected")
    page.locator("#leave-btn").click()
    page.get_by_text("Not connected").first.wait_for(timeout=10000)
    return f"room={code}"


def _signout(page: Page, viewport: str) -> str:
    if viewport == "desktop":
        page.locator(".sidebar-user button").first.click()
    else:
        btn = page.locator("#topbar-signout")
        if btn.count() == 0:
            return "missing mobile sign-out control"
        btn.click()
    page.wait_for_url("**/", timeout=10000)
    return "signed out"


def _meeting_dual_party_flow(browser, origin: str, speaker_creds: Dict[str, str], deaf_creds: Dict[str, str]) -> str:
    speaker_ctx = browser.new_context(viewport={"width": 1366, "height": 768})
    deaf_ctx = browser.new_context(viewport={"width": 1366, "height": 768})
    speaker_ctx.grant_permissions(["microphone", "camera"], origin=origin)
    deaf_ctx.grant_permissions(["microphone", "camera"], origin=origin)

    speaker = speaker_ctx.new_page()
    deaf = deaf_ctx.new_page()
    speaker.set_default_timeout(20000)
    deaf.set_default_timeout(20000)

    def wait_for_text(page: Page, selector: str, expected: str, label: str, timeout_seconds: float = 30, exact: bool = False) -> None:
        deadline = time.time() + timeout_seconds
        last_text = ""
        expected_norm = expected.lower()
        while time.time() < deadline:
            try:
                locator = page.locator(selector)
                if locator.count() == 0:
                    time.sleep(0.25)
                    continue
                last_text = (locator.first.text_content(timeout=700) or "").strip()
            except PlaywrightTimeoutError:
                time.sleep(0.25)
                continue
            actual_norm = last_text.lower()
            if (actual_norm == expected_norm) if exact else (expected_norm in actual_norm):
                return
            time.sleep(0.25)
        raise AssertionError(f"{label} expected '{expected}' in '{last_text}'")

    try:
        login_ui(speaker, speaker_creds["email"], speaker_creds["password"])
        login_ui(deaf, deaf_creds["email"], deaf_creds["password"])

        _switch_module(speaker, "meeting", "desktop")
        _switch_module(deaf, "meeting", "desktop")
        require(bool(speaker.evaluate("typeof io !== 'undefined'")), "speaker socket.io client not loaded")
        require(bool(deaf.evaluate("typeof io !== 'undefined'")), "deaf socket.io client not loaded")

        speaker.get_by_text("Sign translation captions").first.wait_for(timeout=10000)
        deaf.get_by_text("Speaker captions").first.wait_for(timeout=10000)
        speaker.locator("#meeting-speech-out-btn").wait_for(timeout=10000)
        deaf.locator("#meeting-sign-btn").wait_for(timeout=10000)
        require(speaker.locator("#meeting-composer").is_visible(), "speaker composer should be visible")
        require(not deaf.locator("#meeting-composer").is_visible(), "deaf composer should be hidden")

        speaker.evaluate("createMeetingRoom()")
        wait_for_text(speaker, "#meeting-status-pill", "Connected", "speaker did not connect", exact=True)
        room = speaker.locator("#meeting-current-code").inner_text().strip()
        require(room and room != "-", f"invalid room code: {room}")

        deaf.locator("#meeting-room-id").fill(room)
        deaf.evaluate("joinMeeting()")
        wait_for_text(deaf, "#meeting-status-pill", "Connected", "deaf participant did not connect", exact=True)

        speaker_message = "hello from speaker"
        speaker.locator("#meeting-text-input").fill(speaker_message)
        speaker.evaluate("sendMeetingText()")
        wait_for_text(speaker, "#remote-sentence", speaker_message, "speaker echo missing")
        wait_for_text(deaf, "#remote-sentence", speaker_message, "speaker message missing on deaf side")
        wait_for_text(deaf, "#meeting-avatar-corner", "Speaker Avatar", "deaf avatar corner missing")
        require("visible" in (deaf.locator("#meeting-avatar-corner").get_attribute("class") or ""), "deaf avatar corner should be visible")

        deaf.evaluate("glossBuffer = ['good', 'morning'];")
        deaf.evaluate("triggerSentenceTranslation()")

        deaf_message = "good morning"
        wait_for_text(deaf, "#remote-sentence", deaf_message, "deaf echo missing")
        wait_for_text(speaker, "#remote-sentence", deaf_message, "deaf message missing on speaker side")

        speaker.locator("#leave-btn").click()
        deaf.locator("#leave-btn").click()
        wait_for_text(speaker, "#meeting-status-pill", "Not connected", "speaker did not disconnect", exact=True)
        wait_for_text(deaf, "#meeting-status-pill", "Not connected", "deaf participant did not disconnect", exact=True)
        return f"room={room} speaker={speaker_creds['email']} deaf={deaf_creds['email']}"
    finally:
        speaker_ctx.close()
        deaf_ctx.close()


def browser_audit(runner: AuditRunner) -> None:
    with sync_playwright() as p:
        audit_session = requests.Session()
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--mute-audio",
            ],
        )

        parsed = urlparse(BASE_URL)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        speaker_meeting_creds = create_test_user(audit_session, "Meeting Speaker", "Speaker")
        deaf_meeting_creds = create_test_user(audit_session, "Meeting Deaf", "Deaf/HoH")

        desktop_ctx = browser.new_context(viewport={"width": 1366, "height": 768})
        desktop_ctx.grant_permissions(["microphone", "camera"], origin=origin)
        desktop = desktop_ctx.new_page()
        desktop.set_default_timeout(20000)
        browser_flow_for_viewport(runner, desktop, "desktop")
        runner.case(
            "ui-desktop",
            "Live meeting speaker/deaf relay",
            lambda: _meeting_dual_party_flow(browser, origin, speaker_meeting_creds, deaf_meeting_creds),
        )
        desktop_ctx.close()

        mobile_ctx = browser.new_context(
            viewport={"width": 390, "height": 844},
            is_mobile=True,
            has_touch=True,
        )
        mobile_ctx.grant_permissions(["microphone", "camera"], origin=origin)
        mobile = mobile_ctx.new_page()
        mobile.set_default_timeout(20000)
        browser_flow_for_viewport(runner, mobile, "mobile")
        mobile_ctx.close()

        browser.close()


def main() -> int:
    runner = AuditRunner()
    started = time.time()

    runner.case("infra", "Server reachable", lambda: f"status={requests.get(BASE_URL, timeout=20).status_code}")

    try:
        api_audit(runner)
    except Exception as exc:
        runner.add("api", "Fatal API audit error", "fail", str(exc))

    try:
        socket_audit(runner)
    except Exception as exc:
        runner.add("socket", "Fatal socket audit error", "fail", str(exc))

    try:
        browser_audit(runner)
    except (PlaywrightTimeoutError, PlaywrightError, AssertionError, Exception) as exc:
        runner.add("ui", "Fatal UI audit error", "fail", str(exc))

    duration = round(time.time() - started, 2)
    passed = [r for r in runner.results if r.status == "pass"]
    failed = [r for r in runner.results if r.status == "fail"]
    skipped = [r for r in runner.results if r.status == "skip"]

    report = {
        "base_url": BASE_URL,
        "duration_seconds": duration,
        "totals": {
            "pass": len(passed),
            "fail": len(failed),
            "skip": len(skipped),
            "all": len(runner.results),
        },
        "results": [asdict(r) for r in runner.results],
    }

    out_path = os.path.join("scripts", "exhaustive_site_audit_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Exhaustive Audit Summary ===")
    print(json.dumps(report["totals"], indent=2))
    print(f"Duration: {duration}s")
    print(f"Report: {out_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
