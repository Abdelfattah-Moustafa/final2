#!/usr/bin/env python3
"""Repeatable browser regression smoke pack for desktop + mobile viewports.

Usage (PowerShell):
  $env:E2E_EMAIL='user@example.com'
  $env:E2E_PASSWORD='your-password'
  C:/Users/abody/AppData/Local/Python/pythoncore-3.11-64/python.exe scripts/regression_browser_pack.py

Optional env vars:
  BASE_URL (default: http://localhost:8000)
  E2E_HEADLESS (default: 1; set 0 for headed)
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
E2E_EMAIL = os.getenv("E2E_EMAIL", "").strip()
E2E_PASSWORD = os.getenv("E2E_PASSWORD", "").strip()
HEADLESS = os.getenv("E2E_HEADLESS", "1") != "0"

MATRIX = [
    {
        "name": "desktop",
        "context": {
            "viewport": {"width": 1366, "height": 768},
        },
    },
    {
        "name": "mobile",
        "context": {
            "viewport": {"width": 390, "height": 844},
            "is_mobile": True,
            "has_touch": True,
        },
    },
]


class Runner:
    def __init__(self) -> None:
        self.failures: List[str] = []

    def case(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # broad by design for concise CI-like report
            self.failures.append(name)
            print(f"[FAIL] {name}: {exc}")


def test_landing_links(page: Page) -> None:
    page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
    page.get_by_role("link", name="Get Started").first.click()
    page.wait_for_url("**/signup", timeout=10000)

    page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
    page.get_by_role("link", name="Sign In").first.click()
    page.wait_for_url("**/login", timeout=10000)


def test_login_invalid_email(page: Page) -> None:
    page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded")
    page.locator("#inp-email").fill("not-an-email")
    page.locator("#inp-password").fill("12345678")
    page.locator("#submit-btn").click()
    page.get_by_text("Please enter a valid email address.").first.wait_for(timeout=5000)


def login(page: Page) -> None:
    if not E2E_EMAIL or not E2E_PASSWORD:
        raise RuntimeError("E2E_EMAIL / E2E_PASSWORD are required for authenticated tests")

    page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded")
    page.locator("#inp-email").fill(E2E_EMAIL)
    page.locator("#inp-password").fill(E2E_PASSWORD)
    page.locator("#submit-btn").click()
    page.wait_for_url("**/dashboard", timeout=15000)
    page.locator("#topbar-title").wait_for(timeout=10000)


def switch_to_meeting_module(page: Page) -> None:
    mobile_picker = page.locator("#mobile-module-select")
    if mobile_picker.is_visible():
        mobile_picker.select_option(label="Live Meeting")
        return

    page.locator("#nav-meeting").click()
    page.get_by_text("Live Meeting").first.wait_for(timeout=8000)


def test_meeting_flow(page: Page) -> None:
    switch_to_meeting_module(page)
    page.get_by_role("button", name="New meeting").first.click()
    page.get_by_text("Connected").first.wait_for(timeout=8000)

    leave_call = page.locator("#leave-btn")
    leave_call.wait_for(timeout=5000)
    leave_call.click()
    page.get_by_text("Not connected").first.wait_for(timeout=8000)


def run_matrix() -> int:
    runner = Runner()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=HEADLESS)

        for config in MATRIX:
            name = config["name"]
            context = browser.new_context(**config["context"])
            page = context.new_page()
            page.set_default_timeout(15000)

            print(f"\n=== Running {name} regression pack ===")
            runner.case(f"{name}: landing links", lambda: test_landing_links(page))
            runner.case(f"{name}: invalid email validation", lambda: test_login_invalid_email(page))

            if E2E_EMAIL and E2E_PASSWORD:
                runner.case(f"{name}: login", lambda: login(page))
                runner.case(f"{name}: meeting create/leave", lambda: test_meeting_flow(page))
            else:
                print(f"[SKIP] {name}: authenticated cases skipped (missing E2E_EMAIL/E2E_PASSWORD)")

            context.close()

        browser.close()

    print("\n=== Regression Summary ===")
    if runner.failures:
        for idx, item in enumerate(runner.failures, start=1):
            print(f"{idx}. {item}")
        return 1

    print("All regression cases passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_matrix())
    except PlaywrightTimeoutError as exc:
        print(f"[FAIL] Timeout: {exc}")
        raise SystemExit(1)
