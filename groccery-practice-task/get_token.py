#!/usr/bin/env python3
"""Fetch fresh WebSocket game tokens from app.ainm.no/challenge using Playwright.

Uses a persistent browser session so you only need to authenticate (Google OAuth)
once — subsequent runs reuse the saved browser state.

Tracks rate limits (60s cooldown, 40/hour, 300/day) in .game-tracker.json.

Usage:
    uv run python get_token.py easy              # Fetch token for one difficulty
    uv run python get_token.py --all             # Fetch tokens for all difficulties
    uv run python get_token.py --status          # Show rate limit status
    uv run python get_token.py --login           # Just open browser for manual login
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import (
    sync_playwright,
    BrowserContext,
    Page,
    TimeoutError as PwTimeout,
)

PROJECT_ROOT = Path(__file__).resolve().parent
BROWSER_DATA_DIR = PROJECT_ROOT / ".browser-data"
TRACKER_FILE = PROJECT_ROOT / ".game-tracker.json"
ENV_FILE = PROJECT_ROOT / ".env"
CHALLENGE_URL = "https://app.ainm.no/challenge"

DIFFICULTIES = ("easy", "medium", "hard", "expert", "nightmare")

COOLDOWN_SECONDS = 60
MAX_PER_HOUR = 40
MAX_PER_DAY = 300

NAV_TIMEOUT = 30_000
ACTION_TIMEOUT = 15_000
TOKEN_WAIT_TIMEOUT = 20_000

WSS_PATTERN = re.compile(r"wss://game\.ainm\.no/ws\?token=[A-Za-z0-9._-]{20,}")


def _load_tracker() -> dict:
    if TRACKER_FILE.exists():
        try:
            return json.loads(TRACKER_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {"requests": []}


def _save_tracker(data: dict) -> None:
    TRACKER_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _prune_old_entries(data: dict) -> dict:
    cutoff = time.time() - 86400
    data["requests"] = [r for r in data["requests"] if r["timestamp"] > cutoff]
    return data


def record_request(difficulty: str) -> None:
    data = _prune_old_entries(_load_tracker())
    data["requests"].append(
        {
            "timestamp": time.time(),
            "difficulty": difficulty,
            "time_human": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save_tracker(data)


def check_rate_limits() -> tuple[bool, str]:
    """Check if we can make a request. Returns (allowed, reason)."""
    data = _prune_old_entries(_load_tracker())
    now = time.time()
    requests = data["requests"]

    if not requests:
        return True, "No previous requests"

    last_ts = max(r["timestamp"] for r in requests)
    elapsed = now - last_ts
    if elapsed < COOLDOWN_SECONDS:
        remaining = COOLDOWN_SECONDS - elapsed
        return False, f"Cooldown: wait {remaining:.0f}s (60s between games)"

    hour_ago = now - 3600
    hourly_count = sum(1 for r in requests if r["timestamp"] > hour_ago)
    if hourly_count >= MAX_PER_HOUR:
        return (
            False,
            f"Hourly limit reached: {hourly_count}/{MAX_PER_HOUR} games this hour",
        )

    day_ago = now - 86400
    daily_count = sum(1 for r in requests if r["timestamp"] > day_ago)
    if daily_count >= MAX_PER_DAY:
        return False, f"Daily limit reached: {daily_count}/{MAX_PER_DAY} games today"

    return True, "OK"


def get_quota_status() -> str:
    """Return a human-readable rate limit status string."""
    data = _prune_old_entries(_load_tracker())
    _save_tracker(data)
    now = time.time()
    requests = data["requests"]

    hour_ago = now - 3600
    day_ago = now - 86400
    hourly_count = sum(1 for r in requests if r["timestamp"] > hour_ago)
    daily_count = sum(1 for r in requests if r["timestamp"] > day_ago)

    lines = [
        "=== Rate Limit Status ===",
        f"  Hourly:  {hourly_count}/{MAX_PER_HOUR} ({MAX_PER_HOUR - hourly_count} remaining)",
        f"  Daily:   {daily_count}/{MAX_PER_DAY} ({MAX_PER_DAY - daily_count} remaining)",
    ]

    if requests:
        last_ts = max(r["timestamp"] for r in requests)
        elapsed = now - last_ts
        if elapsed < COOLDOWN_SECONDS:
            lines.append(f"  Cooldown: {COOLDOWN_SECONDS - elapsed:.0f}s remaining")
        else:
            lines.append(f"  Cooldown: ready ({elapsed:.0f}s since last)")

        recent = sorted(requests, key=lambda r: r["timestamp"], reverse=True)[:5]
        lines.append("\n  Recent requests:")
        for r in recent:
            t = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
            lines.append(f"    {t.strftime('%H:%M:%S UTC')} - {r['difficulty']}")
    else:
        lines.append("  No requests recorded today")

    return "\n".join(lines)


def update_env(difficulty: str, wss_url: str) -> None:
    key = f"WS_ENDPOINT_{difficulty.upper()}"
    new_line = f"{key}={wss_url}"

    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()
    else:
        lines = []

    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = new_line
            found = True
            break

    if not found:
        lines.append(new_line)

    ENV_FILE.write_text("\n".join(lines) + "\n")
    print(f"  Updated .env: {key}")


def _create_context(playwright) -> BrowserContext:
    BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(BROWSER_DATA_DIR),
        headless=False,
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        timezone_id="Europe/Oslo",
        args=["--disable-blink-features=AutomationControlled"],
    )
    return context


def _wait_for_auth(page: Page) -> bool:
    """Detect auth state after SPA hydration. Prompts user to sign in if needed."""
    page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
    time.sleep(2)

    try:
        page.wait_for_selector(
            "text=/easy|Easy|Play|Start/i",
            timeout=5_000,
        )
        return True
    except PwTimeout:
        pass

    print("\n" + "=" * 60)
    print("  AUTHENTICATION REQUIRED")
    print("  Please sign in with Google in the browser window.")
    print("  The script will continue automatically after login.")
    print("=" * 60 + "\n")

    try:
        page.wait_for_selector(
            "text=/easy|Easy|Play|Start/i",
            timeout=300_000,
        )
        print("  Authentication detected! Continuing...\n")
        time.sleep(2)
        return True
    except PwTimeout:
        print("  Authentication timed out (5 min). Aborting.")
        return False


def _extract_wss_url_from_page(page: Page) -> str | None:
    """Scan code blocks, inputs, and raw page content for wss:// URL."""
    try:
        elements = page.query_selector_all(
            "[class*='mono'], code, pre, input, [class*='code'], [class*='token'], textarea"
        )
        for el in elements:
            text = (
                el.inner_text() if el.is_visible() else el.get_attribute("value") or ""
            )
            match = WSS_PATTERN.search(text)
            if match:
                return match.group(0)
    except Exception:
        pass

    try:
        content = page.content()
        match = WSS_PATTERN.search(content)
        if match:
            return match.group(0)
    except Exception:
        pass

    return None


def fetch_token(
    difficulty: str, *, context: BrowserContext | None = None, playwright_inst=None
) -> str | None:
    """Fetch a fresh WSS token for the given difficulty. Returns wss:// URL or None."""
    owns_context = context is None
    pw = None

    try:
        if context is None:
            pw = sync_playwright().start()
            context = _create_context(pw)

        page = context.new_page()
        captured_wss: list[str] = []

        def _on_response(response):
            """Intercept API responses — token may arrive via JSON from a play endpoint."""
            try:
                body = response.text()
                match = WSS_PATTERN.search(body)
                if match:
                    captured_wss.append(match.group(0))
            except Exception:
                pass

        page.on("response", _on_response)

        print(f"  Navigating to {CHALLENGE_URL}...")
        page.goto(CHALLENGE_URL, timeout=NAV_TIMEOUT, wait_until="networkidle")

        if not _wait_for_auth(page):
            page.close()
            return None

        # Page shows difficulty cards as buttons like:
        #   button "Easy 1 bot 12×10 grid · Solo pathfinding No score yet"
        # Clicking a card triggers POST /games/request which returns the WSS token.
        # There is no separate "Play" button.
        print(f"  Clicking '{difficulty}' card...")

        diff_label = difficulty.capitalize()
        card_selectors = [
            f"button:has-text('{diff_label}')",
            f"button:has-text('{difficulty}')",
        ]

        clicked = False
        for selector in card_selectors:
            try:
                el = page.locator(selector).first
                if el.is_visible(timeout=3_000):
                    el.click(timeout=ACTION_TIMEOUT)
                    clicked = True
                    print(f"  Clicked {diff_label} card")
                    break
            except (PwTimeout, Exception):
                continue

        if not clicked:
            print(f"  Could not find '{diff_label}' card — please click it manually.")
            time.sleep(5)

        print("  Waiting for WSS token...")
        deadline = time.time() + TOKEN_WAIT_TIMEOUT / 1000

        wss_url = None
        while time.time() < deadline:
            if captured_wss:
                wss_url = captured_wss[-1]
                break

            time.sleep(0.5)

        if not wss_url:
            wss_url = _extract_wss_url_from_page(page)

        if not wss_url:
            print("\n  Could not auto-extract WSS URL.")
            print("  Please copy the wss:// URL from the page and paste it here:")
            try:
                wss_url = input("  WSS URL> ").strip()
                if not wss_url.startswith("wss://game.ainm.no/ws?token="):
                    print("  Invalid URL format.")
                    wss_url = None
            except (EOFError, KeyboardInterrupt):
                wss_url = None

        page.close()
        return wss_url

    finally:
        if owns_context:
            try:
                context.close()
            except Exception:
                pass
            if pw:
                try:
                    pw.stop()
                except Exception:
                    pass


def login_only() -> None:
    print("Opening browser for manual login...")
    print("Sign in with Google, then close the browser when done.\n")

    with sync_playwright() as pw:
        context = _create_context(pw)
        page = context.new_page()
        page.goto(CHALLENGE_URL, timeout=NAV_TIMEOUT, wait_until="networkidle")

        print("Browser is open. Sign in and then press Enter here to save state...")
        try:
            input("Press Enter when done> ")
        except (EOFError, KeyboardInterrupt):
            pass

        page.close()
        context.close()

    print("Browser state saved to .browser-data/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch fresh WebSocket game tokens from app.ainm.no/challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_token.py easy          Fetch token for easy difficulty
  python get_token.py --all         Fetch tokens for all difficulties
  python get_token.py --status      Show rate limit status
  python get_token.py --login       Open browser for manual authentication
        """,
    )
    parser.add_argument(
        "difficulty",
        nargs="?",
        choices=DIFFICULTIES,
        help="Difficulty level to fetch token for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch tokens for all difficulties",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show rate limit status and exit",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Open browser for manual authentication only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip rate limit checks (use with caution!)",
    )

    args = parser.parse_args()

    if args.status:
        print(get_quota_status())
        return

    if args.login:
        login_only()
        return

    if not args.difficulty and not args.all:
        parser.print_help()
        sys.exit(1)

    targets = list(DIFFICULTIES) if args.all else [args.difficulty]

    print(get_quota_status())
    print()

    success_count = 0
    with sync_playwright() as pw:
        context = _create_context(pw)

        for i, diff in enumerate(targets):
            print(
                f"\n--- Fetching token for: {diff.upper()} ({i + 1}/{len(targets)}) ---"
            )

            if not args.force:
                allowed, reason = check_rate_limits()
                if not allowed:
                    print(f"  Rate limited: {reason}")
                    if len(targets) > 1:
                        print("  Skipping remaining difficulties.")
                    break

            wss_url = fetch_token(diff, context=context)

            if wss_url:
                print(f"  Got token: {wss_url[:60]}...")
                update_env(diff, wss_url)
                record_request(diff)
                success_count += 1
            else:
                print(f"  FAILED to get token for {diff}")

            if args.all and i < len(targets) - 1:
                print(f"\n  Waiting {COOLDOWN_SECONDS}s cooldown before next fetch...")
                for remaining in range(COOLDOWN_SECONDS, 0, -1):
                    print(
                        f"\r  Cooldown: {remaining}s remaining...  ", end="", flush=True
                    )
                    time.sleep(1)
                print()

        context.close()

    print(f"\n=== Done: {success_count}/{len(targets)} tokens fetched ===")
    if success_count > 0:
        print("Run your bot with: python main.py <difficulty>")


if __name__ == "__main__":
    main()
