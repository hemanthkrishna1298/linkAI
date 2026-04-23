#!/usr/bin/env python3
"""Post text to LinkedIn via /rest/posts.

Usage:
  python linkai_post.py --text "your post body"

Reads credentials from ~/.linkai/credentials.json. If the access token is
expired or rejected, re-runs the OAuth login flow once and retries.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

SCRIPT_DIR = Path(__file__).resolve().parent
AUTH_SCRIPT = SCRIPT_DIR / "linkai_auth.py"
CREDS_PATH = Path.home() / ".linkai" / "credentials.json"

POSTS_URL = "https://api.linkedin.com/rest/posts"
LINKEDIN_VERSION = "202604"

# LinkedIn "Little Text Format" reserved characters — must be backslash-escaped in
# `commentary`, otherwise the parser interprets them as mention/hashtag/markup syntax
# and silently mangles the surrounding text (e.g. parens around a URL drop the URL).
# `#` is handled separately so inline `#hashtags` still linkify.
_LITTLE_TEXT_RESERVED = "|{}@[]()<>*_~"
_BARE_HASH = re.compile(r"#(?![A-Za-z0-9])")


def escape_commentary(text: str) -> str:
    out = text.replace("\\", "\\\\")
    for ch in _LITTLE_TEXT_RESERVED:
        out = out.replace(ch, "\\" + ch)
    return _BARE_HASH.sub(r"\\#", out)


def load_creds() -> dict:
    if not CREDS_PATH.exists():
        raise SystemExit(
            f"No credentials at {CREDS_PATH}. "
            f"Run `python {AUTH_SCRIPT} setup` and then `login` first."
        )
    return json.loads(CREDS_PATH.read_text())


def post_text(text: str, creds: dict) -> str:
    payload = {
        "author": f"urn:li:person:{creds['user_sub']}",
        "commentary": escape_commentary(text),
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    req = urllib_request.Request(
        POSTS_URL,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {creds['access_token']}",
            "Content-Type": "application/json",
            "LinkedIn-Version": LINKEDIN_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
        },
    )
    with urllib_request.urlopen(req) as resp:
        urn = resp.headers.get("x-restli-id") or resp.headers.get("X-RestLi-Id")
        if not urn:
            body = resp.read().decode(errors="replace")
            raise SystemExit(f"Post succeeded but no x-restli-id header returned. Body: {body}")
        return urn


def run_login() -> None:
    print("Re-running LinkedIn login…", flush=True)
    result = subprocess.run([sys.executable, str(AUTH_SCRIPT), "login"])
    if result.returncode != 0:
        raise SystemExit("Re-authentication failed.")


def token_expired(creds: dict) -> bool:
    return time.time() >= creds.get("token_expires_at", 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post text to LinkedIn.")
    parser.add_argument("--text", required=True, help="Post body text.")
    args = parser.parse_args()

    creds = load_creds()
    if "user_sub" not in creds or "access_token" not in creds:
        raise SystemExit(f"Credentials incomplete. Run `python {AUTH_SCRIPT} login`.")

    if token_expired(creds):
        run_login()
        creds = load_creds()

    try:
        urn = post_text(args.text, creds)
    except HTTPError as e:
        if e.code == 401:
            run_login()
            creds = load_creds()
            try:
                urn = post_text(args.text, creds)
            except HTTPError as e2:
                detail = e2.read().decode(errors="replace")
                raise SystemExit(f"Post failed after re-auth: HTTP {e2.code}: {detail}") from None
        else:
            detail = e.read().decode(errors="replace")
            raise SystemExit(f"Post failed: HTTP {e.code}: {detail}") from None
    except URLError as e:
        raise SystemExit(f"Network error: {e.reason}") from None

    print(f"https://www.linkedin.com/feed/update/{urn}/")


if __name__ == "__main__":
    main()
