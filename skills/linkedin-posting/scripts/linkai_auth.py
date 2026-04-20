#!/usr/bin/env python3
"""LinkedIn OAuth helper for the LinkAI Claude Code plugin.

Subcommands:
  setup    Store LinkedIn app credentials (client_id / client_secret).
  login    Run the OAuth flow in your browser and save an access token.
  status   Show whether creds are set up and whether the token is valid.
  logout   Remove the access token (keeps app credentials).

Credentials live at ~/.linkai/credentials.json (0600 on Unix).
Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import string
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from random import SystemRandom
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse

REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}"
SCOPES = "openid profile email w_member_social"
AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
AUTH_TIMEOUT_SECS = 300

CREDS_DIR = Path.home() / ".linkai"
CREDS_PATH = CREDS_DIR / "credentials.json"


def load_creds() -> dict:
    if not CREDS_PATH.exists():
        return {}
    return json.loads(CREDS_PATH.read_text())


def save_creds(creds: dict) -> None:
    CREDS_DIR.mkdir(parents=True, exist_ok=True)
    CREDS_PATH.write_text(json.dumps(creds, indent=2))
    if sys.platform != "win32":
        os.chmod(CREDS_PATH, stat.S_IRUSR | stat.S_IWUSR)


def random_state(n: int = 32) -> str:
    rng = SystemRandom()
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


def http_json(url, data=None, headers=None):
    body = urlencode(data).encode() if data else None
    req = urllib_request.Request(url, data=body, headers=headers or {})
    try:
        with urllib_request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code} from {url}: {detail}") from None
    except URLError as e:
        raise SystemExit(f"Network error contacting {url}: {e.reason}") from None


class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict = {}

    def do_GET(self):
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        _CallbackHandler.result = params
        ok = "code" in params and "error" not in params
        body = _callback_html(ok, params)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        return


def _callback_html(ok: bool, params: dict) -> bytes:
    style = "font-family:system-ui,sans-serif;max-width:560px;margin:4em auto;padding:0 1em;"
    if ok:
        body = f"<html><body style='{style}'><h1>LinkAI is authorized.</h1><p>You can close this tab and return to your terminal.</p></body></html>"
    else:
        err = params.get("error_description", params.get("error", "Unknown error"))
        body = f"<html><body style='{style}'><h1>Authorization failed</h1><p>{err}</p><p>Return to your terminal and try again.</p></body></html>"
    return body.encode()


def oauth_flow(client_id: str, client_secret: str) -> dict:
    state = random_state()
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "state": state,
        "scope": SCOPES,
    }
    auth_url = f"{AUTH_URL}?{urlencode(params)}"

    _CallbackHandler.result = {}
    try:
        server = HTTPServer((REDIRECT_HOST, REDIRECT_PORT), _CallbackHandler)
    except OSError as e:
        raise SystemExit(
            f"Could not bind {REDIRECT_HOST}:{REDIRECT_PORT} ({e}). "
            f"Is another LinkAI login running, or is port {REDIRECT_PORT} in use?"
        ) from None

    server.timeout = AUTH_TIMEOUT_SECS
    try:
        print(f"Opening browser for LinkedIn authorization (listening on {REDIRECT_URI})…", flush=True)
        webbrowser.open(auth_url)
        server.handle_request()
    finally:
        server.server_close()

    result = _CallbackHandler.result
    if not result:
        raise SystemExit(f"No callback received within {AUTH_TIMEOUT_SECS}s. Try `login` again.")
    if "error" in result:
        raise SystemExit(f"LinkedIn returned an error: {result.get('error_description', result['error'])}")
    if result.get("state") != state:
        raise SystemExit("CSRF state mismatch; aborting. Try `login` again.")
    if "code" not in result:
        raise SystemExit("No authorization code returned. Try `login` again.")

    token_response = http_json(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": result["code"],
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if "access_token" not in token_response:
        raise SystemExit(f"Token exchange failed: {token_response}")

    userinfo = http_json(
        USERINFO_URL,
        headers={"Authorization": f"Bearer {token_response['access_token']}"},
    )

    return {
        "access_token": token_response["access_token"],
        "token_expires_at": int(time.time()) + int(token_response.get("expires_in", 0)),
        "user_sub": userinfo.get("sub"),
        "user_name": userinfo.get("name"),
    }


def cmd_setup(args: argparse.Namespace) -> None:
    creds = load_creds()
    creds["client_id"] = args.client_id
    creds["client_secret"] = args.client_secret
    save_creds(creds)
    print(f"Saved app credentials to {CREDS_PATH}")


def cmd_login(_: argparse.Namespace) -> None:
    creds = load_creds()
    if "client_id" not in creds or "client_secret" not in creds:
        raise SystemExit("App credentials missing. Run `setup` first.")
    tokens = oauth_flow(creds["client_id"], creds["client_secret"])
    creds.update(tokens)
    save_creds(creds)
    name = creds.get("user_name") or "your LinkedIn account"
    print(f"Logged in as {name}. Token saved to {CREDS_PATH}.")


def cmd_status(_: argparse.Namespace) -> None:
    creds = load_creds()
    if not creds:
        print(f"No credentials found at {CREDS_PATH}. Run `setup` to begin.")
        return
    print(f"Credentials file: {CREDS_PATH}")
    has_app = "client_id" in creds and "client_secret" in creds
    print(f"App credentials: {'present' if has_app else 'MISSING'}")
    if "access_token" in creds:
        expires_at = creds.get("token_expires_at", 0)
        secs = expires_at - int(time.time())
        if secs > 0:
            days = secs // 86400
            print(f"Access token: valid ({days} days remaining)")
        else:
            print("Access token: EXPIRED — run `login` to refresh")
    else:
        print("Access token: not yet issued — run `login`")
    if creds.get("user_name"):
        print(f"Authorized user: {creds['user_name']}")


def cmd_logout(_: argparse.Namespace) -> None:
    creds = load_creds()
    for k in ("access_token", "token_expires_at", "user_sub", "user_name"):
        creds.pop(k, None)
    save_creds(creds)
    print("Logged out. App credentials preserved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LinkedIn auth for the LinkAI Claude Code plugin.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup", help="Store LinkedIn app credentials.")
    p_setup.add_argument("--client-id", required=True)
    p_setup.add_argument("--client-secret", required=True)
    p_setup.set_defaults(func=cmd_setup)

    p_login = sub.add_parser("login", help="Run the LinkedIn OAuth browser flow.")
    p_login.set_defaults(func=cmd_login)

    p_status = sub.add_parser("status", help="Show auth status.")
    p_status.set_defaults(func=cmd_status)

    p_logout = sub.add_parser("logout", help="Clear saved access token.")
    p_logout.set_defaults(func=cmd_logout)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
