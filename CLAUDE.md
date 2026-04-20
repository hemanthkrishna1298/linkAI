# CLAUDE.md

Guidance for future Claude Code sessions working on this repository.

## What this repo is

LinkAI — a Claude Code plugin that lets a user draft and publish LinkedIn posts from a chat with Claude. One skill, two CLI scripts, stdlib only.

## Layout

```
.claude-plugin/plugin.json        # plugin manifest (lives here, not at root)
README.md                         # user-facing intro + install instructions
LICENSE                           # MIT
skills/
  linkedin-posting/
    SKILL.md                      # Claude's playbook: onboarding → interview → draft → publish
    SETUP.md                      # human-facing LinkedIn dev-app walkthrough
    scripts/
      linkai_auth.py              # OAuth helper: setup / login / status / logout
      linkai_post.py              # POST /rest/posts with --text
```

## Running the scripts locally (without a plugin install)

```bash
python skills/linkedin-posting/scripts/linkai_auth.py status
python skills/linkedin-posting/scripts/linkai_auth.py setup --client-id ID --client-secret SECRET
python skills/linkedin-posting/scripts/linkai_auth.py login
python skills/linkedin-posting/scripts/linkai_post.py --text "hello"
```

No dependencies to install. Standard library only. Python 3.9+.

## Load-bearing design decisions

- **Stdlib-only.** No `requests`, no `pip install`. Keeps the install story for end users to just `/plugin install <url>`. If you're tempted to add a dependency, revisit this trade-off first.
- **Credentials live at `~/.linkai/credentials.json`** (chmod 0600 on Unix). Never in the repo. This lets the skill work from any CWD.
- **Redirect URI is hardcoded to `http://localhost:8765`.** Users must add this exact string to their LinkedIn dev app's Auth tab. A background `HTTPServer` on that port catches the OAuth callback so the user never pastes a URL.
- **Posts go through `POST /rest/posts`** (the modern versioned API), not the deprecated `/v2/ugcPosts`. Required headers: `Authorization: Bearer …`, `LinkedIn-Version: 202604`, `X-Restli-Protocol-Version: 2.0.0`, `Content-Type: application/json`. Update the `LINKEDIN_VERSION` constant in `linkai_post.py` if LinkedIn deprecates `202604`.
- **No refresh tokens.** LinkedIn only issues them to MDP-approved partners. Non-MDP apps (which this is) get 60-day access tokens and must re-auth. `linkai_post.py` handles this by invoking `linkai_auth.py login` on HTTP 401 and retrying once — silent from the user's perspective beyond the browser popup.
- **Persona: none.** The skill writes posts in the user's voice. No "Hi, I'm Link" opener, no AI signature. This is a deliberate reversal of the old architecture.

## OAuth testing caveat

`linkai_auth.py login` cannot be driven end-to-end from a Claude Code session alone — it opens a browser and blocks on the user clicking **Allow** on LinkedIn's site. When working on the auth code, either:
1. Run `login` yourself in a terminal and observe the CLI output, or
2. Unit-test the individual pieces (`oauth_flow` internals, `_CallbackHandler`, `http_json`) with a mock LinkedIn response.

Do not try to "fix" the blocking behavior — it is the correct shape for this flow.

## Plugin manifest

`.claude-plugin/plugin.json` is the canonical location. Claude Code discovers skills from the sibling `skills/` directory automatically; no explicit skill listing is needed in the manifest.

## When you change SKILL.md

That file is the agent's entire instruction set at runtime. Treat changes to it the way you would treat changes to a production system prompt:
- Test the onboarding branch (no creds) and the happy path (valid token) before declaring done.
- Keep imperatives directed at Claude ("Run the status check first"), not at the human user.
- Don't reintroduce a persona or a fixed post opener — those are explicitly out of scope.
