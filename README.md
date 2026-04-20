# LinkAI

A Claude Code plugin that lets Claude draft and publish LinkedIn posts on your behalf.

Tell Claude what you want to share; it interviews you, writes a post in your voice, iterates until you're happy, and publishes to your feed. Text-only posts for now.

## Install

```
/plugin install https://github.com/hemanthkrishna1298/linkai
```

Or clone the repo and copy the `skills/linkedin-posting/` directory into `~/.claude/skills/` if you prefer not to use the plugin system.

## Setup (first time)

You need a LinkedIn developer app so Claude can post on your behalf. Claude walks you through this the first time you ask for a post — or see [`skills/linkedin-posting/SETUP.md`](skills/linkedin-posting/SETUP.md) for the manual steps. The short version:

1. Create an app at https://www.linkedin.com/developers/apps
2. Attach the *Sign In with LinkedIn using OpenID Connect* and *Share on LinkedIn* products
3. Add `http://localhost:8765` as an authorized redirect URI
4. Hand Claude your Client ID and Client Secret when it asks

Claude stores them at `~/.linkai/credentials.json` (chmod 0600 on Unix). The first `login` opens your browser, you click **Allow**, and you're done.

## Usage

Just tell Claude. Examples:

> "Draft a LinkedIn post about the migration I finished last week."
>
> "I want to share that I just joined a new team. Make it friendly, short."
>
> "Post about my weekend project — a Claude Code plugin for LinkedIn. Tech-audience voice."

Claude asks a few questions, drafts, lets you edit, and publishes on your approval. It returns the post URL.

## Requirements

- **Python 3.9+** on your path. No third-party packages — the scripts use only the standard library.
- **Claude Code** with plugin support.
- A **LinkedIn account** willing to create a developer app (free, ~5 min).

## How it works

```
skills/linkedin-posting/
├── SKILL.md            # Claude's playbook — interview, draft, publish
├── SETUP.md            # One-time LinkedIn dev-app walkthrough
└── scripts/
    ├── linkai_auth.py  # OAuth: setup / login / status / logout
    └── linkai_post.py  # POST /rest/posts with --text
```

No server, no daemon. Each script runs as a one-shot command. OAuth uses LinkedIn's standard Authorization Code flow with a short-lived local HTTP listener on `127.0.0.1:8765` to catch the callback — no paste-URL dance.

## Re-auth cadence

LinkedIn access tokens live 60 days and the plugin can't silently refresh them (LinkedIn only issues refresh tokens to Marketing Developer Platform partners). When your token expires, the next post will trigger a one-click browser re-authorization automatically.

## Limitations

- Text-only posts. No images, articles, polls, or multi-image.
- Personal feed only. No Company Page posting.
- Public visibility only (`MAIN_FEED`, `PUBLIC`). No targeted or private posts.

These are scope choices, not hard limits — happy to accept PRs that add them.

## License

See [LICENSE](LICENSE).
