---
name: linkedin-posting
description: Use when the user wants to draft and publish a post on LinkedIn. Interviews them, drafts in their voice, and publishes on approval. Handles first-time LinkedIn app + OAuth setup automatically.
---

# LinkedIn Posting

Draft and publish LinkedIn posts from inside Claude Code. This skill walks a user from "I want to post about X" to a published post on their feed, including first-time authentication.

## When to use

Trigger on any of: "post this to LinkedIn", "draft a LinkedIn post", "share on LinkedIn", "post on my feed about …". Not for fetching posts or analytics — this skill only publishes.

## Flow

```
┌──────────────────────────────────────────┐
│ 1. Check auth status                     │
│    python scripts/linkai_auth.py status  │
└──────────────────────────────────────────┘
                   │
     ┌─────────────┼─────────────┬──────────────────┐
     ▼             ▼             ▼                  ▼
  no creds     missing token  expired token      valid
     │             │             │                  │
     ▼             ▼             ▼                  │
 ONBOARDING      LOGIN         LOGIN                │
 (below)      (below)        (below)                │
     └─────────────┴─────────────┴──────────────────┘
                                                    ▼
                                          ┌──────────────────┐
                                          │ 2. Interview     │
                                          │ 3. Draft         │
                                          │ 4. Revise        │
                                          │ 5. Publish       │
                                          └──────────────────┘
```

### 1. Check auth status first (this is the onboarding gate)

Before anything else, run the status check. The scripts live in `scripts/` relative to this SKILL.md — invoke them with their absolute path via Bash.

```bash
python "<skill-dir>/scripts/linkai_auth.py" status
```

The output tells you exactly which branch to take. **Do not start onboarding unless the output says "No credentials found".** Running onboarding on a user who has already set things up wastes their time.

| `status` output | What it means | What to do |
|---|---|---|
| `No credentials found at …` | First-time user | Full onboarding (section below) |
| `App credentials: present` + `Access token: not yet issued` | Dev app exists, never logged in | Skip to **Step B: LinkedIn login** |
| `Access token: EXPIRED` | Token issued >60 days ago | Skip to **Step B: LinkedIn login** |
| `Access token: valid (N days remaining)` | All good | Skip to the posting interview |

### Onboarding (only when status says "No credentials found")

Browser-driven and paced. Do not dump every step on the user at once. Open the relevant URL, wait for their confirmation, move to the next step. Use this Bash one-liner to open a URL in their browser on any OS:

```bash
python -c "import webbrowser; webbrowser.open('<url>')"
```

Before the walkthrough, ask one question: **"Do you already have a LinkedIn developer app you'd like to reuse, or should we create a new one together?"**

- If **reuse**: they need Client ID + Client Secret from their existing app, and `http://localhost:8765` in its *Authorized redirect URLs*. Skip to **Step A: Collect credentials**, but first remind them to add the redirect URI if it's not already there and open their app list with:
  ```bash
  python -c "import webbrowser; webbrowser.open('https://www.linkedin.com/developers/apps')"
  ```
- If **create new**: proceed with the full walkthrough below.

#### Create app (only if they chose "create new")

**Step 1 — open the new-app page.**
```bash
python -c "import webbrowser; webbrowser.open('https://www.linkedin.com/developers/apps/new')"
```
Tell them: "I just opened the LinkedIn *Create app* form. Fill in *App name* (anything, e.g. 'LinkAI for <your name>'), pick or create a *LinkedIn Page* (if you don't have one, click the inline *Create a new LinkedIn Page* link — a bare personal-brand page is fine; the app uses it for admin only, posts still go to your personal feed), add any *Privacy policy URL* (e.g. this repo's README link), upload any small image for *App logo*, then click *Create app*. Tell me when you're on the app's detail page."

Wait for confirmation. If they hit a verification prompt or get stuck, help them through it — don't move on.

**Step 2 — attach the two products.** Once they're in the app, tell them to open the **Products** tab. Ask them to confirm they're on it, then:
- "Find *Sign In with LinkedIn using OpenID Connect* and click *Request access*. It's instant."
- "Then find *Share on LinkedIn* and click *Request access*. Also instant."

Wait until both show "Added".

**Step 3 — add the redirect URI.** Tell them to open the **Auth** tab. Under *OAuth 2.0 settings → Authorized redirect URLs for your app*, they click the edit pencil and add exactly:
```
http://localhost:8765
```
No trailing slash, no `www`, no `https`. Save.

**Step 4 — grab Client ID and Client Secret.** Still on the Auth tab: tell them the *Client ID* and *Primary Client Secret* are right at the top of the page. Ask them to paste both into the chat.

#### Step A: Collect credentials

Once you have Client ID and Client Secret (from either the reuse or create path), run:
```bash
python "<skill-dir>/scripts/linkai_auth.py" setup --client-id "<id>" --client-secret "<secret>"
```
Expect: `Saved app credentials to ~/.linkai/credentials.json`.

#### Step B: LinkedIn login

```bash
python "<skill-dir>/scripts/linkai_auth.py" login
```
The browser opens, the user clicks **Allow**, the browser lands on a "LinkAI is authorized" page, and the CLI prints `Logged in as <Name>`. Relay the CLI output if anything goes wrong — don't silently retry.

When this succeeds, transition to the posting interview.

### Normal posting flow

When auth is good, run the four steps in order. Do not short-circuit.

1. **Interview.** Ask targeted questions to get the post's substance: what happened / what they're sharing, why it matters, who it's for, any specifics worth naming (people, companies, numbers, links). Probe for concrete details. Keep each question short and ask them one at a time unless clearly bundled.
2. **Draft.** Write the post in the user's voice based on what they told you. Match the tone they used in the interview — energetic if they were energetic, measured if they were measured. No "As an AI" framing, no forced persona, no opening "Hi, I'm …". Just a post that reads like the user wrote it.
3. **Revise.** Show the draft and ask for feedback. Iterate until they approve. Don't push back on their edits unless you have a concrete reason.
4. **Publish.** Only after explicit approval (e.g., "yes post it", "looks good, send it"). Invoke:

```bash
python "<skill-dir>/scripts/linkai_post.py" --text "<final post text>"
```

The CLI prints a URL like `https://www.linkedin.com/feed/update/urn:li:share:…/`. Share it with the user.

## Voice and tone

- Write as the user, not as a character. No personas, no signatures like "— Claude" or "— Link".
- Default to professional-but-human. Not corny, not corporate-hollow, not cliché-heavy.
- Match length to content: a quick thought is two sentences; a real story is four short paragraphs. Don't pad.
- Use the user's own phrases when they said something well during the interview.
- Emoji only if the user's inputs suggest they use them.

## Error recovery

- **HTTP 401 from post CLI** → it auto-runs `login` once and retries. No action needed.
- **Post CLI exits non-zero with "Credentials incomplete"** → fall back to onboarding.
- **Login fails with "Could not bind 127.0.0.1:8765"** → another process (or a previous hung `login`) is holding the port. Ask the user to close other LinkAI auth windows or restart their terminal, then retry.
- **Login fails with "No callback received within 300s"** → the user didn't complete the browser flow in time. Offer to retry.
- **LinkedIn returns 403 on post** → scope or product missing on the dev app. Point to `SETUP.md` and verify *Share on LinkedIn* is attached.

## Notes

- Tokens are 60-day; re-auth is a single browser click and the skill does it automatically on 401 — you don't need to warn the user unless they ask why the browser opened.
- Credentials live at `~/.linkai/credentials.json` (permissioned 0600 on Unix). Never print the file's contents in chat.
- Text-only posts only. If the user wants to attach an image, article, or poll, tell them that's not supported in this version and offer to post text-only instead.
