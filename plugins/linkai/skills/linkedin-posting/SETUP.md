# LinkedIn Dev App Setup

One-time setup to let the LinkAI plugin post to your LinkedIn feed. Takes ~5 minutes.

## 1. Create a LinkedIn developer app

Go to https://www.linkedin.com/developers/apps and click **Create app**.

Fill in:
- **App name** — anything. e.g. "LinkAI for <Your Name>".
- **LinkedIn Page** — LinkedIn requires every developer app to be associated with a Company Page. If you don't have one, click the *Create a new LinkedIn Page* link and create a minimal personal-brand page; you can leave it blank. The app only uses the page for admin purposes; posts still go to your personal feed.
- **Privacy policy URL** — any valid URL is fine for personal use (e.g. a link to this repo's README, or `https://example.com/privacy`).
- **App logo** — any PNG. LinkedIn just needs something uploaded.

Click **Create app** and verify you're an admin of the company page.

## 2. Attach the required products

Open the **Products** tab on your app and request:

1. **Sign In with LinkedIn using OpenID Connect** — approval is instant.
2. **Share on LinkedIn** — approval is instant.

Wait for both products to show "Added" (usually a few seconds).

## 3. Add the redirect URI

Open the **Auth** tab. Under *OAuth 2.0 settings → Authorized redirect URLs for your app*, click the edit pencil and add exactly:

```
http://localhost:8765
```

Save.

(Note: LinkedIn accepts `http://localhost` for dev use even though it otherwise requires HTTPS. The plugin listens on port 8765 during OAuth.)

## 4. Copy your Client ID and Client Secret

Still on the **Auth** tab, copy:
- **Client ID**
- **Primary Client Secret**

Paste both into your chat with Claude when it asks — it will save them locally (`~/.linkai/credentials.json`, chmod 0600 on Unix). These are the only two values you need to remember for this skill; Claude will prompt for them the first time you post.

## What the plugin does with these

- **Client ID / Secret** identify your app to LinkedIn during OAuth.
- **Redirect URI (`http://localhost:8765`)** is where LinkedIn sends you back after you approve the permission request. The plugin spins up a tiny local listener on that port, catches the callback, closes, and exchanges the code for an access token. No copy-pasting URLs.
- **Scopes requested:** `openid profile email w_member_social` — enough to know who you are and to post on your behalf.
- **Access token lifetime:** 60 days. After that, one re-authorization click and you're back in. Claude handles this automatically when posts start failing with 401.

## Troubleshooting

**"Redirect_uri doesn't match"** during login — the URI in your dev app's Auth tab isn't exactly `http://localhost:8765`. No trailing slash, no `www`, no https. Re-save and retry.

**Products tab won't let you add "Share on LinkedIn"** — occasionally LinkedIn requires your app's company page to be verified. Follow the on-page prompts; verification is usually instant.

**Token works once, then dies** — that's normal if 60+ days have passed. Trigger any post and the plugin will re-auth automatically.

**You want to revoke access entirely** — visit https://www.linkedin.com/psettings/permitted-services, find your app, and click **Remove**. Then delete `~/.linkai/credentials.json` locally.
