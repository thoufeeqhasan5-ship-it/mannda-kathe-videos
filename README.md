# Mannda Kathe (ಮಣ್ಣ್‌ದ ಕತೆ) — Tulu video maker

Turns a Tulu script in `topics/` into a 9:16 short with voice, Wikimedia Commons photos, captions and music.
Runs free on GitHub Actions.

## Make a video
Actions → **Make Tulu video** → Run workflow → topic `01`, voice `Suresh` → Run.

## Add a new video
Add `topics/02.json`, `topics/03.json` … in the same format as `topics/01.json`.

## Use a real Tulu voice
Upload a recording as `voice/01.m4a` (same number as the topic). It is used instead of the AI voice.

## Needs
The secret `HF_TOKEN` (a free Hugging Face Read token) in Settings → Secrets and variables → Actions.
