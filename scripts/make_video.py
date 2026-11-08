#!/usr/bin/env python3
"""
Mannda Kathe — makes one Tulu short (1080x1920 MP4).

  python scripts/make_video.py --topic 05
  python scripts/make_video.py --topic 05 --speaker Anu
  python scripts/make_video.py --topic 05 --test     # no internet: tone instead of voice, placeholder photos

Voice:  AI4Bharat Indic Parler-TTS (Apache-2.0), Kannada speaker Suresh (default) or Anu.
        If a file voice/<topic>.wav or .m4a exists in the repo, that recording is used instead.
Photos: Wikimedia Commons, free licences only, credit shown on screen.
Output: out/video.mp4, out/voice.wav, out/caption.txt, out/credits.txt
"""
import argparse, json, math, os, random, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK, OUT = ROOT / "work", ROOT / "out"
FPS, W, H = 30, 1080, 1920
LEAD, GAP, END_CARD = 0.45, 0.30, 2.6
UA = {"User-Agent": "ManndaKatheBot/1.0 (https://github.com; Tulu heritage shorts)"}
OK_LICENCE = re.compile(r"^(cc0|public domain|pd|cc by(-sa)?( \d\.\d)?( [a-z\-]+)?|cc-by(-sa)?-\d\.\d.*)$", re.I)


def log(*a):
    print("[manndakathe]", *a, flush=True)


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        log("FAILED:", " ".join(map(str, cmd))[:400]); log(r.stderr[-2500:]); sys.exit(1)
    return r.stdout


def dur(p):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]).strip())


# ---------------------------------------------------------------- voice
def expected_seconds(text):
    return max(1.2, len(re.sub(r"\s", "", text)) / 11.5)    # rough Kannada speaking rate


def tts_parler(phrases, speaker, outdir):
    import torch, soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 2)
    name = "ai4bharat/indic-parler-tts"
    log("loading", name)
    model = ParlerTTSForConditionalGeneration.from_pretrained(name).eval()
    tok = AutoTokenizer.from_pretrained(name)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    desc = (f"{speaker} speaks in Kannada with a warm, expressive storytelling tone at a moderate pace. "
            f"The recording is very clear and close, with no background noise.")
    d = dtok(desc, return_tensors="pt")
    files = []
    for i, ph in enumerate(phrases):
        text = ph.get("say", ph["text"]).replace("…", ",")   # "say" = numbers written as words
        exp = expected_seconds(text)
        best = None
        for attempt in range(3):
            torch.manual_seed(1234 + attempt * 97)
            p = tok(text, return_tensors="pt")
            t0 = time.time()
            with torch.no_grad():
                g = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                   prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            audio = g.cpu().numpy().squeeze()
            raw = outdir / f"raw_{i:02d}_{attempt}.wav"
            sf.write(raw, audio, model.config.sampling_rate)
            clean = outdir / f"p_{i:02d}_{attempt}.wav"
            trim(raw, clean)
            secs = dur(clean)
            ratio = secs / exp
            log(f"phrase {i + 1}/{len(phrases)} try {attempt + 1}: {secs:.1f}s (expected ~{exp:.1f}s) in {time.time() - t0:.0f}s")
            score = abs(math.log(max(ratio, 1e-3)))
            if best is None or score < best[0]:
                best = (score, clean)
            if 0.6 <= ratio <= 1.8:
                break
        final = outdir / f"phrase_{i:02d}.wav"
        shutil.copy(best[1], final)
        files.append(final)
    return files


def trim(src, dst):
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af",
         "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
         "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,areverse,"
         "aresample=44100", "-ac", "1", dst])


def tts_test(phrases, outdir):
    files = []
    for i, ph in enumerate(phrases):
        f = outdir / f"phrase_{i:02d}.wav"
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             f"sine=frequency={180 + 20 * (i % 4)}:sample_rate=44100:d={expected_seconds(ph.get('say', ph['text'])):.2f}",
             "-af", "tremolo=f=4:d=0.6,volume=0.4", "-ac", "1", f])
        files.append(f)
    return files


# ---------------------------------------------------------------- photos
def commons_search(query, n=25):
    import requests
    r = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=60, params={
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": n, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1600})
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or {}
    out = []
    for p in sorted(pages.values(), key=lambda x: x.get("index", 99)):
        ii = (p.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata") or {}
        lic = re.sub(r"<[^>]+>", "", (md.get("LicenseShortName") or {}).get("value", "")).strip()
        artist = re.sub(r"<[^>]+>", "", (md.get("Artist") or {}).get("value", "")).strip()
        if ii.get("mime") not in ("image/jpeg", "image/png") or not OK_LICENCE.match(lic):
            continue
        if min(ii.get("width", 0), ii.get("height", 0)) < 700:
            continue
        out.append({"title": p["title"], "url": ii.get("thumburl") or ii["url"], "licence": lic,
                    "artist": re.sub(r"\s+", " ", artist)[:60] or "Unknown", "page": ii.get("descriptionurl", "")})
    return out


def fetch_photos(shots, outdir):
    import requests
    used, got = set(), []
    for k, s in enumerate(shots):
        chosen = None
        for q in s["q"]:
            try:
                for c in commons_search(q):
                    if c["title"] not in used:
                        chosen = c; break
            except Exception as e:
                log(f"search failed '{q}': {e}")
            if chosen:
                break
        if not chosen:
            log(f"shot {k}: no free photo found, reusing previous")
            got.append(got[-1] if got else None); continue
        used.add(chosen["title"])
        path = outdir / f"photo_{k:02d}.jpg"
        r = requests.get(chosen["url"], headers=UA, timeout=120); r.raise_for_status()
        path.write_bytes(r.content)
        chosen["path"] = path
        log(f"shot {k}: {chosen['title']} ({chosen['licence']}, {chosen['artist']})")
        got.append(chosen)
    return got


def test_photos(shots, outdir):
    from PIL import Image, ImageDraw
    got = []
    for k, _ in enumerate(shots):
        p = outdir / f"photo_{k:02d}.jpg"
        im = Image.new("RGB", (1600, 1100 if k % 2 else 2000), (60 + 20 * k % 160, 40 + 15 * k % 120, 30))
        dr = ImageDraw.Draw(im)
        for i in range(0, 1600, 100):
            dr.line([(i, 0), (i + 600, 2000)], fill=(120, 90, 60), width=6)
        im.save(p, quality=90)
        got.append({"path": p, "title": f"Test photo {k}", "licence": "CC BY-SA 4.0", "artist": "Test", "page": ""})
    return got


def compose_still(src, dst):
    """Photo fitted over a blurred copy of itself, 1080x1920."""
    from PIL import Image, ImageFilter, ImageEnhance
    im = Image.open(src).convert("RGB")
    bg = im.copy()
    s = max(W / bg.width, H / bg.height)
    bg = bg.resize((math.ceil(bg.width * s), math.ceil(bg.height * s)))
    bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
    bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(40))).enhance(0.55)
    ratio = im.width / im.height
    if ratio < 0.75:                       # tall photo: fill the frame
        fg = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)))
        fg = fg.crop(((fg.width - W) // 2, (fg.height - H) // 2, (fg.width - W) // 2 + W, (fg.height - H) // 2 + H))
        bg = fg
    else:                                  # wide photo: fit width, centred
        fs = min(W / im.width, 1250 / im.height)
        fg = im.resize((int(im.width * fs), int(im.height * fs)))
        bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2 - 60))
    bg.save(dst, quality=92)


def kenburns(still, seconds, idx, dst):
    n = max(1, int(round(seconds * FPS)))
    zin = idx % 2 == 0
    z = f"min(1.0+0.10*on/{n},1.10)" if zin else f"max(1.10-0.10*on/{n},1.0)"
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", still, "-vf",
         f"scale={W * 2}:{H * 2},zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS},format=yuv420p",
         "-frames:v", n, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", dst])


# ---------------------------------------------------------------- timeline
def caption_chunks(text, start, end, max_words=3):
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    if len(chunks) > 1 and len(chunks[-1].split()) == 1:
        last = chunks.pop()
        chunks[-1] += " " + last
    total = sum(len(c) for c in chunks)
    out, t = [], start
    for c in chunks:
        d = (end - start) * len(c) / total
        out.append({"text": c, "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--speaker", default=os.getenv("SPEAKER", "Suresh"))
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    topic = json.loads((ROOT / "topics" / f"{a.topic}.json").read_text(encoding="utf-8"))
    shutil.rmtree(WORK, ignore_errors=True); WORK.mkdir(parents=True); OUT.mkdir(exist_ok=True)
    phrases, shots = topic["phrases"], topic["shots"]

    # 1. voice
    own = next((ROOT / "voice" / f"{a.topic}{ext}" for ext in (".wav", ".m4a", ".mp3")
                if (ROOT / "voice" / f"{a.topic}{ext}").exists()), None)
    if own:
        log("using your own recording:", own.name)
        files = None
    else:
        files = tts_test(phrases, WORK) if a.test else tts_parler(phrases, a.speaker, WORK)

    # 2. timeline + voice track
    t, segs, parts = LEAD, [], []
    if files:
        for ph, f in zip(phrases, files):
            d = dur(f)
            segs.append((t, t + d))
            parts.append((f, t))
            t += d + GAP + float(ph.get("pause_after", 0))
        voice_end = segs[-1][1]
        inputs, filt = [], []
        for i, (f, st) in enumerate(parts):
            inputs += ["-i", f]
            filt.append(f"[{i}:a]adelay={int(st * 1000)}|{int(st * 1000)}[v{i}]")
        filt.append("".join(f"[v{i}]" for i in range(len(parts))) +
                    f"amix=inputs={len(parts)}:normalize=0,apad=whole_dur={voice_end + 0.2:.2f},"
                    "loudnorm=I=-15:TP=-1.5:LRA=11[out]")
        run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filt), "-map", "[out]",
             "-ar", "44100", "-ac", "2", OUT / "voice.wav"])
    else:
        # own recording: spread phrases by text length over its duration
        run(["ffmpeg", "-y", "-v", "error", "-i", own, "-af",
             "silenceremove=start_periods=1:start_threshold=-45dB,loudnorm=I=-15:TP=-1.5:LRA=11",
             "-ar", "44100", "-ac", "2", WORK / "own.wav"])
        total = dur(WORK / "own.wav")
        run(["ffmpeg", "-y", "-v", "error", "-i", WORK / "own.wav", "-af", f"adelay={int(LEAD * 1000)}:all=1",
             OUT / "voice.wav"])
        lens = [len(p["text"]) for p in phrases]
        for L in lens:
            d = total * L / sum(lens)
            segs.append((t, t + d)); t += d
        voice_end = segs[-1][1]
    end_start = voice_end + 0.5
    total_dur = end_start + END_CARD

    # 3. photos and shots
    photos = test_photos(shots, WORK) if a.test else fetch_photos(shots, WORK)
    shot_start = {}
    for i, ph in enumerate(phrases):
        shot_start.setdefault(ph.get("shot", 0), segs[i][0] if i else 0.0)
    order = sorted(shot_start.items(), key=lambda kv: kv[1])
    clips, credits = [], []
    for j, (k, st) in enumerate(order):
        en = order[j + 1][1] if j + 1 < len(order) else end_start
        p = photos[k] if k < len(photos) else None
        if p is None:
            p = next((x for x in photos if x), None)
        still = WORK / f"still_{j:02d}.jpg"
        compose_still(p["path"], still)
        clip = WORK / f"clip_{j:02d}.mp4"
        kenburns(still, en - st, j, clip)
        clips.append(clip)
        credits.append({"text": f"Photo: {p['artist']} · {p['licence']} · Wikimedia Commons" if not a.test else "Photo: test",
                        "start": round(st, 3), "end": round(en, 3)})
    # end card background (black) is drawn by the overlay
    lst = WORK / "clips.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", WORK / "photos.mp4"])

    # 4. overlay timeline
    cards, caps = [], []
    for i, ph in enumerate(phrases):
        st, en = segs[i]
        if ph.get("card"):
            cards.append({"text": ph["card"], "start": round(st, 3), "end": round(min(en + 0.6, end_start), 3)})
        caps += caption_chunks(ph["text"], st, en)
    data = {"endStart": round(end_start, 3), "cards": cards, "caps": caps, "credits": credits}
    frames = WORK / "frames"; frames.mkdir()
    render_overlay(data, total_dur, frames)

    # 5. final mix
    music = ROOT / "assets" / "music.m4a"
    fo = max(0, total_dur - 1.0)
    run(["ffmpeg", "-y", "-v", "error",
         "-stream_loop", "-1", "-i", WORK / "photos.mp4",
         "-framerate", FPS, "-i", frames / "%05d.png",
         "-i", OUT / "voice.wav",
         "-stream_loop", "-1", "-i", music,
         "-filter_complex",
         f"[0:v]tpad=stop_mode=clone:stop_duration=5[bg];[bg][1:v]overlay=0:0:format=auto,format=yuv420p,"
         f"fade=t=out:st={fo:.2f}:d=1[v];"
         f"[3:a]volume=0.16,afade=t=in:d=1,afade=t=out:st={fo:.2f}:d=1[m];"
         f"[2:a]apad[vo];[vo][m]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.9[a]",
         "-map", "[v]", "-map", "[a]", "-t", f"{total_dur:.3f}", "-r", FPS,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", OUT / "video.mp4"])

    (OUT / "caption.txt").write_text(topic.get("caption", ""), encoding="utf-8")
    (OUT / "credits.txt").write_text("\n".join(
        f"{p['title']} — {p['artist']} — {p['licence']} — {p['page']}" for p in photos if p), encoding="utf-8")
    log(f"done: out/video.mp4 ({total_dur:.1f}s), voice: {'own recording' if own else a.speaker}")


def render_overlay(data, total, frames):
    from playwright.sync_api import sync_playwright
    n = int(math.ceil(total * FPS))
    prev, prevf, uniq, t0 = None, None, 0, time.time()
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto((ROOT / "scripts" / "overlay.html").as_uri())
        pg.evaluate("d => setup(d)", data)
        pg.evaluate("document.fonts.ready"); pg.wait_for_timeout(300)
        stage = pg.locator("#stage")
        for f in range(n):
            key = pg.evaluate("t => render(t)", f / FPS)
            out = frames / f"{f:05d}.png"
            if key == prev:
                os.link(prevf, out)
            else:
                stage.screenshot(path=str(out), omit_background=True); prev, prevf = key, out; uniq += 1
        b.close()
    log(f"overlay: {uniq} unique of {n} frames in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
