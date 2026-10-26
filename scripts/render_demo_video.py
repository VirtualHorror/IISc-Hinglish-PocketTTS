#!/usr/bin/env python3
"""Render demo_hinglish_pocket_tts.mp4: the released model reading six showcase sentences.

Two Roman Hinglish, two Devanagari and two mixed-script sentences (demo/presets.json), alternating the
female and male studio voices, synthesised through release/inference.py with the default phonetic
front-end. Each segment shows the typed text (coloured by script), what the front-end hands the model,
the tokenizer pieces, an animated waveform and mel-spectrogram, and the time to first audio and RTF
measured while rendering: on the GPU (whose audio is the soundtrack) and on the CPU with dynamic int8
weights (timing only). With --asr, Whisper large-v3-turbo transcribes every take and its CER is shown.

The soundtrack is the model's real 24 kHz output, muxed as AAC at 24 kHz; video is H.264 1920x1080.

    python scripts/render_demo_video.py --model-dir $WORK/release --asr \
        --out demo_hinglish_pocket_tts.mp4 --telemetry eval/results/demo_video.json

Needs Pillow with raqm (Devanagari shaping), Noto Sans + Noto Sans Devanagari, and ffmpeg (on PATH or
from the imageio-ffmpeg package).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "release"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, features  # noqa: E402

from demo_app import analyse, cpu_name  # noqa: E402
from inference import HinglishTTS  # noqa: E402

SR = 24000
W, H = 1920, 1080
SHOWCASE = [  # (preset id, voice): 2 Roman, 2 Devanagari, 2 mixed; female and male in turn
    ("roman_flight", "indian_female_ananya"),
    ("roman_weather", "indian_male_aarav"),
    ("deva_flight", "indian_female_ananya"),
    ("deva_traffic", "indian_male_aarav"),
    ("mixed_meeting", "indian_female_ananya"),
    ("mixed_model", "indian_male_aarav"),
]
LEAD_S, TAIL_S, TITLE_S, OUTRO_S = 1.0, 1.0, 4.5, 6.0

# the studio's palette: haldi = Devanagari, peacock = Latin
INK = (14, 19, 34)
INK_SUNK = (10, 14, 25)
RULE = (42, 51, 80)
RULE_SOFT = (31, 39, 64)
PAPER = (233, 228, 216)
PAPER_DIM = (185, 180, 168)
MUTED = (138, 147, 173)
HALDI = (233, 180, 76)
PEACOCK = (56, 178, 196)
KUMKUM = (224, 96, 94)
LEAF = (124, 198, 141)
WAVE_DIM = (58, 68, 102)
RAMP = [(10, 14, 25), (17, 52, 66), (56, 178, 196), (233, 180, 76), (252, 244, 224)]
KIND_COLOUR = {"deva": HALDI, "latin": PEACOCK, "digit": PAPER, "punct": PAPER_DIM, "space": PAPER_DIM, "byte": KUMKUM}
CATEGORY = {"roman": ("Roman Hinglish", PEACOCK), "devanagari": ("Devanagari Hinglish", HALDI),
            "mixed": ("Mixed script", None)}

MARGIN = 96
LEFT_W = 1180  # text column; the telemetry column starts at MARGIN + LEFT_W + 72
HUD_X = MARGIN + LEFT_W + 72
PLOT_X, PLOT_W = MARGIN, W - 2 * MARGIN
WAVE_Y, WAVE_H = 692, 120
MEL_Y, MEL_H = 828, 176


# ----- fonts and text ------------------------------------------------------------------------------------
class Fonts:
    def __init__(self, font_dir: Path):
        if not features.check("raqm"):
            raise SystemExit("Pillow without raqm cannot shape Devanagari: install a Pillow wheel with raqm")
        self.dir = font_dir
        self._cache: dict[tuple[str, int, bool], ImageFont.FreeTypeFont] = {}

    def get(self, script: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (script, size, bold)
        if key not in self._cache:
            name = ("NotoSansDevanagari" if script == "deva" else "NotoSans") + ("-Bold" if bold else "-Regular") + ".ttf"
            self._cache[key] = ImageFont.truetype(str(self.dir / name), size, layout_engine=ImageFont.Layout.RAQM)
        return self._cache[key]


def word_script(word: str) -> str:
    if any("ऀ" <= ch <= "ॿ" for ch in word):
        return "deva"
    return "latin" if any(ch.isascii() and ch.isalpha() for ch in word) else "other"


def draw_words(draw: ImageDraw.ImageDraw, fonts: Fonts, text: str, x: int, y: int, max_w: int, size: int,
               colour_by_script: bool = True, colour=PAPER, bold: bool = False, line_h: float = 1.42,
               max_lines: int = 4) -> int:
    """Word-wrapped text, each word in its script's font (and colour); y is the first baseline.
    Returns the baseline after the last line."""
    lines: list[list[tuple[str, ImageFont.FreeTypeFont, tuple]]] = [[]]
    width = 0.0
    space = fonts.get("latin", size, bold).getlength(" ")
    for word in text.split():
        script = word_script(word)
        font = fonts.get("deva" if script == "deva" else "latin", size, bold)
        fill = (HALDI if script == "deva" else PEACOCK if script == "latin" else PAPER) if colour_by_script else colour
        w = font.getlength(word)
        if lines[-1] and width + space + w > max_w:
            lines.append([])
            width = 0.0
        lines[-1].append((word, font, fill))
        width += (space if width else 0) + w
    if len(lines) > max_lines:  # mark the cut
        lines = lines[:max_lines]
        lines[-1].append(("…", fonts.get("latin", size, bold), lines[-1][-1][2] if lines[-1] else colour))
    for i, line in enumerate(lines):
        cx = float(x)
        for word, font, fill in line:
            draw.text((cx, y + i * size * line_h), word, font=font, fill=fill, anchor="ls")
            cx += font.getlength(word) + space
    return int(y + (len(lines) - 1) * size * line_h)


def draw_pieces(draw: ImageDraw.ImageDraw, fonts: Fonts, chunks: list[dict], x: int, y: int, max_w: int,
                size: int = 30, max_lines: int = 3) -> int:
    """Tokenizer pieces hanging from bars in their kind's colour, ids underneath (as in the studio)."""
    id_font = fonts.get("latin", 15)
    cx, line = x, 0
    row_h = int(size * 2.45)
    for chunk in chunks:
        for p in chunk["pieces"]:
            start = p["p"].startswith("▁")
            body = p["p"].replace("▁", "")
            font = fonts.get("deva" if p["k"] == "deva" else "latin", size)
            w = max(18, int(max(font.getlength(body), id_font.getlength(str(p["id"])))) + 8)
            gap = 12 if start and cx > x else 3
            if cx + gap + w > x + max_w:
                line += 1
                if line >= max_lines:
                    return y + line * row_h
                cx, gap = x, 0
            top = y + line * row_h
            colour = KIND_COLOUR[p["k"]]
            if start and gap > 3:  # word boundary: a short dim bar joins the previous piece
                draw.rectangle([cx, top, cx + gap - 1, top + 2], fill=RULE)
            bx = cx + gap
            draw.rectangle([bx, top, bx + w - 1, top + 2], fill=colour)
            if body:
                draw.text((bx + w / 2, top + 6 + size * 0.95), body, font=font, fill=colour, anchor="ms")
            else:  # a lone word-boundary piece
                draw.rectangle([bx + 4, top + 6 + size * 0.75, bx + w - 5, top + 6 + size * 0.75 + 2], fill=MUTED)
            draw.text((bx + w / 2, top + 6 + size * 0.95 + 20), str(p["id"]), font=id_font, fill=MUTED, anchor="ms")
            cx = bx + w
    return y + (line + 1) * row_h


# ----- signal images ----------------------------------------------------------------------------------
def mel_image(audio: np.ndarray, width: int, height: int) -> np.ndarray:
    """80-band log-mel (n_fft 1024, hop 10 ms, 0-12 kHz, HTK mel, 80 dB range) in the studio's colormap."""
    n_fft, hop, n_mels, fmax = 1024, 240, 80, 12000
    spec = torch.stft(torch.from_numpy(audio.astype(np.float32)), n_fft, hop, window=torch.hann_window(n_fft),
                      return_complex=True).abs().pow(2).numpy()
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = mel2hz(np.linspace(0, hz2mel(fmax), n_mels + 2)) * n_fft / SR
    k = np.arange(n_fft // 2 + 1)[None, :]
    lo, mid, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    bank = np.maximum(0, np.minimum((k - lo) / (mid - lo), (hi - k) / (hi - mid)))
    db = 10 * np.log10(np.maximum(bank @ spec, 1e-10))
    v = np.clip((db - (db.max() - 80)) / 80, 0, 1)[::-1]  # low frequencies at the bottom
    ramp = np.array(RAMP, dtype=np.float32)
    pos = v * (len(RAMP) - 1)
    j = np.minimum(len(RAMP) - 2, pos.astype(int))
    t = (pos - j)[..., None]
    rgb = (ramp[j] * (1 - t) + ramp[j + 1] * t).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR))


def wave_images(audio: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """(played, upcoming) waveform renderings: per-column min/max envelopes."""
    edges = np.linspace(0, len(audio), width + 1).astype(int)
    lo = np.array([audio[a:b].min() if b > a else 0 for a, b in zip(edges[:-1], edges[1:], strict=True)])
    hi = np.array([audio[a:b].max() if b > a else 0 for a, b in zip(edges[:-1], edges[1:], strict=True)])
    peak = max(1e-3, float(np.abs(audio).max()))
    y = np.arange(height)[:, None]
    top = (height / 2 - hi / peak * height / 2 * 0.92)[None, :]
    bot = (height / 2 - lo / peak * height / 2 * 0.92)[None, :]
    mask = (y >= np.floor(top)) & (y <= np.ceil(bot))
    mask[height // 2, :] = True
    out = []
    for colour in (PAPER, WAVE_DIM):
        img = np.empty((height, width, 3), dtype=np.uint8)
        img[:] = INK_SUNK
        img[mask] = colour
        out.append(img)
    return out[0], out[1]


# ----- synthesis --------------------------------------------------------------------------------------
def synthesize_all(model_dir: Path, items: list[dict], device: str) -> None:
    tts = HinglishTTS.from_pretrained(str(model_dir), device=device)
    for v in {it["voice"] for it in items}:
        tts.voice_state(v)
    for it in items[:2]:  # warm-up
        tts.synthesize(it["text"], voice=it["voice"])
    for it in items:
        torch.manual_seed(zlib.crc32(f"video|{it['id']}|{it['voice']}".encode()) & 0x7FFFFFFF)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunks, ttfa = [], None
        for c in tts.stream(it["text"], voice=it["voice"]):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(c)
        total = time.perf_counter() - t0
        it["audio"] = np.concatenate(chunks)
        audio_s = len(it["audio"]) / SR
        it["gpu"] = {"ttfa_ms": round(ttfa * 1000, 1), "rtf": round(total / audio_s, 4), "synth_s": round(total, 3)}
        it["audio_s"] = round(audio_s, 3)
        info = analyse(SimpleNamespace(tts=tts, model=tts.model), it["text"], "phonetic")
        it["model_input"], it["chunks"], it["n_tokens"] = info["model_input"], info["chunks"], info["n_tokens"]
        print(f"{it['id']}: {audio_s:.2f} s, TTFA {it['gpu']['ttfa_ms']} ms, RTF {it['gpu']['rtf']}", flush=True)
    del tts
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def time_cpu(model_dir: Path, items: list[dict], threads: int) -> None:
    """Time each sentence on the CPU (dynamic int8): a separate generation, timing only."""
    torch.set_num_threads(threads)
    tts = HinglishTTS.from_pretrained(str(model_dir), device="cpu", quantize=True)
    for v in {it["voice"] for it in items}:
        tts.voice_state(v)
    for it in items[:2]:
        tts.synthesize(it["text"], voice=it["voice"])
    for it in items:
        torch.manual_seed(zlib.crc32(f"video|{it['id']}|{it['voice']}".encode()) & 0x7FFFFFFF)
        t0 = time.perf_counter()
        n, ttfa = 0, None
        for c in tts.stream(it["text"], voice=it["voice"]):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            n += len(c)
        total = time.perf_counter() - t0
        it["cpu"] = {"ttfa_ms": round(ttfa * 1000, 1), "rtf": round(total / (n / SR), 4), "threads": threads,
                     "precision": "int8"}
        print(f"{it['id']}: CPU TTFA {it['cpu']['ttfa_ms']} ms, RTF {it['cpu']['rtf']}", flush=True)


# ----- frames -----------------------------------------------------------------------------------------
def base_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), INK)
    draw = ImageDraw.Draw(img)
    draw.rectangle([MARGIN, 52, W - MARGIN, 54], fill=PAPER)  # the headline bar everything hangs from
    return img, draw


def brand(draw: ImageDraw.ImageDraw, fonts: Fonts, size: int, x: int, y: int) -> None:
    f_lat, f_dev = fonts.get("latin", size, True), fonts.get("deva", size, True)
    draw.text((x, y), "Pocket TTS", font=f_lat, fill=PEACOCK, anchor="ls")
    draw.text((x + f_lat.getlength("Pocket TTS "), y), "हिंग्लिश", font=f_dev, fill=HALDI, anchor="ls")


def title_frame(fonts: Fonts, meta: dict) -> np.ndarray:
    img, draw = base_canvas()
    brand(draw, fonts, 132, MARGIN, 330)
    draw.text((MARGIN, 420), "Indian Hinglish text-to-speech from Roman, Devanagari and mixed-script text",
              font=fonts.get("latin", 42), fill=PAPER, anchor="ls")
    lines = [
        "Kyutai Pocket TTS (24-layer FlowLM + Mimi codec, 24 kHz), fine-tuned on 10.2 h of human-recorded",
        "Hindi, English and code-switched speech, every utterance in all three scripts.",
        f"Benchmark: {meta['cer']} CER with the phonetic front-end (Whisper round trip; real recordings: {meta['floor']}).",
        "Six sentences follow, read by two studio voices. The audio is the model's own output;",
        "the latency figures were measured while this video was rendered.",
    ]
    for i, line in enumerate(lines):
        draw.text((MARGIN, 540 + i * 52), line, font=fonts.get("latin", 31), fill=PAPER_DIM, anchor="ls")
    draw.text((MARGIN, H - 72), "Model: MSRIT-NBM/pocket-tts-indian-hinglish     Data: MSRIT-NBM/iisc-hinglish-pocket-tts-dataset",
              font=fonts.get("latin", 26), fill=MUTED, anchor="ls")
    return np.asarray(img)


def outro_frame(fonts: Fonts, meta: dict, items: list[dict]) -> np.ndarray:
    img, draw = base_canvas()
    brand(draw, fonts, 72, MARGIN, 170)
    draw.text((MARGIN, 260), "Benchmark: 24 held-out sentences × 4 voices, Whisper large-v3-turbo round trip",
              font=fonts.get("latin", 34), fill=PAPER, anchor="ls")
    rows = [("Roman Hinglish", meta["roman"], PEACOCK), ("Devanagari", meta["devanagari"], HALDI),
            ("Mixed script", meta["mixed"], PAPER), ("All sentences", meta["cer"], PAPER)]
    for i, (name, cer, colour) in enumerate(rows):
        y = 350 + i * 64
        draw.text((MARGIN, y), name, font=fonts.get("latin", 36), fill=colour, anchor="ls")
        draw.text((MARGIN + 470, y), f"{cer} CER", font=fonts.get("latin", 36, True), fill=PAPER, anchor="ls")
    draw.text((MARGIN, 640), f"Real recordings through the same ASR: {meta['floor']} CER (the measurement floor).",
              font=fonts.get("latin", 30), fill=PAPER_DIM, anchor="ls")
    gpu = [it["gpu"] for it in items]
    cpu = [it.get("cpu") for it in items if it.get("cpu")]
    audio = sum(it["audio_s"] for it in items)
    speed = [f"In this video: {meta['gpu_label']} RTF {sum(g['synth_s'] for g in gpu) / audio:.3f}, "
             f"median first audio {np.median([g['ttfa_ms'] for g in gpu]):.0f} ms"]
    if cpu:
        speed.append(f"{meta['cpu_label']}, int8, {cpu[0]['threads']} threads: RTF {np.mean([c['rtf'] for c in cpu]):.3f}, "
                     f"median first audio {np.median([c['ttfa_ms'] for c in cpu]):.0f} ms")
    student = sorted((REPO_ROOT / "eval" / "results").glob("evaluation_results_student_*l.json"))
    if student:
        sb = json.loads(student[0].read_text(encoding="utf-8"))
        n = student[0].stem.split("_")[-1].removesuffix("l")
        s_cpu = sb.get("speed", {}).get("cpu_int8_4threads")
        speed.append(f"{n}-layer CPU student: {100 * sb['quality']['default']['overall']['cer']:.1f} % CER"
                     + (f", int8 4 threads RTF {s_cpu['rtf_total']:.3f}" if s_cpu else ""))
    for i, line in enumerate(speed):
        draw.text((MARGIN, 740 + i * 52), line, font=fonts.get("latin", 31), fill=PAPER, anchor="ls")
    draw.text((MARGIN, H - 72), "Try it: python demo_app.py (interactive studio), or python inference.py --text \"...\" "
              "with the model repo", font=fonts.get("latin", 26), fill=MUTED, anchor="ls")
    return np.asarray(img)


def segment_background(fonts: Fonts, it: dict, k: int, n: int) -> Image.Image:
    img, draw = base_canvas()
    brand(draw, fonts, 30, MARGIN, 100)
    cat_label, cat_colour = CATEGORY[it["category"]]
    f = fonts.get("latin", 30)
    right = f"{k + 1} of {n}"
    draw.text((W - MARGIN, 100), right, font=f, fill=MUTED, anchor="rs")
    cx = W - MARGIN - f.getlength(right) - 40 - f.getlength(cat_label)
    draw.text((cx, 100), cat_label, font=fonts.get("latin", 30, True), fill=PAPER, anchor="ls")
    sw = (cx - 26, 82, cx - 10, 98)
    if cat_colour is None:
        draw.rectangle([sw[0], sw[1], (sw[0] + sw[2]) // 2, sw[3]], fill=HALDI)
        draw.rectangle([(sw[0] + sw[2]) // 2, sw[1], sw[2], sw[3]], fill=PEACOCK)
    else:
        draw.rectangle(sw, fill=cat_colour)

    # left column: typed text, front-end output, pieces
    label = fonts.get("latin", 24)
    draw.text((MARGIN, 172), "Typed", font=label, fill=MUTED, anchor="ls")
    y = draw_words(draw, fonts, it["text"], MARGIN, 232, LEFT_W, 46, max_lines=2)
    y += 74
    draw.text((MARGIN, y), "What the model reads (phonetic Hinglish front-end)", font=label, fill=MUTED, anchor="ls")
    y = draw_words(draw, fonts, it["model_input"], MARGIN, y + 52, LEFT_W, 36, colour_by_script=True, max_lines=2)
    y += 58
    draw.text((MARGIN, y), f"{it['n_tokens']} pieces of the 4,000-piece tokenizer", font=label, fill=MUTED, anchor="ls")
    draw_pieces(draw, fonts, it["chunks"], MARGIN, y + 16, LEFT_W, size=27, max_lines=2)

    # right column: telemetry
    hx = HUD_X
    gpu, cpu = it["gpu"], it.get("cpu")
    voice = it["voice"].split("_")[-1].capitalize()
    draw.text((hx, 172), "Voice", font=label, fill=MUTED, anchor="ls")
    draw.text((hx, 214), f"{voice}, {'female' if '_female_' in it['voice'] else 'male'}",
              font=fonts.get("latin", 34, True), fill=PAPER, anchor="ls")
    draw.text((hx, 268), "Time to first audio", font=label, fill=MUTED, anchor="ls")
    big = fonts.get("latin", 66, True)
    draw.text((hx, 338), f"{gpu['ttfa_ms']:.0f}", font=big, fill=PAPER, anchor="ls")
    draw.text((hx + big.getlength(f"{gpu['ttfa_ms']:.0f}") + 10, 338), "ms", font=fonts.get("latin", 30), fill=PAPER_DIM, anchor="ls")
    draw.text((hx + 240, 338), it["gpu_label"], font=fonts.get("latin", 24), fill=PAPER_DIM, anchor="ls")
    if cpu:
        draw.text((hx, 382), f"{cpu['ttfa_ms']:.0f} ms on {it['cpu_label']}, int8, {cpu['threads']} threads",
                  font=fonts.get("latin", 24), fill=PAPER_DIM, anchor="ls")
    draw.text((hx, 434), "Real-time factor", font=label, fill=MUTED, anchor="ls")
    draw.text((hx, 504), f"{gpu['rtf']:.3f}", font=big, fill=PAPER, anchor="ls")
    draw.text((hx + 240, 504), f"{1 / gpu['rtf']:.1f}× real time", font=fonts.get("latin", 24), fill=PAPER_DIM, anchor="ls")
    if cpu:
        draw.text((hx, 548), f"{cpu['rtf']:.3f} on the CPU ({1 / cpu['rtf']:.1f}× real time)", font=fonts.get("latin", 24),
                  fill=PAPER_DIM, anchor="ls")
    if it.get("asr"):
        draw.text((hx, 600), f"Whisper heard (CER {100 * it['asr']['cer']:.1f} %)", font=label, fill=MUTED, anchor="ls")

    # signal area: frame, axis labels
    draw.text((PLOT_X, WAVE_Y - 14), f"Waveform and mel-spectrogram (80 bands, 0-12 kHz), {it['audio_s']:.2f} s at 24 kHz",
              font=fonts.get("latin", 21), fill=MUTED, anchor="ls")
    step = 1 if it["audio_s"] <= 7 else 2
    for t in range(0, int(it["audio_s"]) + 1, step):
        x = PLOT_X + t / it["audio_s"] * PLOT_W
        if x < PLOT_X + PLOT_W - 30:
            draw.text((x, MEL_Y + MEL_H + 30), f"{t} s", font=fonts.get("latin", 21), fill=MUTED,
                      anchor="ls" if t == 0 else "ms")
    draw.rectangle([PLOT_X, MEL_Y, PLOT_X + PLOT_W - 1, MEL_Y + MEL_H - 1], fill=INK_SUNK)
    return img


def draw_asr_text(img: Image.Image, fonts: Fonts, it: dict) -> None:
    if it.get("asr"):
        draw = ImageDraw.Draw(img)
        hx = HUD_X
        draw_words(draw, fonts, it["asr"]["hyp"], hx, 636, W - MARGIN - hx, 24, colour_by_script=False,
                   colour=PAPER_DIM, line_h=1.35, max_lines=2)


def fade(frame: np.ndarray, a: float) -> np.ndarray:
    if a >= 1:
        return frame
    return (frame.astype(np.float32) * a + np.array(INK, np.float32) * (1 - a)).astype(np.uint8)


def ffmpeg_exe(arg: str | None) -> str:
    if arg:
        return arg
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "demo_hinglish_pocket_tts.mp4")
    ap.add_argument("--telemetry", type=Path, default=REPO_ROOT / "eval" / "results" / "demo_video.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-threads", type=int, default=4, help="0 skips the CPU timing")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/noto"))
    ap.add_argument("--ffmpeg", default=None)
    cli = ap.parse_args()

    fonts = Fonts(cli.font_dir)
    presets = {p["id"]: (c["id"], p) for c in json.loads((REPO_ROOT / "demo" / "presets.json").read_text(encoding="utf-8"))["categories"]
               for p in c["presets"]}
    items = [{"id": pid, "category": presets[pid][0], "text": presets[pid][1]["text"], "voice": voice}
             for pid, voice in SHOWCASE]
    gpu_label = torch.cuda.get_device_name().replace("NVIDIA GeForce ", "") if cli.device.startswith("cuda") else "CPU"
    cpu_label = cpu_name().replace("13th Gen Intel(R) Core(TM) ", "").replace("Intel(R) ", "")

    synthesize_all(cli.model_dir, items, cli.device)
    if cli.asr:
        from eval.asr import HinglishScorer, WhisperASR

        asr, scorer = WhisperASR(device=cli.device), HinglishScorer()
        hyps = asr.transcribe([it["audio"] for it in items], sample_rates=[SR] * len(items), language="hi")
        scores = scorer.score([it["text"] for it in items], hyps)
        for it, hyp, s in zip(items, hyps, scores["per_utt"], strict=True):
            it["asr"] = {"hyp": hyp, "cer": round(s["cer"], 4), "wer": round(s["wer"], 4)}
        del asr
        torch.cuda.empty_cache()
    if cli.cpu_threads:
        time_cpu(cli.model_dir, items, cli.cpu_threads)
    for it in items:
        it["gpu_label"], it["cpu_label"] = gpu_label, cpu_label

    bench = json.loads((REPO_ROOT / "eval" / "evaluation_results.json").read_text(encoding="utf-8"))
    q = bench["quality"]["default"]
    summary = json.loads((REPO_ROOT / "training" / "training_summary.json").read_text(encoding="utf-8"))
    floor = summary["asr_floor_real_recordings"]["selection"]  # Whisper on the real held-out recordings
    pct = lambda x: f"{100 * x:.1f} %"  # noqa: E731
    meta = {"cer": pct(q["overall"]["cer"]), "roman": pct(q["input_script=roman"]["cer"]),
            "devanagari": pct(q["input_script=devanagari"]["cer"]), "mixed": pct(q["input_script=code_switched"]["cer"]),
            "floor": pct(floor["cer"]), "gpu_label": gpu_label, "cpu_label": cpu_label}

    # timeline
    fps = cli.fps
    segments, t = [], TITLE_S
    for it in items:
        segments.append((t, it))
        t += LEAD_S + it["audio_s"] + TAIL_S
    total_s = t + OUTRO_S
    n_frames = int(round(total_s * fps))
    track = np.zeros(int(round(n_frames / fps * SR)), dtype=np.float32)
    for start, it in segments:
        a = int(round((start + LEAD_S) * SR))
        track[a:a + len(it["audio"])] = it["audio"][: len(track) - a]

    title, outro = title_frame(fonts, meta), outro_frame(fonts, meta, items)
    prepared = []
    for k, (start, it) in enumerate(segments):
        bg = segment_background(fonts, it, k, len(items))
        draw_asr_text(bg, fonts, it)
        played, upcoming = wave_images(it["audio"], PLOT_W, WAVE_H)
        prepared.append((start, it, np.asarray(bg).copy(), played, upcoming, mel_image(it["audio"], PLOT_W, MEL_H)))

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "track.wav"
        sf.write(str(wav), track, SR, subtype="PCM_16")
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg_exe(cli.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-i", str(wav),
               "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-profile:v", "high",
               "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "1",
               "-movflags", "+faststart", "-shortest", str(cli.out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for f in range(n_frames):
            now = f / fps
            if now < TITLE_S:
                frame = fade(title, min(1.0, now / 0.6, (TITLE_S - now) / 0.35))
            elif now >= total_s - OUTRO_S:
                into = now - (total_s - OUTRO_S)
                frame = fade(outro, min(1.0, into / 0.35, (OUTRO_S - into) / 0.8))
            else:
                start, it, bg, played, upcoming, mel = next(p for p in reversed(prepared) if p[0] <= now)
                frame = bg.copy()
                progress = np.clip((now - start - LEAD_S) / it["audio_s"], 0.0, 1.0)
                px = int(round(progress * PLOT_W))
                frame[WAVE_Y:WAVE_Y + WAVE_H, PLOT_X:PLOT_X + px] = played[:, :px]
                frame[WAVE_Y:WAVE_Y + WAVE_H, PLOT_X + px:PLOT_X + PLOT_W] = upcoming[:, px:]
                frame[MEL_Y:MEL_Y + MEL_H, PLOT_X:PLOT_X + px] = mel[:, :px]
                if 0 < progress < 1:
                    x = PLOT_X + px
                    frame[WAVE_Y:WAVE_Y + WAVE_H, x:x + 3] = HALDI
                    frame[MEL_Y:MEL_Y + MEL_H, x:x + 3] = HALDI
                seg_end = start + LEAD_S + it["audio_s"] + TAIL_S
                frame = fade(frame, min(1.0, (now - start) / 0.3, (seg_end - now) / 0.25))
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg failed")

    record = {"video": cli.out.name, "size": f"{W}x{H}", "fps": fps, "duration_s": round(n_frames / fps, 2),
              "audio": {"sample_rate": SR, "codec": "aac", "bitrate": "192k"}, "rendered": time.strftime("%Y-%m-%d %H:%M"),
              "gpu": gpu_label, "cpu": cpu_label, "front_end": "phonetic (default)",
              "protocol": "one take per sentence, fixed seed; TTFA = stream call (text normalisation included) to the "
                          "first decoded chunk, voice state cached, after 2 warm-up sentences; CPU = separate "
                          "generation timed with dynamic int8 weights",
              "sentences": [{k: v for k, v in it.items() if k not in ("audio", "chunks", "gpu_label", "cpu_label")}
                            for it in items]}
    cli.telemetry.parent.mkdir(parents=True, exist_ok=True)
    cli.telemetry.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {cli.out} ({n_frames / fps:.1f} s) and {cli.telemetry}", flush=True)


if __name__ == "__main__":
    main()
