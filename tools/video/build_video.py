#!/usr/bin/env python3
"""Neurosonde video manual, in the plug-in's own look (after Ryoji Ikeda).

  python3 tools/video/build_video.py [--out ~/Desktop/Neurosonde-Video-Manual.mp4]
        [--only 0-3] [--still 6.0] [--overlay] [--print] [--no-bed] [--jobs N]

Black ground, a faint measuring grid (a dot every 20 px, a cross every 100 px),
white hairlines, small lowercase IBM Plex Mono Light and Inter, ruler ticks and
binary cells; no colour, glow or gradients. Every chapter opens on a title card
(index, lowercase title, a ruler), then the camera moves over REAL screenshots
of the editor while the narrator explains them: slow, precise pans and zooms,
hairline bracket callouts with lowercase labels on the controls being named,
the chapter's points on the right (the one being spoken lit, like the
plug-in's lists), a minimap with the view, and a data line at the bottom like
the plug-in's footer.

Inputs (all regenerated if missing):
  build/video/shots, build/video/shots-space   NeuroTest / NeuroSpaceTest --shots
  build/video/tts/*.wav                        Kokoro "Emma", cached by text hash
  build/video/bed.wav                          sound made by Neurosonde (render_bed.py)

The script is chapters.py. Needs Pillow, numpy, ffmpeg (system python3); the
TTS runs in the Cirrostratus Kokoro venv (NS_TTS_PY to override), the bed in
build/video/.venv-audio (pedalboard).
"""
import argparse, hashlib, json, math, os, subprocess, sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True         # no __pycache__ in the repo
from chapters import CHAPTERS  # noqa: E402

WORK = os.path.join(ROOT, "build", "video")
SHOTS = os.path.join(WORK, "shots")
SHOTS_SPACE = os.path.join(WORK, "shots-space")
TTS_DIR = os.path.join(WORK, "tts")
FONTS = os.path.join(ROOT, "Resources", "fonts")
TTS_PY = os.environ.get("NS_TTS_PY", os.path.expanduser("~/cirrostratus/cirrostratus-monsoon/tools/video/.venv/bin/python"))
AUDIO_PY = os.path.join(WORK, ".venv-audio", "bin", "python")
VOICE, SPEED = "bf_emma", 1.0

W, H, FPS, SR = 1920, 1080, 30, 48000
SW, SH = 1280, 800                     # screenshot size
VX, VY, VW, VH = 40, 112, 1400, 875    # the viewport (aspect 1.6, like the editor)
RX, RW = 1480, 400                     # the right column
CARD, GAP, TAIL, MOVE = 3.2, 0.55, 1.4, 1.1
MAXZOOM = 2.4                          # screenshot px -> at most 2.4 x the full-view scale

# the plug-in's greys (Source/ui/Theme.h)
BG = (0, 0, 0)
EDGE = (0x26,) * 3
TEXT = (0xee,) * 3
TEXT2 = (0x99,) * 3
MUTED = (0x59,) * 3
SELECTED = (0x1a,) * 3
WHITE = (255, 255, 255)


def grey(a):
    v = int(round(255 * max(0.0, min(1.0, a))))
    return (v, v, v)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


F_MONO = font("IBMPlexMono-Light.ttf", 17)
F_MONO_S = font("IBMPlexMono-Light.ttf", 15)
F_MONO_M = font("IBMPlexMono-Medium.ttf", 17)
F_MONO_L = font("IBMPlexMono-Light.ttf", 26)
F_HEAD = font("Inter-Medium.ttf", 22)
F_SANS = font("Inter-Regular.ttf", 30)
F_TITLE = font("Inter-Medium.ttf", 84)
F_CTITLE = font("Inter-Medium.ttf", 40)
F_IDX = font("IBMPlexMono-Light.ttf", 64)


# ---------------------------------------------------------------------------
# The drawing vocabulary of Theme.h, at video scale.

def grid(d, x0=0, y0=0, x1=W, y1=H):
    """A dot every 20 px, a cross every 100 px."""
    for y in range(0, H + 1, 20):
        if y < y0 or y > y1:
            continue
        for x in range(0, W + 1, 20):
            if x < x0 or x > x1:
                continue
            if x % 100 == 0 and y % 100 == 0:
                d.line([(x - 4, y), (x + 4, y)], fill=(46, 46, 46))
                d.line([(x, y - 4), (x, y + 4)], fill=(46, 46, 46))
            else:
                d.point((x, y), fill=(30, 30, 30))


def ruler(d, x, y, w, h, step=4, a=0.22):
    """Fine ticks, longer every 5 / 10 (Ikeda's gauge strip); y is the bottom."""
    k = 0
    xx = x
    while xx <= x + w:
        th = h if k % 10 == 0 else (h * 0.6 if k % 5 == 0 else h * 0.3)
        d.line([(xx, y - th), (xx, y)], fill=grey(a))
        xx += step
        k += 1


def corners(d, r, a=0.7, n=5):
    x0, y0, x1, y1 = r
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        d.line([(cx - n, cy), (cx + n, cy)], fill=grey(a))
        d.line([(cx, cy - n), (cx, cy + n)], fill=grey(a))


def window(d, r, title, index=None, right=None):
    """Hairline frame, corner crosses, index + lowercase title, a ruler under the title."""
    x0, y0, x1, y1 = r
    d.rectangle(r, outline=EDGE)
    corners(d, r)
    t = ("%02x  " % index if index is not None else "") + title.lower()
    d.text((x0 + 12, y0 + 9), t, font=F_MONO_S, fill=TEXT2)
    if right:
        d.text((x1 - 12, y0 + 9), right, font=F_MONO_S, fill=MUTED, anchor="ra")
    ruler(d, x0 + 12, y0 + 36, x1 - x0 - 24, 6, 5, 0.16)
    return (x0 + 12, y0 + 44, x1 - 12, y1 - 8)


def brackets(d, r, ln, col, t=1):
    x0, y0, x1, y1 = r
    for (ax, ay, bx, by) in ((x0, y0, x0 + ln, y0), (x0, y0, x0, y0 + ln), (x1, y0, x1 - ln, y0), (x1, y0, x1, y0 + ln),
                             (x0, y1, x0 + ln, y1), (x0, y1, x0, y1 - ln), (x1, y1, x1 - ln, y1), (x1, y1, x1, y1 - ln)):
        d.line([(ax, ay), (bx, by)], fill=col, width=t)


def bits(d, x, y, value, n, cw=7, ch=12, gap=3):
    """Binary cells, most significant first: white = 1, faint = 0."""
    for b in range(n):
        on = (value >> (n - 1 - b)) & 1
        d.rectangle([x + b * (cw + gap), y, x + b * (cw + gap) + cw - 1, y + ch - 1], fill=WHITE if on else grey(0.10))


def wrap(text, fnt, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if fnt.getlength(t) <= width:
            line = t
        else:
            out.append(line)
            line = w
    if line:
        out.append(line)
    return out


def tc(t):
    return "%02d:%02d:%05.2f" % (t // 3600, (t // 60) % 60, t % 60)


# ---------------------------------------------------------------------------
# Screenshots and the camera.

def shot_path(name):
    if name.startswith("space/"):
        return os.path.join(SHOTS_SPACE, name.split("/", 1)[1] + ".png")
    return os.path.join(SHOTS, name + ".png")


_shots = {}


def shot(name):
    if name not in _shots:
        _shots[name] = Image.open(shot_path(name)).convert("RGB")
    return _shots[name]


def view_for(focus):
    """The screenshot rectangle (x, y, w, h) shown for a focus: aspect 1.6, a margin, clamped, zoom limited."""
    if focus is None:
        return (0.0, 0.0, float(SW), float(SH))
    x, y, w, h = focus
    cx, cy = x + w / 2, y + h / 2
    w, h = w * 1.10 + 16, h * 1.10 + 16
    if w / h < 1.6:
        w = h * 1.6
    else:
        h = w / 1.6
    minw = SW / MAXZOOM
    if w < minw:
        w, h = minw, minw / 1.6
    if w > SW:
        w, h = SW, SH
    x = min(max(cx - w / 2, 0), SW - w)
    y = min(max(cy - h / 2, 0), SH - h)
    return (x, y, w, h)


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def lerp_view(a, b, u):
    # interpolate centre linearly and size geometrically (an even zoom)
    ca = (a[0] + a[2] / 2, a[1] + a[3] / 2)
    cb = (b[0] + b[2] / 2, b[1] + b[3] / 2)
    w = a[2] * (b[2] / a[2]) ** u
    cx, cy = ca[0] + (cb[0] - ca[0]) * u, ca[1] + (cb[1] - ca[1]) * u
    return (cx - w / 2, cy - w / 1.6 / 2, w, w / 1.6)


def push(v, u, amount=0.035):
    """The slow push-in while a segment is held: a few percent, toward its centre."""
    s = 1 - amount * u
    cx, cy = v[0] + v[2] / 2, v[1] + v[3] / 2
    w = v[2] * s
    hh = w / 1.6
    x = min(max(cx - w / 2, 0), SW - w)
    y = min(max(cy - hh / 2, 0), SH - hh)
    return (x, y, w, hh)


# ---------------------------------------------------------------------------
# Layout of one chapter in time.

def chapter_plan(c, seg_durs):
    """Segment start times within the chapter (the card comes first)."""
    t = CARD
    starts = []
    for d in seg_durs:
        starts.append(t)
        t += d + GAP
    return starts, t - GAP + TAIL


def camera(c, starts, t):
    """(shot name, view rect, segment index, seconds into segment, move progress)."""
    segs = c["segs"]
    k = 0
    for i, s in enumerate(starts):
        if t >= s:
            k = i
    s = segs[k]
    target = view_for(s.get("focus"))
    into = t - starts[k]
    end = starts[k + 1] if k + 1 < len(starts) else starts[k] + 60
    hold = max(0.1, end - starts[k] - MOVE)
    if k > 0 and segs[k - 1]["shot"] == s["shot"]:
        prev = push(view_for(segs[k - 1].get("focus")), 1.0)
        u = ease(into / MOVE)
        v = lerp_view(prev, target, u) if u < 1 else target
        v = push(v, max(0.0, (into - MOVE) / hold))
        return s["shot"], v, k, into, u
    # a cut to a new screenshot (or the first segment): start a little wide, settle in
    wide = (target[0] - target[2] * 0.04, target[1] - target[3] * 0.04, target[2] * 1.08, target[3] * 1.08)
    wide = (max(0.0, min(wide[0], SW - min(wide[2], SW))), max(0.0, min(wide[1], SH - min(wide[3], SH))),
            min(wide[2], SW), min(wide[3], SH))
    u = ease(into / (MOVE * 1.4))
    v = lerp_view(wide, target, u) if u < 1 else target
    v = push(v, max(0.0, (into - MOVE * 1.4) / hold))
    return s["shot"], v, k, into, u


# ---------------------------------------------------------------------------
# Static layers.

def base_frame(c, chapters, ci):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    grid(d)
    # header (like the editor's): name left, chapter centred as the selected tab, status right
    d.rectangle([0, 0, W, 62], fill=BG)
    d.text((40, 20), "neurosonde", font=F_HEAD, fill=TEXT)
    d.text((40 + F_HEAD.getlength("neurosonde") + 18, 24), "video manual", font=F_MONO_S, fill=MUTED)
    tab = "%02x  %s" % (ci, c["title"])
    tw = F_MONO.getlength(tab) + 36
    d.rectangle([W / 2 - tw / 2, 18, W / 2 + tw / 2, 44], fill=TEXT)
    d.text((W / 2, 31), tab, font=F_MONO, fill=BG, anchor="mm")
    d.line([(0, 62), (W, 62)], fill=grey(0.14))
    # footer rule
    d.rectangle([0, H - 34, W, H], fill=BG)
    d.line([(0, H - 34), (W, H - 34)], fill=grey(0.14))
    # right column: the chapter window
    inner = window(d, (RX, VY - 30, RX + RW, VY + 500), "chapter", ci, "%02x / %02x" % (ci, len(chapters) - 1))
    x, y = inner[0], inner[1] + 6
    d.text((x, y), "%02x" % ci, font=F_IDX, fill=TEXT)
    bits(d, x + 100, y + 22, ci, 8, 9, 22, 4)
    y += 82
    for ln in wrap(c["title"], F_CTITLE, RW - 30)[:2]:
        d.text((x, y), ln, font=F_CTITLE, fill=TEXT)
        y += 48
    ruler(d, x, y + 10, RW - 24, 8, 5, 0.25)
    y += 22
    for ln in wrap(c["sub"], F_MONO_S, RW - 30)[:3]:
        d.text((x, y), ln, font=F_MONO_S, fill=TEXT2)
        y += 21
    c["_rows_y"] = y + 14
    # minimap window
    window(d, (RX, VY + 520, RX + RW, VY + VH), "view", None, "1280 x 800")
    return im


MINI = (RX + 12, VY + 584, 376, 235)   # x, y, w, h of the minimap (1280 x 800 scaled 0.29375)


def card_frame(c, ci, t, total_chapters):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    grid(d)
    cx, cy = 260, 430
    a = min(1.0, t / 0.5)
    d.text((cx, cy - 80), "%02x" % ci if ci else "neurosonde / video manual", font=F_MONO_L, fill=grey(0.6 * a))
    if ci:
        bits(d, cx + 70, cy - 74, ci, 8, 10, 22, 5)
    d.text((cx - 6, cy - 20), c["title"], font=F_TITLE, fill=grey(0.93 * a))
    u = ease(t / 1.2)
    ruler(d, cx, cy + 112, 1400 * u, 14, 7, 0.35)
    d.line([(cx, cy + 113), (cx + 1400 * u, cy + 113)], fill=grey(0.35))
    if t > 0.6:
        b = min(1.0, (t - 0.6) / 0.5)
        d.text((cx, cy + 140), c["sub"], font=F_SANS, fill=grey(0.6 * b))
        pts = [s["point"] for s in c["segs"]]
        yy = cy + 210
        for i, p in enumerate(pts):
            d.text((cx, yy), "%02x" % i, font=F_MONO, fill=grey(0.35 * b))
            d.text((cx + 50, yy), p, font=F_MONO, fill=grey(0.6 * b))
            yy += 28
    corners(d, (cx - 40, cy - 130, cx + 1440, cy + 200 + 28 * max(4, len(c["segs"]))), 0.5 * a, 6)
    return im


# ---------------------------------------------------------------------------
# One chapter, frame by frame, piped to ffmpeg.

def mark_rect(m, v):
    """A screenshot rect -> video coordinates through the view v."""
    x, y, w, h = m[:4]
    s = VW / v[2]
    return (VX + (x - v[0]) * s, VY + (y - v[1]) * s, VX + (x + w - v[0]) * s, VY + (y + h - v[1]) * s)


def draw_mark(d, r, label, side, a):
    x0, y0, x1, y1 = r
    # keep it inside the viewport
    x0, x1 = max(VX + 2, x0), min(VX + VW - 2, x1)
    y0, y1 = max(VY + 2, y0), min(VY + VH - 2, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    ln = min(14, (x1 - x0) / 3, (y1 - y0) / 3) * min(1.0, a * 1.5)
    brackets(d, (x0 - 3, y0 - 3, x1 + 3, y1 + 3), max(3, ln), grey(a))
    if not label:
        return
    tw = F_MONO.getlength(label)
    lead = 26
    if side == "n":
        ax, ay = (x0 + x1) / 2, y0 - 3
        lx, ly = ax, ay - lead
        box = (lx - tw / 2 - 8, ly - 26, lx + tw / 2 + 8, ly)
    elif side == "s":
        ax, ay = (x0 + x1) / 2, y1 + 3
        lx, ly = ax, ay + lead
        box = (lx - tw / 2 - 8, ly, lx + tw / 2 + 8, ly + 26)
    elif side == "e":
        ax, ay = x1 + 3, (y0 + y1) / 2
        lx, ly = ax + lead, ay
        box = (lx, ly - 13, lx + tw + 16, ly + 13)
    else:
        ax, ay = x0 - 3, (y0 + y1) / 2
        lx, ly = ax - lead, ay
        box = (lx - tw - 16, ly - 13, lx, ly + 13)
    # nudge the label into the viewport
    dx = max(0, VX + 4 - box[0]) - max(0, box[2] - (VX + VW - 4))
    dy = max(0, VY + 4 - box[1]) - max(0, box[3] - (VY + VH - 4))
    box = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
    lx, ly = lx + dx, ly + dy
    d.line([(ax, ay), (lx, ly)], fill=grey(0.7 * a))
    d.rectangle(box, fill=BG, outline=grey(0.55 * a))
    d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + 1), label, font=F_MONO, fill=grey(a), anchor="mm")


def render_chapter(args):
    ci, c, chapters, starts, dur, t0, env, path = args
    if os.path.exists(path + ".done") and open(path + ".done").read() == "%.3f" % dur:
        return path
    base = base_frame(c, chapters, ci)
    mini_cache = {}
    nframes = int(round(dur * FPS))
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                           "-tune", "stillimage", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in range(nframes):
        t = f / FPS
        ff.stdin.write(frame(c, ci, chapters, starts, dur, t0, env, base, mini_cache, t).tobytes())
    ff.stdin.close()
    ff.wait()
    open(path + ".done", "w").write("%.3f" % dur)
    return path


def frame(c, ci, chapters, starts, dur, t0, env, base, mini_cache, t):
    gt = t0 + t
    if t < CARD:
        im = card_frame(c, ci, t, len(chapters))
        if t > CARD - 0.25:   # a short fade to black into the page
            im = Image.blend(Image.new("RGB", (W, H), BG), im, max(0.0, (CARD - t) / 0.25))
        footer(ImageDraw.Draw(im), c, ci, chapters, gt, t, None, None, env)
        return im
    name, v, k, into, u = camera(c, starts, t)
    im = base.copy()
    d = ImageDraw.Draw(im)
    src = shot(name)
    view = src.resize((VW, VH), Image.LANCZOS, box=(v[0], v[1], v[0] + v[2], v[1] + v[3]))
    seg = c["segs"][k]
    marks = seg.get("marks", [])
    a = ease((into - (MOVE if (k > 0 and c["segs"][k - 1]["shot"] == name) else MOVE * 1.4) + 0.2) / 0.45)
    if marks and a > 0:
        # dim what is not being explained (to 45 %), keep the marked controls at full brightness
        dim = view.point(lambda p: int(p * (1 - 0.55 * a)))
        for m in marks:
            r = mark_rect(m, v)
            box = tuple(int(round(q - VX if i % 2 == 0 else q - VY)) for i, q in enumerate((r[0] - 4, r[1] - 4, r[2] + 4, r[3] + 4)))
            box = (max(0, box[0]), max(0, box[1]), min(VW, box[2]), min(VH, box[3]))
            if box[2] > box[0] and box[3] > box[1]:
                dim.paste(view.crop(box), box[:2])
        view = dim
    im.paste(view, (VX, VY))
    # viewport frame + title line
    d.rectangle([VX - 1, VY - 1, VX + VW, VY + VH], outline=EDGE)
    corners(d, (VX - 1, VY - 1, VX + VW, VY + VH), 0.7, 6)
    page = {"page0": "constellation", "page1": "network", "page2": "synthesis", "page3": "patch notes",
            "page4": "corpus", "space/page5": "space", "page6": "lab"}[name]
    d.text((VX, VY - 34), "page %s  %s%s" % (name[-1], page, "   (neurosonde space)" if name.startswith("space") else ""),
           font=F_MONO_S, fill=TEXT2)
    zoom = SW / v[2]
    d.text((VX + VW, VY - 34), "zoom %.2fx   x %4d  y %4d   %4d x %3d" % (zoom, v[0], v[1], v[2], v[3]),
           font=F_MONO_S, fill=MUTED, anchor="ra")
    ruler(d, VX, VY - 8, VW, 7, 5, 0.18)
    for m in marks:
        if a > 0:
            draw_mark(d, mark_rect(m, v), m[4], m[5], a)
    # the chapter's points: the one being spoken lit
    y = c["_rows_y"]
    for i, s in enumerate(c["segs"]):
        row = (RX + 10, y - 3, RX + RW - 10, y + 23)
        if i == k:
            d.rectangle(row, fill=SELECTED)
            d.rectangle((row[0], row[1], row[0] + 2, row[3]), fill=WHITE)
            col, icol = TEXT, TEXT2
        elif i < k:
            col, icol = TEXT2, MUTED
        else:
            col, icol = MUTED, (0x40,) * 3
        d.text((RX + 20, y), "%02x" % i, font=F_MONO_S, fill=icol)
        lab = s["point"]
        while F_MONO_S.getlength(lab) > RW - 90:
            lab = lab[:-2]
        d.text((RX + 56, y), lab, font=F_MONO_S, fill=col)
        y += 26
    # minimap: the whole screenshot, the view as a hairline rectangle
    if name not in mini_cache:
        mini_cache[name] = src.resize((MINI[2], MINI[3]), Image.LANCZOS).point(lambda p: int(p * 0.6))
    im.paste(mini_cache[name], (MINI[0], MINI[1]))
    s = MINI[2] / SW
    vr = (MINI[0] + v[0] * s, MINI[1] + v[1] * s, MINI[0] + (v[0] + v[2]) * s, MINI[1] + (v[1] + v[3]) * s)
    d.rectangle(vr, outline=WHITE)
    brackets(d, (vr[0] - 3, vr[1] - 3, vr[2] + 3, vr[3] + 3), 6, WHITE)
    d.rectangle((MINI[0], MINI[1], MINI[0] + MINI[2], MINI[1] + MINI[3]), outline=EDGE)
    footer(d, c, ci, chapters, gt, t, k, zoom, env)
    if t > dur - 0.2:
        im = Image.blend(Image.new("RGB", (W, H), BG), im, max(0.0, (dur - t) / 0.2))
    return im


def footer(d, c, ci, chapters, gt, t, k, zoom, env):
    """The data line, like the plug-in's footer: everything here is measured."""
    fi = int(round(gt * FPS))
    lv = env[min(len(env) - 1, int(t * FPS))] if len(env) else 0.0
    db = 20 * math.log10(lv + 1e-6)
    seg = "card" if k is None else "seg %02x/%02x" % (k, len(c["segs"]))
    line = "t %s   frame %06d   ch %02x/%02x %s   %s" % (tc(gt), fi, ci, len(chapters) - 1, c["title"], seg)
    y = H - 24
    d.text((40, y), line, font=F_MONO_S, fill=MUTED)
    # voice level: a ruler and a bar (the narration, measured)
    x = 960
    d.text((x, y), "voice", font=F_MONO_S, fill=MUTED)
    ruler(d, x + 56, y + 14, 200, 8, 4, 0.18)
    d.rectangle([x + 56, y + 7, x + 56 + 200 * max(0.0, min(1.0, (db + 60) / 60)), y + 8], fill=WHITE)
    d.text((x + 268, y), ("%5.1f db" % db) if db > -60 else " -inf db", font=F_MONO_S, fill=MUTED)
    # the frame number in binary cells
    bits(d, 1340, y + 2, fi, 16, 6, 11, 3)
    d.text((W - 40, y), "%s" % "~/neural-instrument/tools/video", font=F_MONO_S, fill=MUTED, anchor="ra")


# ---------------------------------------------------------------------------
# Audio.

def speak_all(texts):
    os.makedirs(TTS_DIR, exist_ok=True)
    paths, jobs = [], []
    for s in texts:
        h = hashlib.sha1(("%s|%.2f|%s" % (VOICE, SPEED, s)).encode()).hexdigest()[:16]
        p = os.path.join(TTS_DIR, "tts_%s.wav" % h)
        paths.append(p)
        jobs.append(dict(text=s, out=p))
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    if todo:
        jf = os.path.join(WORK, "tts_jobs.json")
        json.dump(todo, open(jf, "w"))
        subprocess.run([TTS_PY, os.path.join(HERE, "tts_kokoro.py"), jf, VOICE, str(SPEED)], check=True)
        os.remove(jf)
    out = []
    for p in paths:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                             check=True, stdout=subprocess.PIPE).stdout
        v = np.frombuffer(raw, np.float32).copy()
        # trim Kokoro's leading / trailing silence so the timing is the speech
        nz = np.nonzero(np.abs(v) > 0.004)[0]
        if len(nz):
            v = v[max(0, nz[0] - 480): nz[-1] + 2400]
        out.append(v)
    return out


def ensure_inputs(seconds, want_bed):
    if not os.path.exists(os.path.join(SHOTS, "page6.png")):
        os.makedirs(SHOTS, exist_ok=True)
        subprocess.run([os.path.join(ROOT, "build/NeuroTest_artefacts/Release/NeuroTest"), "--shots", SHOTS], check=False)
    if not os.path.exists(os.path.join(SHOTS_SPACE, "page5.png")):
        os.makedirs(SHOTS_SPACE, exist_ok=True)
        subprocess.run([os.path.join(ROOT, "build/NeuroSpaceTest_artefacts/Release/NeuroSpaceTest"), "--shots", SHOTS_SPACE], check=False)
    bed = os.path.join(WORK, "bed.wav")
    if want_bed:
        ok = False
        if os.path.exists(bed):
            info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", bed],
                                  capture_output=True, text=True).stdout.strip()
            ok = info and float(info) >= seconds - 0.1
        if not ok and os.path.exists(AUDIO_PY):
            subprocess.run([AUDIO_PY, os.path.join(HERE, "render_bed.py"), "%.2f" % (seconds + 1), bed], check=True)
    return bed if want_bed and os.path.exists(bed) else None


def mix_audio(total, placed, bed_path, out):
    n = int(math.ceil(total * SR))
    voice = np.zeros(n, np.float32)
    for start, v in placed:
        s = int(start * SR)
        e = min(n, s + len(v))
        voice[s:e] += v[:e - s]
    vpk = np.abs(voice).max()
    if vpk > 0:
        voice *= 0.8 / vpk
    mix = np.stack([voice, voice], 1)
    if bed_path:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", bed_path, "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"],
                             check=True, stdout=subprocess.PIPE).stdout
        bed = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
        if len(bed) < n:
            bed = np.concatenate([bed] * int(math.ceil(n / max(1, len(bed)))))
        bed = bed[:n]
        rms = np.sqrt((bed ** 2).mean()) + 1e-9
        bed *= 10 ** (-20 / 20) / rms           # -20 dB rms, then ducked
        # duck under the voice: an envelope follower (attack 40 ms, release 600 ms)
        env = np.zeros(n, np.float32)
        att, rel = 1 - math.exp(-64 / (0.04 * SR)), 1 - math.exp(-64 / (0.6 * SR))
        acc = 0.0
        for i in range(0, n, 64):
            x = float(np.abs(voice[i:i + 64]).max())
            acc += (att if x > acc else rel) * (x - acc)
            env[i:i + 64] = acc
        speaking = np.clip(env * 10, 0, 1)
        gain = 0.25 - 0.15 * speaking             # about -32 dB in the gaps, -40 dB under the voice
        fade = np.ones(n, np.float32)
        fi, fo = int(3 * SR), int(4 * SR)
        fade[:fi] = np.linspace(0, 1, fi)
        fade[-fo:] = np.linspace(1, 0, fo)
        mix += bed * (gain * fade)[:, None]
    pk = np.abs(mix).max()
    if pk > 0.84:                                 # headroom for the AAC encoder (about -1.5 dBFS)
        mix *= 0.84 / pk
    mix.astype(np.float32).tofile(out)
    return voice


def frame_env(voice, t0, dur):
    a, b = int(t0 * SR), int((t0 + dur) * SR)
    seg = voice[a:b]
    hop = SR // FPS
    k = len(seg) // hop
    if k == 0:
        return np.zeros(1)
    return np.sqrt((seg[:k * hop].reshape(k, hop) ** 2).mean(1))


# ---------------------------------------------------------------------------

def overlay():
    """Every mark of every chapter drawn on its screenshot (to check the coordinates)."""
    out = os.path.join(WORK, "overlay")
    os.makedirs(out, exist_ok=True)
    for ci, c in enumerate(CHAPTERS):
        for k, s in enumerate(c["segs"]):
            im = shot(s["shot"]).copy()
            d = ImageDraw.Draw(im)
            if s.get("focus"):
                x, y, w, h = s["focus"]
                d.rectangle([x, y, x + w, y + h], outline=(160, 160, 160))
            for m in s.get("marks", []):
                x, y, w, h = m[:4]
                d.rectangle([x, y, x + w, y + h], outline=WHITE)
                d.text((x + 2, y + h + 2), m[4], font=F_MONO_S, fill=WHITE)
            im.save(os.path.join(out, "c%02d_s%02d.png" % (ci, k)))
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/Neurosonde-Video-Manual.mp4"))
    ap.add_argument("--only", help="chapter range a-b (preview)")
    ap.add_argument("--still", type=float, help="write one PNG per segment at this many seconds into it, and stop")
    ap.add_argument("--overlay", action="store_true", help="draw every mark on the screenshots, and stop")
    ap.add_argument("--print", action="store_true", help="print the narration and stop")
    ap.add_argument("--no-bed", action="store_true", help="no Neurosonde bed under the voice")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    if a.print:
        for ci, c in enumerate(CHAPTERS):
            print("== %02d %s" % (ci, c["title"]))
            for s in c["segs"]:
                print("  [%s] %s" % (s["point"], s["say"]))
        return
    ensure_inputs(0, False)
    if a.overlay:
        overlay()
        return

    sel = list(range(len(CHAPTERS)))
    if a.only:
        lo, hi = (int(x) for x in a.only.split("-"))
        sel = sel[lo:hi + 1]
    texts = [s["say"] for ci in sel for s in CHAPTERS[ci]["segs"]]
    voices = speak_all(texts)
    vi = 0
    plans, placed, t0 = [], [], 0.0
    for ci in sel:
        c = CHAPTERS[ci]
        vs = voices[vi:vi + len(c["segs"])]
        vi += len(c["segs"])
        starts, dur = chapter_plan(c, [len(v) / SR for v in vs])
        for st, v in zip(starts, vs):
            placed.append((t0 + st + 0.25, v))   # the voice starts just after the camera does
        plans.append((ci, starts, dur, t0))
        t0 += dur
    total = t0
    print("total %.1f s (%d:%02d), %d chapters" % (total, total // 60, total % 60, len(sel)))

    if a.still is not None:
        out = os.path.join(WORK, "stills")
        os.makedirs(out, exist_ok=True)
        for ci, starts, dur, t0 in plans:
            c = CHAPTERS[ci]
            base = base_frame(c, CHAPTERS, ci)
            for k, st in enumerate(starts):
                tt = min(st + a.still, dur - 0.3)
                frame(c, ci, CHAPTERS, starts, dur, t0, np.zeros(1), base, {}, tt).save(
                    os.path.join(out, "c%02d_s%02d.png" % (ci, k)))
            frame(c, ci, CHAPTERS, starts, dur, t0, np.zeros(1), base, {}, 1.8).save(os.path.join(out, "c%02d_card.png" % ci))
        print(out)
        return

    bed = None if a.no_bed else ensure_inputs(total, True)
    audio = os.path.join(WORK, "audio.f32")
    voice = mix_audio(total, placed, bed, audio)
    parts_dir = os.path.join(WORK, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    jobs = [(ci, CHAPTERS[ci], CHAPTERS, starts, dur, t0, frame_env(voice, t0, dur), os.path.join(parts_dir, "c%02d.mp4" % ci))
            for ci, starts, dur, t0 in plans]
    with Pool(min(a.jobs, len(jobs))) as pool:
        parts = pool.map(render_chapter, jobs)
    listing = os.path.join(WORK, "parts.txt")
    open(listing, "w").write("".join("file '%s'\n" % p for p in parts))
    video = os.path.join(WORK, "video.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", video], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", audio,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart",
                    os.path.expanduser(a.out)], check=True)
    # timing sheet: when each chapter and segment starts (to check sync)
    with open(os.path.join(WORK, "timing.txt"), "w") as f:
        for ci, starts, dur, t0 in plans:
            f.write("%s  %02d %s (%.1f s)\n" % (tc(t0), ci, CHAPTERS[ci]["title"], dur))
            for k, st in enumerate(starts):
                f.write("   %s  seg %02d  %s\n" % (tc(t0 + st), k, CHAPTERS[ci]["segs"][k]["point"]))
    print(os.path.expanduser(a.out))


if __name__ == "__main__":
    main()
