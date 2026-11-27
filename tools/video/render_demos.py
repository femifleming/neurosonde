#!/usr/bin/env python3
"""The audio demonstration in the video manual: real sound through the real plug-in.

  build/video/.venv-audio/bin/python tools/video/render_demos.py SOURCES_DIR OUT_DIR

Hosts the installed VST3 with pedalboard, on a FRESH brain in OUT_DIR/brain
(NEUROSONDE_BRAIN_DIR: the user's own brain is never touched), and plays it
in real time (the audio is fed block by block at the speed it would play, so
the network learns exactly what it would learn live). One continuous session:

  01 the input            the spoken phrase, dry
  02 an empty brain       its first hearing: the network knows nothing yet
  03 after a minute       the same phrase after about 75 s of listening
  04 lucidity 0           it stops following the input and dreams forward
  05 grains               Hopfield recall: remembered moments of what it heard
  06 a new sound          a plucked melody, learning paused: answered in the voice
  07 - 11                 new in 1.1: bowed, modal, membrane, scanned, swarm
  12 morph                two layers, modal and swarm, halfway
  13 hydrophone           a preset, on the phrase

Writes OUT_DIR/NN_in.wav and NN_out.wav (48 kHz stereo float, each normalised
to the same loudness so in and out compare fairly) and OUT_DIR/demos.json.
SOURCES_DIR holds voice.wav (Kokoro) and pluck.wav (made here if missing).
"""
import json, os, shutil, sys, time

import numpy as np
import soundfile as sf

SR = 48000
BLOCK = 512
PLUGIN = os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/Neurosonde.vst3")


def load(path):
    from pedalboard.io import AudioFile
    with AudioFile(path).resampled_to(SR) as f:
        x = f.read(f.frames)
    x = x.mean(0) if x.shape[0] > 1 else x[0]
    return x.astype(np.float32)


def pluck_melody(seconds=9.0, seed=3):
    """A plucked string melody (Karplus-Strong), a slow phrase in A minor."""
    rng = np.random.default_rng(seed)
    notes = [57, 60, 64, 62, 60, 59, 57, 52, 55, 57, 60, 64, 67, 64]
    dur = seconds / len(notes)
    out = np.zeros(int(seconds * SR) + SR, np.float32)
    for i, m in enumerate(notes):
        f = 440.0 * 2 ** ((m - 69) / 12)
        p = int(round(SR / f))
        buf = rng.uniform(-1, 1, p).astype(np.float32)
        n = int(dur * 1.8 * SR)
        y = np.zeros(n, np.float32)
        for k in range(n):
            j = k % p
            y[k] = buf[j]
            buf[j] = 0.996 * 0.5 * (buf[j] + buf[(j + 1) % p])
        s = int(i * dur * SR)
        out[s:s + n] += y * 0.5
    return out[:int(seconds * SR)]


def norm(x, rms=0.08):
    """To the same loudness (rms), peak at most 0.9."""
    r = float(np.sqrt((x ** 2).mean())) + 1e-9
    y = x * (rms / r)
    pk = float(np.abs(y).max())
    if pk > 0.9:
        y *= 0.9 / pk
    return y


class Session:
    def __init__(self, plugin):
        self.p = plugin
        self.heard = 0.0

    def play(self, mono, pace=True):
        """Feed mono audio through in real time; returns the stereo output (2, n)."""
        n = len(mono)
        x = np.stack([mono, mono]).astype(np.float32)
        out = np.zeros((2, n), np.float32)
        t0 = time.time()
        for a in range(0, n, BLOCK):
            b = min(n, a + BLOCK)
            out[:, a:b] = self.p.process(x[:, a:b], SR, buffer_size=BLOCK, reset=False)
            if pace:
                ahead = (b / SR) - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)
        self.heard += n / SR
        return out

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self.p, k, v)


def phrase_with_gap(v, gap=1.0):
    return np.concatenate([v, np.zeros(int(gap * SR), np.float32)])


def main():
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    brain = os.path.join(out_dir, "brain")
    shutil.rmtree(brain, ignore_errors=True)
    os.makedirs(brain)
    os.environ["NEUROSONDE_BRAIN_DIR"] = brain     # a fresh brain, never the user's

    voice = load(os.path.join(src_dir, "voice.wav"))
    voice = 0.5 * voice / (np.abs(voice).max() + 1e-9)
    pluck_path = os.path.join(src_dir, "pluck.wav")
    if not os.path.exists(pluck_path):
        sf.write(pluck_path, pluck_melody(), SR, subtype="FLOAT")
    pluck = load(pluck_path)
    pluck = 0.5 * pluck / (np.abs(pluck).max() + 1e-9)

    import pedalboard
    s = Session(pedalboard.load_plugin(PLUGIN))
    s.set(program="first light")
    s.set(dry_wet=100.0, learning="online", listen=True)
    demos = []

    def keep(code, title, point, inp, out, settings, play):
        sf.write(os.path.join(out_dir, code + "_in.wav"), np.stack([norm(inp), norm(inp)]).T, SR, subtype="FLOAT")
        o = out.copy()
        r = float(np.sqrt((o ** 2).mean())) + 1e-9
        o *= 0.08 / r
        pk = float(np.abs(o).max())
        if pk > 0.9:
            o *= 0.9 / pk
        sf.write(os.path.join(out_dir, code + "_out.wav"), o.T, SR, subtype="FLOAT")
        demos.append(dict(code=code, title=title, point=point, settings=settings, play=play,
                          input="plucked melody" if inp is q else "the spoken phrase",
                          seconds=len(inp) / SR, heard=round(s.heard, 1),
                          out_rms_raw=r))
        print("%s %-22s heard %5.1f s  out rms %.4f" % (code, title, s.heard, r), flush=True)

    # (pedalboard sets parameters in their displayed units: percentages are 0-100, not 0-1)
    def layers(*engines, character=30.0):
        for k in range(4):
            on = k < len(engines)
            setattr(s.p, "layer_%d_on" % (k + 1), on)
            if on:
                setattr(s.p, "layer_%d_engine" % (k + 1), engines[k])
                setattr(s.p, "layer_%d_character" % (k + 1), character)

    p = phrase_with_gap(voice)
    q = phrase_with_gap(pluck, 0.8)      # (keep() tells the two inputs apart by identity)
    layers("harmonic")
    s.set(lucidity=70.0, temperature=50.0, combine="stack")
    base = [("engine", "harmonic"), ("lucidity", "70 %"), ("learning", "online")]
    # 01, 02: the input and the first hearing
    o = s.play(p)
    keep("01", "the input", "a spoken phrase, dry", p, o, [("source", "kokoro voice, 8 s")], ["in"])
    keep("02", "an empty brain", "its first hearing", p, o, base + [("heard", "0 s")], ["out"])
    # about 75 s of listening
    while s.heard < 80:
        s.play(p)
    o = s.play(p)
    keep("03", "after a minute", "the same phrase, learned", p, o, base + [("heard", "%d s" % round(s.heard - len(p) / SR))],
         ["in", "out"])
    # 04 dreaming
    s.set(lucidity=0.0, temperature=90.0)
    s.play(p[: int(2 * SR)])
    o = s.play(p)
    keep("04", "lucidity 0", "it dreams forward", p, o, [("engine", "harmonic"), ("lucidity", "0 %"), ("temperature", "90 %")],
         ["out"])
    s.set(lucidity=70.0, temperature=50.0)
    # 05 grains (Hopfield recall)
    layers("grains"); s.set(grain_size_ms=160.0, spray=40.0)
    s.play(p[: int(2 * SR)])
    o = s.play(p)
    keep("05", "grains", "remembered moments", p, o, [("engine", "grains"), ("grain size", "160 ms"), ("lucidity", "70 %")],
         ["out"])
    # 06 a new sound, learning paused
    layers("harmonic"); s.set(listen=False)
    s.play(q[: int(2 * SR)])
    o = s.play(q)
    keep("06", "a new sound", "a pluck, in the voice it learned", q, o,
         [("source", "plucked melody"), ("engine", "harmonic"), ("listen", "off (paused)")], ["in", "out"])
    s.set(listen=True)
    # 07 - 11: new in 1.1, five of the seven new engines, one at a time, on the pluck (learning paused)
    s.set(listen=False)
    for code, eng, char, point, src in (("07", "bowed", 40.0, "a bowed string, friction and body", q),
                                        ("08", "modal", 80.0, "forty modes, rung like a bell", q),
                                        ("09", "membrane", 40.0, "a struck skin, pitch bending", q),
                                        ("10", "scanned", 50.0, "a ring of masses, pushed by the spectrum", p),
                                        ("11", "swarm", 40.0, "partials flocking to the peaks", p)):
        layers(eng, character=char)
        s.play(src[: int(2 * SR)])
        o = s.play(src)
        keep(code, eng, point, src, o, [("engine", eng), ("character", "%d %%" % char), ("listen", "off")], ["out"])
    s.set(listen=True)
    # 12 two layers stacked
    layers("modal", "swarm", character=60.0); s.set(combine="morph", morph=50.0)
    s.play(p[: int(2 * SR)])
    o = s.play(p)
    keep("12", "morph", "modal and swarm, halfway", p, o, [("layers", "modal + swarm"), ("combine", "morph 50 %")], ["out"])
    # 13 a preset
    s.set(program="hydrophone")
    s.play(p[: int(3 * SR)])
    o = s.play(p)
    keep("13", "hydrophone", "a preset, on the phrase", p, o, [("preset", "hydrophone"), ("layers", "as stored")], ["out"])

    json.dump(dict(sr=SR, demos=demos), open(os.path.join(out_dir, "demos.json"), "w"), indent=1)
    print("demos ->", out_dir)


if __name__ == "__main__":
    main()
