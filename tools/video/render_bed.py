#!/usr/bin/env python3
"""The quiet bed under the video manual: sound actually made by Neurosonde.

  build/video/.venv-audio/bin/python tools/video/render_bed.py SECONDS OUT.wav

Hosts the installed VST3 (~/Library/Audio/Plug-Ins/VST3/Neurosonde.vst3) with
pedalboard and plays a slow, plain source through it (a drifting sine chord and
a little band-limited noise). The plug-in resynthesises it from what its brain
has learned: phantom (magnitude-only resynthesis) and grains (Hopfield recall),
tilted dark, into the nebula.

The brain is a COPY of the test brain the NeuroTest run trains
(/private/tmp/neurosonde_test_brain -> build/video/brain), pointed to by
NEUROSONDE_BRAIN_DIR, so the user's real brain is never touched.

Setup (once): python3 -m venv build/video/.venv-audio &&
              build/video/.venv-audio/bin/pip install pedalboard numpy soundfile
"""
import os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(ROOT, "build", "video")
BRAIN = os.path.join(WORK, "brain")
TEST_BRAIN = "/private/tmp/neurosonde_test_brain"
PLUGIN = os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/Neurosonde.vst3")
SR = 48000


def source(seconds):
    import numpy as np
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    out = np.zeros((2, n), np.float32)
    # a slow chord that moves every 16 s between four voicings (Hz)
    chords = [[110.0, 164.8, 220.0, 329.6], [98.0, 146.8, 233.1, 293.7],
              [87.3, 130.8, 220.0, 261.6], [103.8, 155.6, 207.7, 311.1]]
    seg = 16.0
    for ci in range(int(seconds / seg) + 1):
        a, b = int(ci * seg * SR), min(n, int((ci + 1) * seg * SR + 4 * SR))
        if a >= n:
            break
        tt = t[a:b] - ci * seg
        env = np.clip(tt / 4.0, 0, 1) * np.clip((seg + 4 - tt) / 4.0, 0, 1)
        for k, f in enumerate(chords[ci % 4]):
            lfo = 0.55 + 0.45 * np.sin(2 * np.pi * (0.05 + 0.03 * k) * tt + k)
            x = 0.06 * env * lfo * np.sin(2 * np.pi * f * tt + 0.3 * np.sin(2 * np.pi * 0.2 * tt))
            out[0, a:b] += x * (0.6 + 0.1 * k)
            out[1, a:b] += x * (1.0 - 0.1 * k)
    # band-limited noise, swelling slowly
    w = rng.standard_normal((2, n)).astype(np.float32)
    k = 64
    w = np.stack([np.convolve(c, np.ones(k) / k, mode="same") for c in w])
    out += (0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 23.0)) ** 3 * w).astype(np.float32)
    return out


def main():
    seconds, path = float(sys.argv[1]), sys.argv[2]
    import numpy as np
    import soundfile as sf
    if not os.path.isdir(BRAIN):
        if not os.path.isdir(TEST_BRAIN):
            sys.exit("no trained test brain: run build/NeuroTest_artefacts/Release/NeuroTest --shots <dir> first")
        shutil.copytree(TEST_BRAIN, BRAIN)
    os.environ["NEUROSONDE_BRAIN_DIR"] = BRAIN   # never the user's brain
    import pedalboard
    p = pedalboard.load_plugin(PLUGIN)
    # phantom + grains, dark and slow (tried: the four default layers were a bright harmonic buzz)
    # (pedalboard sets parameters in their displayed units: percentages are 0-100, not 0-1)
    for k, v in dict(layer_1_engine="phantom", layer_2_on=True, layer_2_engine="grains", tilt=-0.5,
                     lucidity=60.0, nebula=60.0, nebula_decay=80.0, dry_wet=100.0).items():
        try:
            setattr(p, k, v)
        except Exception as e:
            print("param", k, e)
    src = source(seconds + 2)
    y = p.process(src, SR, buffer_size=512, reset=False)
    y = y[:, int(2 * SR):]          # skip the first 2 s (the plug-in settling)
    y = y[:, :int(seconds * SR)]
    pk = float(np.abs(y).max()) or 1.0
    y = y / pk * 0.9
    sf.write(path, y.T, SR, subtype="FLOAT")
    print("bed %.1f s, peak %.3f -> %s" % (y.shape[1] / SR, pk, path))


if __name__ == "__main__":
    main()
