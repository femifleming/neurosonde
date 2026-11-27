#!/usr/bin/env python3
"""Narration for the video manual: Kokoro, British English, voice "Emma" (bf_emma),
the same narrator as the Cirrostratus and AERIA OS videos.

  <kokoro venv>/bin/python tools/video/tts_kokoro.py jobs.json [voice] [speed]

jobs.json is a list of {"text": ..., "out": ".../x.wav"}. Existing outputs are
skipped (build_video.py names them by a hash of voice, speed and text, so re-runs
only speak what changed). Output: 24 kHz mono WAV. Misaki's inline phonemes work:
[word](/phonemes/).

The venv is the Cirrostratus one (~/cirrostratus/cirrostratus-monsoon/tools/video/.venv);
to make one: python3.12 -m venv .venv && .venv/bin/pip install kokoro soundfile
"""
import json, os, sys

import numpy as np
import soundfile as sf
from kokoro import KPipeline


def main():
    voice = sys.argv[2] if len(sys.argv) > 2 else "bf_emma"
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    jobs = [j for j in json.load(open(sys.argv[1])) if not os.path.exists(j["out"])]
    if not jobs:
        return
    pipe = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")
    for i, j in enumerate(jobs):
        parts = [a.numpy() for _, _, a in pipe(j["text"], voice=voice, speed=speed)]
        audio = np.concatenate(parts) if parts else np.zeros(2400, np.float32)
        tmp = j["out"] + ".tmp.wav"
        sf.write(tmp, audio, 24000)
        os.replace(tmp, j["out"])
        print("tts %d/%d" % (i + 1, len(jobs)), flush=True)


if __name__ == "__main__":
    main()
