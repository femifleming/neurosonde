"""The Neurosonde video manual: the script.

One chapter per page / topic of README.md. Each chapter is a title card
followed by segments; a segment is one spoken paragraph with what the camera
looks at while it is spoken:

  shot     which screenshot (page0..page6, from NeuroTest / NeuroSpaceTest --shots)
  focus    (x, y, w, h) in screenshot pixels (1280 x 800); None = the whole window
  marks    hairline bracket callouts: (x, y, w, h, "label", side) with side n / s / e / w
  point    the short lowercase line that lights up in the chapter list while it is spoken
  say      the narration (Kokoro speaks it as written; spelled for the ear)

Coordinates are measured from the screenshots and the layout in Source/Editor.cpp
(Editor::resized). Run build_video.py --overlay to draw every mark on the shots.
"""

S = "page0"  # constellation
N = "page1"  # network
Y = "page2"  # synthesis
P = "page3"  # patch notes
C = "page4"  # corpus
SP = "space/page5"  # space (NeuroSpaceTest)
L = "page6"  # lab
G = "page7"  # gain staging (1.1)

MAP = (408, 106, 464, 464)          # the 16 x 16 SOM grid
HUD = (420, 640, 440, 128)

CHAPTERS = [
    dict(title="neurosonde", sub="an audio effect that trains a neural network on what it hears", segs=[
        dict(shot=S, focus=None, point="audio in, audio out", marks=[],
             say="This is neurosonde. It is an audio effect: audio in, audio out, and no MIDI. "
                 "Put it on a track, and it learns from the sound that passes through it."),
        dict(shot=S, focus=(0, 776, 640, 24), point="analysis every 10 ms",
             marks=[(10, 782, 420, 14, "the data line: loss, replay, surprise, level, pitch", "n")],
             say="Every ten milliseconds it analyses the sound: sixty-four mel bands, chroma, pitch, voicing, "
                 "loudness, brightness and noisiness, plus the harmonic and noise targets a synthesiser needs."),
        dict(shot=N, focus=(330, 64, 620, 545), point="a conditional vae with film",
             marks=[(352, 72, 290, 520, "encoder q(z | x, notes)", "s"), (664, 100, 30, 486, "latent z, 16 dims", "n"),
                    (740, 72, 190, 520, "decoder", "s")],
             say="From those frames it trains a real neural network, while you play. At its heart is a conditional "
                 "variational autoencoder, with film layers conditioned on your patch notes. It squeezes each "
                 "moment of sound into sixteen numbers, and learns to rebuild the sound from them."),
        dict(shot=S, focus=(380, 40, 520, 590), point="gru, echo state network, som",
             marks=[MAP + ("self-organising map, 16 x 16", "n")],
             say="Around it: a G.R.U. that predicts the next latent, so it can dream forward; an echo state network, "
                 "a second dreamer; and a sixteen by sixteen self-organising map, which you see as the constellation. "
                 "A modern Hopfield memory recalls remembered moments."),
        dict(shot=S, focus=(10, 36, 270, 350), point="four engine layers",
             marks=[(18, 60, 252, 60, "layer 1: harmonic", "e"), (18, 120, 252, 60, "layer 2: grains", "e"),
                    (18, 180, 252, 60, "layer 3: strings", "e"), (18, 240, 252, 60, "layer 4: phantom", "e")],
             say="Then it resynthesises sound from what it has learned, through four engine layers, "
                 "into a stereo field and a reverb."),
        dict(shot=S, focus=(860, 0, 420, 60), point="one shared brain",
             marks=[(1086, 6, 182, 18, "learning or sleeping / steps / frames", "s")],
             say="There is one brain, shared by every instance in a host. Other processes, like the standalone app, "
                 "merge their learning into it through a brain file. Everything it hears teaches the same network."),
    ]),

    dict(title="hearing it", sub="real sound through the real plug-in: a fresh brain, in real time", kind="demo", segs=[
        dict(demo="01", play=["in"], point="the input",
             say="Before the tour, listen. This is what goes in: a spoken phrase. Every sound in this chapter is the real "
                 "plug-in, starting from an empty brain, fed in real time."),
        dict(demo="02", play=["out"], point="an empty brain",
             say="Its first hearing. The network knows nothing yet, so the engine follows the analysis, frame by frame: "
                 "pitch, loudness, the spectrum. A rough sketch of the voice."),
        dict(demo="03", play=["in", "out"], point="after a minute of listening",
             say="After about eighty seconds of listening it has trained for thousands of steps, and now every frame "
                 "goes through its sixteen numbers and back out. First the phrase as it went in, then as the network "
                 "rebuilds it."),
        dict(demo="04", play=["out"], point="lucidity 0: it dreams",
             say="Turn lucidity down to zero and it stops following the input. The recurrent network dreams forward from "
                 "what it has learned, here at a higher temperature."),
        dict(demo="05", play=["out"], point="grains: remembered moments",
             say="The grains engine plays back remembered moments of what it heard, chosen by Hopfield attention to "
                 "where the network is now."),
        dict(demo="06", play=["in", "out"], point="a new sound, learning paused",
             say="Now a different sound, a plucked melody, with learning paused. The network can only answer with what "
                 "it knows, so the pluck comes back shaped by the voice."),
        dict(demo="07", play=["out"], point="bowed (new in 1.1)",
             say="New in one point one: seven more engines. Bowed is a waveguide string with bow friction and a wooden "
                 "body; the network's loudness is the bow speed."),
        dict(demo="08", play=["out"], point="modal",
             say="Modal rings forty resonant modes tuned to the decoded spectrum. With the character turned up, they "
                 "spread into the inharmonic ratios of a bell."),
        dict(demo="09", play=["out"], point="membrane",
             say="Membrane is a struck drum skin, simulated in two dimensions. Every onset strikes it at a point the "
                 "latent space chooses, and a hard hit bends the pitch."),
        dict(demo="10", play=["out"], point="scanned",
             say="Scanned synthesis: a ring of sixty-four masses on springs, each pushed by one band of the decoded "
                 "spectrum. Its slowly moving shape is the waveform."),
        dict(demo="11", play=["out"], point="swarm",
             say="Swarm: forty-eight partials flocking in frequency, drawn to the peaks of the decoded spectrum and "
                 "kept apart from each other."),
        dict(demo="12", play=["out"], point="two layers, morphed",
             say="Layers combine. Here modal and swarm, morphed halfway."),
        dict(demo="13", play=["out"], point="a preset: hydrophone",
             say="And a preset, hydrophone, on the same phrase. Every clip here stayed under the output ceiling: the "
                 "new gain page shows each stage."),
    ]),

    dict(title="constellation", sub="the main page: the self-organising map, drawn as it is", segs=[
        dict(shot=S, focus=None, point="the main page",
             marks=[(14, 36, 262, 348, "layers", "e"), (14, 392, 262, 380, "dream", "e"),
                    (1004, 36, 262, 348, "overview", "w"), (1004, 392, 262, 380, "learning", "w"),
                    HUD + ("hud", "n")],
             say="The main page is the constellation. Layers and dream on the left, overview and learning on the right, "
                 "the hud at the bottom. Every page is mirror-symmetric about the centre line."),
        dict(shot=S, focus=MAP, point="16 x 16 nodes at their true grid",
             marks=[(408, 106, 29, 29, "node 000", "n"), (843, 541, 29, 29, "node 0ff", "s")],
             say="In the middle is what the network has learned, drawn as it is: a self-organising map, sixteen by "
                 "sixteen nodes that tile the sixteen-dimensional latent space. Each node sits at its true grid position."),
        dict(shot=S, focus=(560, 250, 312, 200), point="shade: the u-matrix",
             marks=[(784, 309, 88, 58, "light walls", "e"), (611, 280, 58, 29, "u-matrix", "n")],
             say="The shade of each cell is the U-matrix: the mean distance from a node's codebook vector to its four "
                 "neighbours. Light walls are boundaries between kinds of sound the network tells apart."),
        dict(shot=S, focus=(408, 100, 464, 160), point="dots: visits",
             marks=[(414, 112, 16, 16, "visited often", "s"), (559, 112, 16, 16, "", "s"), (849, 112, 16, 16, "", "s")],
             say="The dots are visits. The bigger and brighter the dot, the more often the sound has landed on that "
                 "node; the counts slowly decay."),
        dict(shot=S, focus=(700, 100, 200, 130), point="words from the notes",
             marks=[(862, 114, 26, 14, "a patch-note word", "s")],
             say="On the most visited nodes, a word appears: the patch-note word most often active when that node was hit."),
        dict(shot=S, focus=(408, 190, 464, 380), point="output and input latents",
             marks=[(850, 214, 15, 15, "output z, solid", "w"), (424, 542, 15, 15, "input z, hollow", "e")],
             say="The solid square is the output: the latent the network is playing, with its recent path. "
                 "The hollow square is the input's latent. The dashed line between them is how far the network moved the sound."),
        dict(shot=S, focus=(540, 270, 230, 140), point="the probe and gravity",
             marks=[(624, 322, 32, 32, "probe", "w"), (656, 343, 74, 12, "gravity 0%", "s")],
             say="The circle is the probe. Drag it to move it. Alt-drag sets its gravity: how hard it pulls the sound "
                 "toward that spot."),
        dict(shot=S, focus=(408, 40, 464, 80), point="route and autopilot",
             marks=[(522, 66, 236, 14, "route / autopilot", "s")],
             say="Shift-click nodes to build a route, and the autopilot travels it. Command-click clears the route. "
                 "Hover any node for its index, visits, decoded centroid and word."),
        dict(shot=S, focus=(388, 574, 500, 50), point="the read-outs",
             marks=[(392, 593, 200, 13, "output node, quantisation error", "n"),
                    (392, 608, 206, 13, "surprise", "s"), (738, 593, 136, 13, "input node, presence", "n"),
                    (734, 608, 140, 13, "latent distance in to out", "s")],
             say="Under the map, the read-outs. The output node and its quantisation error. Surprise: the echo state "
                 "network's prediction error. The input node and presence. And the latent distance from in to out."),
    ]),

    dict(title="around the map", sub="hud, layers, dream, overview, learning", segs=[
        dict(shot=S, focus=HUD, point="the hud: four sliders",
             marks=[(430, 676, 202, 28, "lucidity", "n"), (648, 676, 202, 28, "temperature", "n"),
                    (430, 720, 202, 28, "drift", "s"), (648, 720, 202, 28, "gravity", "s")],
             say="Below the map is the hud, four sliders. Lucidity is how much the input steers: the latent is a mix of "
                 "the input's latent and the dream, and lucidity sets the mix."),
        dict(shot=S, focus=HUD, point="temperature, drift, gravity",
             marks=[(648, 676, 202, 28, "temperature", "n"), (430, 720, 202, 28, "drift", "s"),
                    (648, 720, 202, 28, "gravity", "s")],
             say="Temperature is how freely it dreams: the G.R.U. samples its next latent at that temperature. "
                 "Drift adds a slow random walk. Gravity is the probe's pull."),
        dict(shot=S, focus=(10, 36, 270, 290), point="layers: engine, character",
             marks=[(22, 64, 30, 24, "on", "s"), (57, 62, 212, 28, "engine", "e"), (22, 94, 120, 26, "level", "s"),
                    (146, 94, 124, 26, "character", "s")],
             say="On the left, the layers. Each has an on switch, an engine, a level and a character, which every engine "
                 "reads in its own way: phase scatter, inharmonicity, grain density, the strings' bridge."),
        dict(shot=S, focus=(10, 36, 270, 290), point="eighteen engines",
             marks=[(57, 62, 212, 28, "harmonic: ddsp", "e"), (57, 122, 212, 28, "grains: cloudfin", "e"),
                    (57, 182, 212, 28, "strings: tantri", "e"), (57, 242, 212, 28, "phantom", "e")],
             say="Click an engine to choose one of eighteen: harmonic, phantom, partials, grains, mosaic, ear tones, "
                 "strings, voice, gendy, table and tape, and, new in one point one, modal, membrane, F.M., bowed, "
                 "scanned, pulsar and swarm."),
        dict(shot=S, focus=(14, 296, 262, 84), point="stack, morph, merge, imprint",
             marks=[(118, 304, 150, 18, "combine", "n"), (118, 358, 150, 18, "morph by", "s")],
             say="The layers combine by stack; by morph; by merge, a spectral log-magnitude interpolation; or by imprint, "
                 "one layer's envelope pressed onto another. Morph moves by hand, by the network, by the stars, "
                 "or by an L.F.O."),
        dict(shot=S, focus=(14, 392, 262, 88), point="dream with, autopilot",
             marks=[(118, 418, 150, 18, "gru / reservoir / both", "n"), (22, 444, 92, 26, "autopilot", "s"),
                    (122, 446, 146, 24, "speed", "s")],
             say="Below is the dream panel. Choose the dreamer: the G.R.U., the reservoir, or both. "
                 "Turn on the autopilot and set its speed, in stars per minute."),
        dict(shot=S, focus=(14, 474, 262, 296), point="the latent as binary cells",
             marks=[(48, 478, 40, 210, "value", "n"), (102, 478, 66, 210, "8 bits", "n"),
                    (174, 478, 96, 210, "input grey, sound white", "n"), (20, 736, 250, 30, "presence, surprise", "s")],
             say="Then the latent itself: sixteen dimensions, each as its value, as eight binary cells, and on a small "
                 "ruler where the grey tick is the input and the white tick is the sound. Presence and surprise sit below."),
        dict(shot=S, focus=(1000, 36, 270, 500), point="overview and learning",
             marks=[(1010, 62, 250, 136, "nearest remembered moments", "s"), (1010, 418, 250, 104, "learning", "s")],
             say="On the right, the overview lists the remembered moments nearest to the sound, with their strength, "
                 "note and level. The learning panel shows the steps, the losses, the replay, and whether the brain is "
                 "learning or sleeping."),
    ]),

    dict(title="network", sub="learning | the network diagram and training curves | memories", segs=[
        dict(shot=N, focus=(330, 64, 620, 545), point="drawn from its real weights",
             marks=[(345, 72, 48, 14, "features 84", "n"), (430, 72, 34, 14, "128 film", "n"),
                    (588, 72, 30, 14, "mu, sigma", "n"), (664, 72, 30, 14, "z 16", "s"), (894, 72, 34, 14, "out 164", "n")],
             say="The network page draws the model from its real weights. Eighty-four input features; a hundred and "
                 "twenty-eight units with film; sixty-four; the mean and spread of sixteen latents; the sampled latent z; "
                 "and the decoder, back up to a hundred and sixty-four outputs: features, harmonics and noise."),
        dict(shot=N, focus=(560, 90, 260, 300), point="bright positive, faint negative",
             marks=[(668, 100, 22, 290, "z: brightens with the sound", "e")],
             say="Bright lines are positive weights, faint lines negative. The latent column follows the current z."),
        dict(shot=N, focus=(330, 612, 620, 155), point="training curves, held out",
             marks=[(334, 628, 124, 124, "loss curves", "n"), (734, 616, 210, 44, "reconstruction, held out vs train", "n"),
                    (826, 666, 118, 42, "active latent units", "s")],
             say="Below are the training curves. Every tenth frame is held out and never trained on, so held-out error "
                 "against training error shows generalisation, not just fit. The bars count the active latent units: "
                 "the check against posterior collapse."),
        dict(shot=N, focus=(10, 40, 300, 270), point="learning: online / offline",
             marks=[(114, 72, 188, 20, "online / offline", "e"), (18, 100, 110, 28, "listen", "e"),
                    (18, 144, 284, 26, "plasticity", "e"), (18, 186, 284, 26, "sleep", "e"),
                    (18, 228, 284, 26, "kl weight beta", "e"), (18, 270, 284, 26, "listen above", "e")],
             say="On the left, learning. Online or offline. Listen, on or off. Plasticity scales the learning rate. "
                 "Sleep sets how much it consolidates from replay when it is quiet. Then the K.L. weight, beta, and the level the input must pass "
                 "to be learned."),
        dict(shot=N, focus=(966, 40, 304, 740), point="memories",
             marks=[(978, 740, 89, 28, "remember", "n"), (1075, 740, 89, 28, "recall", "n"), (1172, 740, 90, 28, "forget all", "n")],
             say="On the right, memories. Remember keeps the brain as it is now, as a named snapshot. Recall goes back "
                 "to the one you select. Forget all returns the shared brain to random weights; saved memories stay."),
    ]),

    dict(title="synthesis", sub="every sound parameter, in four mirrored columns", segs=[
        dict(shot=Y, focus=None, point="four columns",
             marks=[(10, 40, 306, 362, "layers 1 2", "s"), (326, 40, 306, 240, "dream", "s"),
                    (643, 40, 306, 240, "memory, grains", "s"), (961, 40, 310, 362, "spectral, stargate, nebula", "s")],
             say="The synthesis page holds every sound parameter, in four columns, each split at the same height as "
                 "its mirror."),
        dict(shot=S, focus=(14, 56, 262, 250), point="the engines' lineage",
             marks=[(57, 62, 212, 28, "harmonic: ddsp harmonic + noise", "e"), (57, 122, 212, 28, "grains: hopfield recall", "e"),
                    (57, 182, 212, 28, "strings: sympathetic + jawari", "e"), (57, 242, 212, 28, "phantom: magnitude only", "e")],
             say="The engines come from other instruments. Harmonic is D.D.S.P. additive synthesis plus noise. Phantom "
                 "resynthesises magnitudes only: the network's own artefacts. Grains recall remembered moments by "
                 "Hopfield attention. Strings are sympathetic strings with a [jawari](/ʤəwˈɑːɹi/) bridge."),
        dict(shot=S, focus=(14, 56, 262, 250), point="partials, mosaic, voice, gendy ...",
             marks=[(57, 62, 212, 28, "click: choose the engine", "e")],
             say="Partials are inharmonic additive peaks. Mosaic is concatenative, by point, ring or path. Ear tones "
                 "make difference tones. Voice is a glottal source through formants. Gendy is Xenakis's dynamic "
                 "stochastic synthesis. Table is a wavetable from the decoded spectrum. And tape skips micro-loops, with dust."),
        dict(shot=S, focus=(14, 56, 262, 250), point="seven more, new in 1.1",
             marks=[(57, 62, 212, 28, "modal, membrane, fm, bowed ...", "e")],
             say="New in one point one, seven more, each a whole model. Modal: forty resonant modes, from a string to a "
                 "bell. Membrane: a struck drum skin, simulated in two dimensions. F.M.: four operators whose ratios "
                 "come from the decoded harmonics. Bowed: a waveguide string with bow friction and a body. Scanned: a "
                 "ring of masses pushed by the spectrum, read as a wavetable. Pulsar: trains of pulsarets at the "
                 "formants. And swarm: partials flocking to the spectrum's peaks."),
        dict(shot=Y, focus=(10, 40, 306, 362), point="layers: level, pan, character",
             marks=[(18, 70, 100, 26, "layer on", "e"), (18, 108, 292, 26, "level", "e"), (18, 148, 292, 26, "pan", "e"),
                    (18, 188, 292, 26, "character", "e")],
             say="Layers one to four, each with an on switch, level, pan and character."),
        dict(shot=Y, focus=(326, 40, 306, 400), point="dream and pitch",
             marks=[(334, 78, 292, 26, "lucidity", "e"), (334, 118, 292, 26, "temperature", "e"),
                    (334, 238, 292, 26, "morph rate", "e"), (430, 322, 196, 20, "pitch from input / network", "s"),
                    (334, 400, 292, 26, "pitch follow", "s")],
             say="Dream: lucidity, temperature, drift, probe gravity and the morph rate. Pitch: take it from the input "
                 "or from the network, transpose it, and set how closely it follows."),
        dict(shot=Y, focus=(643, 40, 306, 345), point="grains, mosaic, similarity",
             marks=[(651, 78, 292, 26, "grain size", "e"), (651, 118, 292, 26, "spray", "e"),
                    (748, 322, 196, 20, "similarity", "s"), (748, 354, 196, 20, "mosaic: point / ring / path", "s")],
             say="Memory and grains: grain size, spray, recall focus and ring size. Then what similarity means, latent, "
                 "timbre, chroma or loudness, and how the mosaic plays: point, ring or path."),
        dict(shot=Y, focus=(961, 40, 310, 735), point="nebula, input, output",
             marks=[(968, 78, 294, 26, "imprint", "w"), (968, 158, 294, 26, "width", "w"), (968, 238, 294, 26, "nebula", "w"),
                    (968, 490, 294, 26, "gate", "w"), (968, 610, 294, 26, "notes influence", "w")],
             say="Spectral, stargate and nebula: imprint, tilt, the stereo width and spin, and the nebula, a feedback "
                 "delay network reverb, with its decay. Input and output: input gain, gate, dry and wet, output level, "
                 "and how strongly the patch notes act on the sound."),
    ]),

    dict(title="gain staging", sub="new in 1.1: every stage metered, and the output can't clip", segs=[
        dict(shot=G, focus=None, point="the gain page",
             marks=[(320, 40, 640, 734, "the meters", "s")],
             say="New in one point one: the gain page. Left: the input and the layers. In the middle, the signal at every "
                 "stage. Right: the output and the limiter."),
        dict(shot=G, focus=(320, 40, 640, 734), point="a meter at every stage",
             marks=[(392, 84, 36, 540, "input", "w"), (458, 84, 236, 540, "the four layers", "n"),
                    (722, 84, 36, 540, "synthesis", "n"), (788, 84, 36, 540, "to the limiter", "n"), (852, 84, 36, 540, "output", "e")],
             say="Each stage has a peak meter from minus sixty to plus six decibels, with its peak held: the input after "
                 "its gain, each layer, the synthesis after the output gain, what reaches the limiter, and what leaves. "
                 "The dashed line is the ceiling."),
        dict(shot=G, focus=(320, 660, 640, 110), point="headroom, reduction, overs",
             marks=[(334, 676, 600, 18, "limiter, ceiling, look-ahead, gain reduction", "n"),
                    (334, 716, 600, 18, "overs: before the limiter / at the output", "s")],
             say="Below, the numbers: the limiter, the ceiling and its look-ahead, the gain reduction, the output peak and "
                 "its headroom, loudness in and out, and two counts of overs: samples that reached the limiter above the "
                 "ceiling, and samples that left above zero decibels. With the limiter on, that second count stays at zero. "
                 "Click to reset."),
        dict(shot=G, focus=(970, 40, 300, 280), point="ceiling, limiter, auto gain",
             marks=[(978, 160, 284, 30, "ceiling", "w"), (978, 210, 284, 26, "clean / soft", "w"),
                    (978, 242, 110, 28, "auto gain", "s")],
             say="The ceiling, minus one decibel by default. The limiter looks one point three milliseconds ahead, so "
                 "nothing passes the ceiling and the gain never steps: clean is transparent, and soft saturates into the "
                 "ceiling first. It can't be switched off. Auto gain brings the synthesis to the input's loudness over a "
                 "few seconds."),
        dict(shot=G, focus=(10, 40, 300, 320), point="match engine levels",
             marks=[(18, 74, 284, 32, "input gain", "e"), (18, 120, 284, 170, "layer levels", "e"),
                    (18, 304, 190, 28, "match engine levels", "e")],
             say="And every engine is now matched to the loudness the network decodes, so any engine sits where the input "
                 "does and follows its dynamics. The footer of every page shows the output peak, and whether anything "
                 "clipped."),
    ]),

    dict(title="patch notes", sub="what the notes say | the notes | journal", segs=[
        dict(shot=P, focus=(320, 40, 640, 130), point="words about the sound",
             marks=[(334, 74, 400, 30, "the notes", "s")],
             say="Patch notes are words you write about the sound. They are not just comments: they condition the network."),
        dict(shot=P, focus=(320, 690, 640, 90), point="a 32-d vector",
             marks=[(328, 752, 624, 18, "hashed character n-grams + 8 lexicon axes", "n")],
             say="The text becomes a thirty-two dimensional vector: hashed character n-grams, in the style of fast text, "
                 "through a random projection, plus eight lexicon axes."),
        dict(shot=P, focus=(10, 60, 300, 210), point="eight axes, with negation",
             marks=[(92, 74, 206, 16, "bright", "e"), (92, 98, 206, 16, "warm", "e"), (92, 146, 206, 16, "motion: negative", "e"),
                    (92, 242, 206, 16, "dream", "e")],
             say="The axes are bright, warm, dense, motion, rough, wide, wet and dream, and they understand negation. "
                 "A bar to the right is positive, a grey bar to the left negative: here, motion comes out negative."),
        dict(shot=P, focus=(10, 260, 300, 160), point="conditions, names, steers",
             marks=[(18, 284, 274, 46, "recognised words", "s"), (95, 376, 130, 30, "log entry", "s")],
             say="The words it recognised are listed below. The vector conditions the encoder and decoder through film "
                 "layers, names the stars on the constellation, and steers the sound directly."),
        dict(shot=P, focus=(640, 40, 640, 400), point="the journal",
             marks=[(974, 66, 292, 20, "journal", "s")],
             say="Log entry writes the notes into the journal, which is shared by every instance and the app."),
    ]),

    dict(title="corpus", sub="memory | files | offline training", segs=[
        dict(shot=C, focus=(320, 40, 640, 250), point="drop to feed, process, train",
             marks=[(330, 112, 620, 140, "drop zone", "s")],
             say="The corpus page is for files. Drop audio files or folders here. A plain drop feeds them: the network "
                 "learns from them and remembers them. Alt-drop processes them, rendering through the network into "
                 "the neurosonde folder in your music folder. Shift-drop trains offline."),
        dict(shot=C, focus=(320, 40, 640, 250), point="buttons and jobs",
             marks=[(330, 74, 149, 30, "feed", "n"), (487, 74, 149, 30, "process", "n"), (644, 74, 149, 30, "train offline", "n"),
                    (801, 74, 149, 30, "open output folder", "n"), (330, 258, 620, 18, "jobs", "s")],
             say="The buttons do the same: feed, process, train offline, and open the output folder. "
                 "Jobs are listed below with their progress."),
        dict(shot=C, focus=(10, 40, 300, 300), point="memory: how moments are recalled",
             marks=[(114, 72, 188, 20, "similarity", "e"), (114, 106, 188, 20, "playback", "e"), (18, 266, 284, 26, "recall focus", "e")],
             say="On the left, memory: how remembered moments are found and played back. Similarity, playback, "
                 "ring size, grain size, spray, recall focus and pitch follow."),
        dict(shot=C, focus=(966, 40, 304, 300), point="batch learning, 2 s blocks",
             marks=[(978, 78, 284, 26, "offline epochs (max)", "s"), (978, 120, 284, 14, "epoch, train, validation, best", "s")],
             say="Offline training is batch learning. The shared brain is frozen while playing. Standardisation comes "
                 "from the whole set, and it is split into training and validation by contiguous two-second blocks, "
                 "so near-identical neighbouring frames never straddle the split."),
        dict(shot=C, focus=(966, 130, 304, 645), point="early stopping, best epoch",
             marks=[(1040, 536, 184, 62, "train, solid", "n"), (1040, 606, 184, 66, "validation, dashed", "s"),
                    (1220, 140, 14, 620, "best epoch", "w")],
             say="It trains in shuffled minibatch epochs and validates after each. The solid line is training loss, the "
                 "dashed line validation. After five epochs without improvement it stops early, restores the best "
                 "epoch's weights, and refits the constellation. Then switch back to online, or stay frozen."),
    ]),

    dict(title="lab", sub="the training, measured: objective, optimiser, layers", segs=[
        dict(shot=L, focus=None, point="seven panels",
             marks=[(10, 40, 306, 362, "objective", "s"), (10, 412, 306, 363, "optimiser", "n"),
                    (322, 40, 636, 258, "layers", "s"), (322, 308, 636, 204, "inside the network", "s"),
                    (322, 522, 636, 253, "training log", "n"), (966, 40, 304, 362, "latent to synthesis", "s"),
                    (966, 412, 304, 363, "held-out frame", "n")],
             say="The lab shows the training itself, measured from the real networks during the real optimisation. "
                 "Seven panels."),
        dict(shot=L, focus=(10, 40, 306, 362), point="objective, term by term",
             marks=[(16, 66, 294, 74, "the loss, written out", "e"), (16, 146, 294, 74, "value, weight, contribution, share", "e"),
                    (18, 224, 284, 158, "per step faint, ema bright", "e")],
             say="Objective: the loss, term by term. Reconstruction, plus a quarter of the harmonic cross-entropy, "
                 "plus the noise bands, plus beta times the K.L. divergence. Each term's value, weight, contribution "
                 "and share, and log-scale curves of the last six hundred steps."),
        dict(shot=L, focus=(10, 412, 306, 363), point="optimiser: adamw, clipping",
             marks=[(16, 440, 294, 36, "adamw, clip, learning rate, beta", "e"), (16, 480, 294, 122, "step, gradient, batch", "e"),
                    (18, 604, 284, 68, "gradient norm vs clip", "e"), (18, 684, 284, 68, "learning rate", "e")],
             say="Optimiser: Adam W, with decoupled weight decay and global-norm clipping at one. The learning rate "
                 "decays toward a floor, scaled by plasticity, and beta warms up over the first three thousand steps. "
                 "Below: Adam's step, the gradient norm before clipping, steps per second, and the batch: sixteen "
                 "recent frames and sixteen from replay when live, all replay offline."),
        dict(shot=L, focus=(322, 40, 636, 258), point="layers: one row per layer",
             marks=[(328, 88, 50, 176, "enc to gru", "s"), (666, 88, 76, 176, "update ratio, log", "n"),
                    (848, 88, 104, 176, "weight and gradient histograms", "s")],
             say="Layers: one row per layer of the V.A.E. and the G.R.U. Shape, activation, parameter count, the size "
                 "of the weights and gradients, and the update ratio, the change in the weights over their size, on a "
                 "log bar with a tick at one in a thousand, the usual healthy rate. Then activation statistics, the "
                 "share of leaky ReLU outputs below zero, and histograms of the weights and gradients."),
    ]),

    dict(title="inside the lab", sub="inside the network, the training log, latent to synthesis, the held-out frame", segs=[
        dict(shot=L, focus=(322, 308, 636, 204), point="weights as pixels, activations",
             marks=[(330, 350, 94, 142, "enc.1", "s"), (435, 350, 142, 142, "dec.2", "s"), (589, 350, 112, 142, "dec.out", "s"),
                    (714, 346, 238, 146, "batch activations", "s")],
             say="Inside the network: three weight matrices as pixels. White is positive, grey negative, brightness the "
                 "size. Beside them, the activations of the last batch, thirty-two frames by units, layer by layer."),
        dict(shot=L, focus=(322, 522, 636, 253), point="the training log",
             marks=[(328, 544, 40, 196, "prefixes", "n"), (328, 734, 500, 12, "== offline epochs and early stopping", "s")],
             say="The training log: one line per event, newest at the bottom. V.A.E. and offline optimiser steps with "
                 "every loss term, G.R.U. steps, held-out evaluations every two hundred steps, and the offline epochs. "
                 "The mouse wheel scrolls back; a double-click returns to live."),
        dict(shot=L, focus=(966, 40, 304, 362), point="latent to synthesis",
             marks=[(976, 70, 288, 22, "how z was made", "s"), (978, 150, 284, 38, "z, with z in and z dream", "s"),
                    (978, 270, 284, 30, "harmonic distribution", "w"), (978, 312, 284, 82, "noise bands", "w")],
             say="Latent to synthesis: how z was made. The input's latent and the dream, mixed by lucidity, pulled "
                 "toward the probe by gravity, then nudged by drift. Then what the decoder sends to the engines: the "
                 "harmonic distribution, the noise bands, pitch and where it comes from, voicing, loudness, centroid "
                 "and flatness. Until step two hundred the engines follow the analysis; after that, the network drives them."),
        dict(shot=L, focus=(966, 412, 304, 363), point="a frame never trained on",
             marks=[(978, 460, 284, 58, "mel: x solid, x-hat dashed", "w"), (978, 530, 284, 40, "harmonics", "w"),
                    (978, 686, 284, 38, "latent mu plus or minus sigma, kl", "w"), (978, 742, 284, 28, "error per output", "w")],
             say="Held-out frame: a frame never trained on, through the encoder's mean and the decoder. The mel bands, "
                 "harmonics and noise, target against prediction; its latent; the K.L. per dimension; and the error per "
                 "output: which parts of the sound it has learned."),
    ]),

    dict(title="space", sub="neurosonde space: the same instrument, rendered into space", segs=[
        dict(shot=SP, focus=None, point="the same brain, in space",
             marks=[(10, 40, 340, 410, "scene", "s"), (360, 40, 560, 735, "field", "s"), (930, 40, 340, 410, "spatial network", "s"),
                    (930, 460, 340, 315, "input image", "n")],
             say="Neurosonde Space is the same instrument rendered into space, sharing the same brain."),
        dict(shot=SP, focus=(930, 40, 340, 410), point="24 stereo features",
             marks=[(938, 120, 326, 108, "level difference, 8 bands", "w"), (938, 228, 326, 106, "coherence, 8 bands", "w"),
                    (938, 334, 326, 108, "width, azimuth, diffuseness, motion", "w")],
             say="It listens to the stereo image: twenty-four features every ten milliseconds. Level difference and "
                 "coherence in eight bands, then width, correlation, balance, azimuth, diffuseness and motion."),
        dict(shot=SP, focus=(930, 40, 340, 410), point="the spatial network, beta-nll",
             marks=[(938, 70, 326, 44, "steps, nll, held out", "s"), (1018, 120, 246, 320, "thin measured, thick predicted", "w")],
             say="A spatial network learns to predict that image from the latent and the notes, as a Gaussian with its "
                 "own uncertainty, trained with beta N.L.L. The thin marks are measured; the thick ones are predicted, "
                 "plus or minus sigma."),
        dict(shot=SP, focus=(360, 40, 920, 735), point="the field, the input image",
             marks=[(408, 180, 464, 470, "the field: drag to turn", "e"), (976, 486, 248, 256, "goniometer", "w")],
             say="The field shows the result as a sphere you can turn: the sources, their stereo pairs, the reflections, "
                 "and the intensity vector. Beside it, the input's goniometer and correlation."),
        dict(shot=SP, focus=(10, 40, 340, 410), point="scene: ambisonics to binaural",
             marks=[(114, 72, 228, 20, "output", "e"), (114, 102, 228, 20, "order 1 to 7", "e"),
                    (18, 140, 324, 26, "follow the network", "e")],
             say="Scene: the output, Ambisonics up to seventh order, binaural, stereo, quad, five point one or "
                 "seven one four. How far to follow the network, the azimuth and elevation spans, layer spread, orbit, "
                 "distance, yaw and tilt."),
        dict(shot=SP, focus=(10, 450, 340, 330), point="room, binaural head",
             marks=[(18, 496, 324, 26, "room size", "e"), (18, 534, 324, 26, "early reflections", "e"),
                    (18, 572, 324, 26, "late field", "e"), (18, 610, 324, 26, "room decay", "e")],
             say="Room: its size, early reflections from an image-source shoebox, the late field, and its decay. "
                 "Binaural output goes through a structural head model, after Brown and Duda."),
    ]),

    dict(title="learning and sharing", sub="online or offline, one shared brain, memories", segs=[
        dict(shot=N, focus=(10, 40, 300, 100), point="online by default",
             marks=[(114, 72, 188, 20, "online / offline", "s")],
             say="To close: how it learns. Online is the default: it learns continuously from what it hears. In a D.A.W. "
                 "bounce it waits for the brain to take every frame, so a bounce teaches it what playing live would."),
        dict(shot=S, focus=(1004, 392, 262, 150), point="held out, replay, sleep",
             marks=[(1010, 480, 250, 14, "replay reservoir", "w"), (1010, 494, 250, 14, "sleeping / replay", "w")],
             say="A replay reservoir guards against forgetting, and when it is quiet, it sleeps: it consolidates "
                 "from replay."),
        dict(shot=S, focus=(1004, 392, 262, 150), point="one brain, merged",
             marks=[(1010, 508, 250, 14, "merges", "w")],
             say="The brain is shared. Every instance in a host uses the same one; other processes merge through a "
                 "brain file, by federated delta averaging under a file lock. Merges counts how often."),
        dict(shot=N, focus=(966, 40, 304, 740), point="memories, to go back to",
             marks=[(976, 64, 290, 30, "named snapshots", "s"), (978, 740, 89, 28, "remember", "n")],
             say="And memories are named snapshots of the brain to go back to, the way Empty Set built tracks from "
                 "earlier, rougher stages."),
        dict(shot=S, focus=None, point="neurosonde", marks=[],
             say="That is neurosonde, version one point one, for the Mac, Windows and Linux: a network that learns from "
                 "what it hears, and plays it back."),
    ]),
]
