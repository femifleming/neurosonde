# Research notes (2026-11-24)

## Emptyset — Blossoms (Thrill Jockey THRILL-507, 2019)
Generated entirely from a neural network trained on their own catalogue + ~10 h of new improvisations on wood, metal, skins (1–5 s slices, tagged). ~18 months development with an unnamed international network of programmers; architecture not published (raw-audio, sample-level). Narrow data "resolved to an average"; they preferred the "rougher, semi-congealed" earlier stages ("uncanny valley"), building tracks from earlier-stage outputs: "we were more engaged by the system's failure to mimic our work than its success."
Sources: thrilljockey.com/products/blossoms · thequietus.com/?p=27287 · metalmagazine.eu/post/emptyset · whitefungus.com/emptyset-uncanny-processes · clotmag.com/?p=20841

## Dillon Bastan — Coalescence (M4L)
Concatenative multi-sampler; slices (transients or frames) analysed by Chroma / Mel / Bark / MFCC, organised by a Kohonen SOM into a 2-D circle; Point / Rings / Paths playback; sidechain input as trigger and as similarity cursor ("pseudo style transfer"); phase-vocoder voice with spectral attack/release; LFOs (Perlin), envelopes, MPE.
Source: dillonbastan.com/store/maxforlive/index.php?product=coalescence

## Methods used
Conditional VAE (Kingma & Welling 1312.6114) with FiLM (Perez 1709.07871), DDSP-style harmonic+noise synthesis (Engel 2001.04643) trained by regression on analysed targets, GRU latent predictor (Cho 1406.1078), echo state network + RLS readout (Jaeger 2001), Kohonen SOM, modern Hopfield retrieval (Ramsauer 2008.02217) for corpus recall, reservoir-sampling replay (Vitter 1985), Adam (1412.6980) with lr floor (loss of plasticity, Dohare 2024), global-norm clipping, running standardisation, federated delta averaging between instances (McMahan 1602.05629), fastText-style hashed n-grams (1607.04606) + hashing trick (0902.2206) for text; MIDI: Continuator PST (Pachet 2003), Factor Oracle / OMax (Assayag & Dubnov 2004), GRU.

## EVE Online Photon UI (reference images: reference/eve/, private, © CCP / EVE Uni wiki)
Palette (measured): bg #090E0E–#0B1213 (header), body #161A1B @85–90%, raised #202726, hover #192A2C, selected #222F31,
divider #293030, accent cyan #58A7BF, active top line #4E91A5, bar #54A4D6, probe sphere #568EA2 (lines ~30%),
d-scan teal #3E7E70, text #E0E2E2 / #B0B0B0 / #8A8A8A, amber #EB953E, yellow #FAE400, good #66B27A, danger #AB2929,
stars #DDA276→#714218 + white. Scan strength: red <25, orange 25–75, yellow 75–99, green 100.
Rules: square windows, 1px #293030 edge, active window 1–2px #4E91A5 top line + glow; header 28px #0B1213; text tabs with
2px underline; buttons 1px accent border @70% + 15% fill, bottom-right corner cut 45°; rows 20px, hover/selected fills,
2px left bar; HUD-style segmented rings; thin tracks with 10% ticks; wireframe probe spheres (additive); constellation:
2–4px glowing nodes, 1px links @35%, spaced-caps region labels, bracket corners on selection; tooltips #0B1213 @95%.
Fonts: EVE Sans Neue is proprietary → Saira Semi Condensed / Titillium Web + Share Tech Mono (OFL).

## EVE Online in-game rendering (reference/eve-render/, private; palettes sampled from the frames)
Nebulas are painted cube maps, one per region; highsec warmer/saturated, low/null cooler with more stars (CCP 2011);
2021 "Bring Out EVE's Colors": darker space, smoother gradients; 2022: doubled cube-map resolution, gas clouds back.
Signature: diffuse domain-warped cloud + DARK DUST LANES (ridged noise) + thin emission filaments (rust #c0602f),
nebula over 40–60 % of the frame, dense power-law star field (1 star / 400 px², 1 % bright with halo).
Palettes: amber #282218 #3b3221 #605942 #878570 (#c4b38f); teal #0e1211 #252e29 #3c4c3f #708873 (#afc7b3);
null #121217 #262627 #3e3c3b #6d7377; red/mauve #2b2126 #604649 #8a524c #9d7d6b; blue-grey #0b0f11 #2c363f #576b78 #8c9ba6.
Suns: white core + tinted glow (blue #3463ae, red #ab6461, orange #b06d39, deep red #d7542f) + horizontal anamorphic
streak (30–60 % of width) + 6–12 rays + ghosts along the line through screen centre (CCP "Solar Flares", 2014).
Trails: tapering cyan ribbons (halo #3a7fb0 10 px, body #56bef0 3 px, white core). Warp: 40–80 radial streaks #bfe3ff.
Probe spheres: lattice #426d6d–#47757e at 0.3–0.5, rim brighter, additive overlaps. Rings: #9fb6be 0.15–0.35,
ticks every 10°/30°, stalks to the plane, far half faded. Wormholes: dark sphere, red-orange core, 30–60 swept
filaments (golden #bc8957 / teal / blue by mass), dying = darker + wobblier (CCP 2019). Post: vignette 25–35 %,
grain ±2–3 levels, no chromatic aberration.

## spatial (neurosonde space)
- Seitzer, Tavakoli, Antic, Martius (2022), "On the pitfalls of heteroscedastic uncertainty estimation with probabilistic
  neural networks" (ICLR): β-NLL. With plain Gaussian NLL, the spatial net's variance collapsed on deterministic
  features (e.g. the coherence of a panned tone), and its mean predictions stalled at the class mean. β = 0.5 fixed it.
- Brown & Duda (1998), "A structural model for binaural sound synthesis": head shadow H(s) = (αs + β)/(s + β),
  β = 2c/a, α(θ) = 1.05 + 0.95 cos(θ·180°/150°); Woodworth ITD.
- Zotter & Frank (2012), "All-round ambisonic panning and decoding": max-rE weights g_n = P_n(cos(137.9°/(N + 1.51))).
- Allen & Berkley (1979), image-source method for small rooms (first-order images only here).
- Pulkki (2007), DirAC: the intensity vector from first-order B-format gives direction and diffuseness for the meters.
