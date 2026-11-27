// Neurosonde — one instance: analysis → brain (learning) → inference (encode / dream / probe) → decode → four engine
// layers → stargate (stereo field) → nebula (FDN reverb) → output. JUCE-free so it can be tested and run offline.
#pragma once
#include "ml/Brain.h"
#include "dsp/Synths.h"
#include "dsp/Stereo.h"
#include "dsp/Spatial.h"
#include <mutex>

namespace ns
{
using namespace ns::ml;
using namespace ns::dsp;

struct MidiEv { int sample = 0; int type = 0; int note = 0; int vel = 0; float bend = 0; };   // type 0 off, 1 on, 2 bend

struct Params
{
    struct L { bool on = false; int mode = 0; float level = 1, pan = 0, character = 0.3f; };
    std::array<L, Synth::kLayers> layer {};
    int combine = 1;            // stack / morph / merge / imprint
    float morph = 0.0f; int morphSource = 0;  // 0 manual, 1 network (latent), 2 constellation x, 3 lfo
    float morphRate = 0.05f;
    float lucid = 0.7f, temperature = 0.5f, drift = 0.1f; int dreamEngine = 0;   // 0 gru, 1 reservoir, 2 both
    float gravity = 0.0f, probeX = 0.5f, probeY = 0.5f; bool route = false; float routeSpeed = 4.0f;  // stars / minute
    bool learn = true, listen = true; float gateDb = -58, inGainDb = 0;
    float mix = 1.0f, outDb = 0, transpose = 0; int pitchSource = 0; bool keysOnly = false;    // pitch: 0 input, 1 network, 2 keys
    float attack = 0.02f, release = 0.6f, glide = 0.05f;
    float grainSize = 120, spray = 0.3f, focus = 6; int space = 0, playback = 2; float ringSize = 6, pitchFollow = 0.5f;
    float imprint = 0.0f, tilt = 0.0f, notesInfluence = 0.6f;
    float width = 0.7f, spin = 0.0f, nebula = 0.25f, decay = 0.6f;
    float learnedWidth = 0.0f;      // stereo build: lets the spatial network's predicted width open the stereo field
    // spatial build (Neurosonde Space)
    bool spatial = false; int order = 3; int format = 1;   // OutFormat
    float follow = 0.7f, span = 0.75f, elevSpan = 0.5f, orbit = 0.0f, distance = 2.0f, layerSpread = 0.35f;
    float room = 1.0f, early = 0.35f, late = 0.3f, roomDecay = 0.6f, yaw = 0.0f, sceneTilt = 0.0f;
};

/** what the UI reads (copied under a tiny lock, once per hop) */
struct Telemetry
{
    float z[kZ] {}, zIn[kZ] {}, zDream[kZ] {};
    int bmu = -1, inputBmu = -1;
    float inLoud = -100, outLoud = -100, f0 = 0, voicing = 0, presence = 0, surprise = 0;
    float mel[kMel] {}, melIn[kMel] {}, chroma[kChroma] {};
    // the stereo image: measured from the input, and what the spatial network predicts for the current latent
    float spatIn[kSpatial] {}, spatPred[kSpatial] {}, spatSd[kSpatial] {}; bool stereoInput = false; uint64_t spatSteps = 0;
    struct Src { float az = 0, el = 0, width = 0, dist = 1, weight = 0; };
    std::array<Src, 5> sources {}; std::array<std::pair<float, float>, 6> reflections {}; int nRefl = 0;
    float fieldAz = 0, fieldEl = 0, fieldDiffuse = 1;
    std::array<float, 256> scopeL {}, scopeR {};
    float weights[Synth::kLayers] {};
    struct Hit { int unit; float strength; float loud; float f0; };
    std::array<Hit, 8> hits {}; int nHits = 0;
    int corpusUnits = 0; float routePos = 0;
    uint64_t hop = 0;
    // the path from the latent into the synthesis (lab page): how z was made, and what the decoder told the engines
    float zPull[kZ] {}, zDrift[kZ] {};                 // the probe/route pull and the drift added to z
    float lucidMix = 0, gravMix = 0, driftMix = 0; bool havePull = false, netDrives = false;
    float synHarm[kHarm] {}, synNoise[kNoise] {};      // harmonic distribution (normalised) and noise bands (dB) sent to the engines
    float netF0 = 0, synF0 = 0, synVoicing = 0, synLoud = 0, synCentroid = 0, synFlat = 0, synFlux = 0; int pitchFrom = 0;
    uint64_t netSteps = 0;
};

/** stereo field: width (mid/side) and a slow rotation ("stargate") */
struct Stargate
{
    float ph = 0, sr = 48000;
    void process (float* L, float* R, int n, float width, float spin)
    {
        for (int i = 0; i < n; ++i)
        {
            const float m = 0.5f * (L[i] + R[i]), s = 0.5f * (L[i] - R[i]) * width * 1.6f;
            float l = m + s, r = m - s;
            if (spin > 0.001f)
            {
                ph += spin * 0.5f / sr; if (ph > 1) ph -= 1;
                const float a = 0.5f * (1 + std::sin (kTwoPi * ph)) * spin;
                const float c = std::cos (a * 0.7f), sn = std::sin (a * 0.7f);
                const float nl = c * l - sn * r, nr = sn * l + c * r; l = nl; r = nr;
            }
            L[i] = l; R[i] = r;
        }
    }
};

/** 8-line feedback delay network with a Householder matrix and slow modulation ("nebula") */
struct Nebula
{
    static constexpr int D = 8;
    std::array<std::vector<float>, D> line; std::array<int, D> len {}, w {}; std::array<float, D> lp {}; float sr = 48000, ph = 0;
    void prepare (float s)
    {
        sr = s; const int base[D] = { 1031, 1327, 1523, 1801, 2063, 2371, 2647, 3011 };
        for (int i = 0; i < D; ++i) { len[(size_t) i] = (int) (base[i] * s / 48000.0f * 1.7f); line[(size_t) i].assign ((size_t) len[(size_t) i] + 64, 0); w[(size_t) i] = 0; lp[(size_t) i] = 0; }
    }
    void process (float* L, float* R, int n, float mix, float decay)
    {
        if (mix <= 0.001f) return;
        const float g = 0.70f + 0.29f * std::clamp (decay, 0.0f, 1.0f);
        for (int i = 0; i < n; ++i)
        {
            ph += 0.13f / sr; if (ph > 1) ph -= 1;
            float o[D]; float sum = 0;
            for (int k = 0; k < D; ++k)
            {
                const float mod = 12 * std::sin (kTwoPi * (ph + k * 0.125f));
                const float rp = (float) w[(size_t) k] - (float) len[(size_t) k] + mod;
                const int sz = (int) line[(size_t) k].size();
                float pos = std::fmod (rp + (float) sz * 4, (float) sz); const int a = (int) pos; const float f = pos - a;
                const float v = line[(size_t) k][(size_t) a] * (1 - f) + line[(size_t) k][(size_t) ((a + 1) % sz)] * f;
                lp[(size_t) k] += (v - lp[(size_t) k]) * 0.35f;
                o[k] = lp[(size_t) k]; sum += o[k];
            }
            const float h = sum * (2.0f / D);
            const float inL = L[i], inR = R[i];
            for (int k = 0; k < D; ++k)
            {
                const float x = (o[k] - h) * g + (k % 2 ? inR : inL) * 0.35f;
                line[(size_t) k][(size_t) w[(size_t) k]] = x;
                w[(size_t) k] = (w[(size_t) k] + 1) % (int) line[(size_t) k].size();
            }
            const float wl = o[0] + o[2] + o[4] + o[6], wr = o[1] + o[3] + o[5] + o[7];
            L[i] = inL * (1 - mix * 0.5f) + wl * mix * 0.45f;
            R[i] = inR * (1 - mix * 0.5f) + wr * mix * 0.45f;
        }
    }
};

class Instance
{
public:
    explicit Instance (std::shared_ptr<Brain> b) : brain (std::move (b)) { ring = brain->addFeeder(); setNotes (""); }
    ~Instance() { brain->removeFeeder (ring); }

    void prepare (double sampleRate, int maxBlock)
    {
        sinceHop = 0;   // the hop size depends on the rate: a stale counter would make negative segments
        sr = (float) sampleRate;
        analyser.prepare (sampleRate);
        corpus.prepare (sr);
        synth.prepare (sr, std::max (maxBlock, 2048), &corpus);
        synth.phantom.carrier = carrierWin.data();
        nebula.prepare (sr); gate.sr = sr;
        stereo.prepare (sampleRate); spatNow.fill (0);
        scene.prepare (sr, std::max (maxBlock, 2048));
        for (auto& b : layL) b.assign ((size_t) std::max (maxBlock, 2048), 0.0f);
        for (auto& b : layR) b.assign ((size_t) std::max (maxBlock, 2048), 0.0f);
        dryL.assign ((size_t) std::max (maxBlock, 2048), 0.0f); dryR = dryL;
        wetL.assign ((size_t) std::max (maxBlock, 2048), 0); wetR = wetL; mono = wetL;
        std::fill (gruH, gruH + kGruH, 0.0f); esnS.assign (EchoState::N, 0.0f);
        std::fill (zPrev, zPrev + kZ, 0.0f); std::fill (driftV, driftV + kZ, 0.0f);
        stargate.sr = sr; hopCount = 0; presence = 0; held.reserve (128);
        synth.setControl (ctl);
    }

    /** patch notes → conditioning vector, lexicon axes, words for star names */
    void setNotes (const std::string& text)
    {
        std::vector<std::string> words;
        auto c = TextEmbed::embed (text, &words);
        std::lock_guard<std::mutex> l (notesLock);
        pendingCond = c; notesChanged = true;
        noteWords = words;
        brain->setNoteWords (words);
    }
    std::vector<std::string> recognisedWords() { std::lock_guard<std::mutex> l (notesLock); return noteWords; }

    /** audio + units analysed off the audio thread (files): the audio thread copies one chunk per block into the
        corpus. Unit starts are relative to the chunk. */
    struct Chunk { std::vector<float> audio; std::vector<Unit> units; };
    bool offerChunk (Chunk* c) { return inbox.push (c); }
    Chunk* takeDone() { Chunk* c = nullptr; return done.pop (c) ? c : nullptr; }

    void setRoute (const std::vector<int>& stars) { std::lock_guard<std::mutex> l (routeLock); pendingRoute = stars; routeChanged = true; }

    Params params;
    /** Neurosonde Space: where process() writes the spatial output (nOut channels); nullptr = stereo */
    void setSpatialOutput (float* const* outs, int nOut) { spatialOut = outs; spatialN = nOut; }

    /** anaL/anaR: an optional separate signal to listen to (a sidechain); the dry mix always uses inL/inR */
    void process (const float* inL, const float* inR, float* outL, float* outR, int n, const std::vector<MidiEv>& midi = {},
                  const float* anaL = nullptr, const float* anaR = nullptr)
    {
        if (notesChanged.load()) { std::unique_lock<std::mutex> l (notesLock, std::try_to_lock); if (l.owns_lock()) { cond = pendingCond; notesChanged = false; } }
        if (routeChanged.load()) { std::unique_lock<std::mutex> l (routeLock, std::try_to_lock); if (l.owns_lock()) { route = pendingRoute; routeChanged = false; } }
        for (int l = 0; l < Synth::kLayers; ++l)
        {
            auto& L = synth.layers[(size_t) l]; const auto& P = params.layer[(size_t) l];
            if (L.mode != P.mode && P.on) { /* engines keep their state; nothing to reset */ }
            L.on = P.on; L.mode = std::clamp (P.mode, 0, (int) Mode::Count - 1); L.level = P.level; L.pan = P.pan; L.character = P.character;
        }
        applyEngineParams();
        if (Chunk* ch = nullptr; inbox.pop (ch) && ch)
        {
            const int64_t base = corpus.written;
            corpus.write (ch->audio.data(), (int) ch->audio.size());
            for (auto u : ch->units) { u.start += base; corpus.add (u); }
            done.push (ch);   // returned to the feeder for deletion (no free on the audio thread)
        }
        const float inGain = dbToGain (params.inGainDb);
        size_t mi = 0;
        int i = 0;
        while (i < n)
        {
            if (sinceHop >= analyser.hopSize()) sinceHop = 0;
            const int seg = std::min (n - i, analyser.hopSize() - sinceHop);
            // MIDI events inside this segment
            while (mi < midi.size() && midi[mi].sample < i + seg) handleMidi (midi[mi++]);
            const float* aL = anaL ? anaL : inL; const float* aR = anaR ? anaR : (anaL ? anaL : inR);
            for (int k = 0; k < seg; ++k) mono[(size_t) k] = 0.5f * (aL[i + k] + aR[i + k]) * inGain;
            if (params.listen) corpus.write (mono.data(), seg);
            // a stereo image to learn from: two different channels (a bit-identical pair is a mono source duplicated)
            stereoIn = aL != aR && ! std::equal (aL + i, aL + i + seg, aR + i);
            stereo.push (aL + i, aR + i, seg, [this] (const std::array<float, kSpatial>& f) { spatNow = f; });
            for (int k = 0; k < seg; k += 4) { scope[(size_t) scopePos].first = aL[i + k]; scope[(size_t) scopePos].second = aR[i + k]; scopePos = (scopePos + 1) % 256; }
            analyser.push (mono.data(), seg, [this] (const RawFrame& f) { hop (f); });
            sinceHop = (sinceHop + seg) % analyser.hopSize();
            if (params.spatial && spatialOut)
            {
                renderSpatial (inL + i, inR + i, i, seg);
                i += seg;
                continue;
            }
            std::fill (wetL.begin(), wetL.begin() + seg, 0.0f); std::fill (wetR.begin(), wetR.begin() + seg, 0.0f);
            synth.render (wetL.data(), wetR.data(), seg, (Combine) std::clamp (params.combine, 0, 3), morphNow);
            // gate envelope (keys)
            for (int k = 0; k < seg; ++k) { const float g = gate.next(); if (params.keysOnly) { wetL[(size_t) k] *= g; wetR[(size_t) k] *= g; } }
            const float learnedW = params.learnedWidth * juce_clamp01 (spatPredNow[16] * 2.5f);
            stargate.process (wetL.data(), wetR.data(), seg, params.width * (1 + 0.5f * axis (TextEmbed::Wide)) * (1 + learnedW), params.spin);
            nebula.process (wetL.data(), wetR.data(), seg, std::clamp (params.nebula + 0.4f * std::max (0.0f, axis (TextEmbed::Wet)), 0.0f, 1.0f), params.decay);
            const float og = dbToGain (params.outDb), mx = params.mix;
            double e = 0;
            for (int k = 0; k < seg; ++k)
            {
                float l = wetL[(size_t) k] * og, r = wetR[(size_t) k] * og;
                dcL += 0.0005f * (l - dcL); dcR += 0.0005f * (r - dcR); l -= dcL; r -= dcR;
                l = std::tanh (l * 0.9f) / 0.9f; r = std::tanh (r * 0.9f) / 0.9f;
                if (! std::isfinite (l)) l = 0; if (! std::isfinite (r)) r = 0;
                outL[i + k] = inL[i + k] * (1 - mx) + l * mx;
                outR[i + k] = inR[i + k] * (1 - mx) + r * mx;
                e += l * l + r * r;
            }
            outLoud = 0.9f * outLoud + 0.1f * (float) (10 * std::log10 (e / (2 * seg) + 1e-12));
            i += seg;
        }
        while (mi < midi.size()) handleMidi (midi[mi++]);
    }

    /** copy of the telemetry for the UI */
    Telemetry telemetry() { std::lock_guard<std::mutex> l (telLock); return tel; }
    const Corpus& getCorpus() const { return corpus; }
    std::shared_ptr<Brain> getBrain() const { return brain; }
    float sampleRate() const { return sr; }

private:
    float axis (int a) const { return cond[(size_t) (TextEmbed::kHash + a)] * params.notesInfluence; }
    static float juce_clamp01 (float x) { return std::clamp (x, 0.0f, 1.0f); }

    /** rotate a direction by the scene's yaw (about z) and tilt (about the left-right axis) */
    void rotateDir (float& az, float& el) const
    {
        float x = std::cos (el) * std::cos (az), y = std::cos (el) * std::sin (az), zz = std::sin (el);
        const float ct = std::cos (params.sceneTilt), st = std::sin (params.sceneTilt);
        const float x2 = x * ct - zz * st, z2 = x * st + zz * ct; x = x2; zz = z2;
        az = std::atan2 (y, x) + params.yaw; el = std::atan2 (zz, std::sqrt (x * x + y * y));
    }

    /** where each layer sits: the constellation gives the base direction, the spatial network's prediction (how the
        input's stereo image went with this timbre) gives balance, width, diffuseness and motion — blended by `follow` */
    void placeSources (float dt)
    {
        const float f = spatTrained ? params.follow : 0.0f;
        const float predAz = std::clamp (spatPredNow[19], -1.0f, 1.0f);               // −1 right .. +1 left
        const float predW = std::clamp (spatPredNow[16], 0.0f, 1.0f);                  // side / (mid + side)
        const float predDiff = std::clamp (spatPredNow[20], 0.0f, 1.0f);
        const float predMotion = std::clamp (spatPredNow[21], 0.0f, 1.0f);
        const float sdAz = std::clamp (spatSdNow[19], 0.0f, 1.0f);
        orbitPhase += (params.orbit / 60.0f + f * predMotion * 0.15f) * dt * 2 * (float) M_PI;
        const float baseAz = (0.5f - mapU) * 2 * (float) M_PI * params.span + f * predAz * (float) M_PI * 0.5f + orbitPhase;
        const float baseEl = (0.5f - mapV) * (float) M_PI * 0.5f * params.elevSpan;
        for (int k = 0; k < Synth::kLayers; ++k)
        {
            auto& c = srcCtl[(size_t) k];
            float az = baseAz + (k - 1.5f) * params.layerSpread * (float) M_PI * 0.5f + 0.25f * zNow[std::min (kZ - 1, 5 + k)] * params.layerSpread;
            float el = std::clamp (baseEl + 0.12f * zNow[std::min (kZ - 1, 9 + k)] * params.elevSpan, -1.4f, 1.4f);
            rotateDir (az, el);
            const float tAz = az, tEl = el;
            // smooth toward the target (no zipper), the uncertain the slower
            const float k1 = std::clamp (dt * (8.0f - 6.0f * sdAz), 0.01f, 1.0f);
            float d = std::remainder (tAz - c.az, 2 * (float) M_PI);
            c.az += k1 * d; c.el += k1 * (tEl - c.el);
            c.width = std::clamp (0.25f + f * predW * 2.6f + (1 - f) * 0.5f, 0.0f, (float) M_PI);
            c.diffuse = std::clamp (f * predDiff * 0.7f, 0.0f, 0.9f);
            c.dist = params.distance * (1 + f * predDiff * 1.5f);
            c.send = 0.2f + 0.6f * (f * predDiff + (1 - f) * 0.4f);
        }
        auto& dry = srcCtl[4]; dry.az = 0; dry.el = 0; dry.width = (float) M_PI / 3; dry.dist = 1.5f; dry.diffuse = 0; dry.send = 0.15f;
        rotateDir (dry.az, dry.el);
    }

    void renderSpatial (const float* inL, const float* inR, int offset, int seg)
    {
        scene.setOrder (params.order, (OutFormat) std::clamp (params.format, 0, 5));
        scene.roomSize = params.room; scene.early = params.early; scene.late = params.late; scene.decay = params.roomDecay;
        placeSources ((float) seg / sr);
        float* Ls[4]; float* Rs[4]; for (int k = 0; k < 4; ++k) { Ls[k] = layL[(size_t) k].data(); Rs[k] = layR[(size_t) k].data(); }
        synth.renderLayers (Ls, Rs, seg, (Combine) std::clamp (params.combine, 0, 3), morphNow);
        const float og = dbToGain (params.outDb) * params.mix;
        for (int k = 0; k < 4; ++k) for (int s2 = 0; s2 < seg; ++s2) { Ls[k][s2] *= og; Rs[k][s2] *= og; if (params.keysOnly) { const float g = gate.next(); Ls[k][s2] *= g; Rs[k][s2] *= g; } }
        for (int s2 = 0; s2 < seg; ++s2) { dryL[(size_t) s2] = inL[s2] * (1 - params.mix); dryR[(size_t) s2] = inR[s2] * (1 - params.mix); }
        std::array<const float*, SpatialScene::kSources> l {}, r {};
        for (int k = 0; k < 4; ++k) { const bool live = synth.layerLive (k) || (k == 0 && (params.combine >= 2)); l[(size_t) k] = live ? Ls[k] : nullptr; r[(size_t) k] = live ? Rs[k] : nullptr; }
        l[4] = params.mix < 0.999f ? dryL.data() : nullptr; r[4] = params.mix < 0.999f ? dryR.data() : nullptr;
        float* outs[ambi::kCh]; const int nOut = std::min (spatialN, ambi::kCh);
        for (int c = 0; c < nOut; ++c) outs[c] = spatialOut[c] + offset;
        scene.process (l, r, srcCtl, outs, nOut, seg);
        // safety: soft-limit every output channel
        for (int c = 0; c < nOut; ++c) for (int s2 = 0; s2 < seg; ++s2) { float& x = outs[c][s2]; x = std::isfinite (x) ? std::tanh (x * 0.9f) / 0.9f : 0.0f; }
        double e = 0; for (int c = 0; c < std::min (2, nOut); ++c) for (int s2 = 0; s2 < seg; ++s2) e += outs[c][s2] * outs[c][s2];
        outLoud = 0.9f * outLoud + 0.1f * (float) (10 * std::log10 (e / (2 * seg) + 1e-12));
    }

    void applyEngineParams()
    {
        synth.grains.sizeMs = params.grainSize; synth.grains.spray = params.spray; synth.grains.beta = params.focus;
        synth.grains.pitchFollow = params.pitchFollow;
        synth.mosaic.space = (Space) std::clamp (params.space, 0, 3); synth.mosaic.mode = std::clamp (params.playback, 0, 2);
        synth.mosaic.ringK = params.ringSize; synth.mosaic.pitchFollow = params.pitchFollow;
        synth.phantom.imprint = params.imprint;
    }

    struct Env
    {
        float v = 0, a = 0.01f, r = 0.5f, sr = 48000; bool on = false;
        float next() { const float t = on ? 1.0f : 0.0f; const float tc = on ? a : r; v += (t - v) * (1 - std::exp (-1.0f / (std::max (0.001f, tc) * sr))); return v; }
    } gate;

    void handleMidi (const MidiEv& e)
    {
        if (e.type == 1) { held.push_back (e.note); noteVel = e.vel / 127.0f; gate.on = true; }
        else if (e.type == 0) { held.erase (std::remove (held.begin(), held.end(), e.note), held.end()); gate.on = ! held.empty(); }
        else if (e.type == 2) bend = e.bend;
        gate.a = params.attack; gate.r = params.release;
    }

    void hop (const RawFrame& raw)
    {
        ++hopCount;
        const bool present = raw.loud > params.gateDb;
        presence += ((present ? 1.0f : 0.0f) - presence) * (present ? 0.2f : 0.02f);
        // ---- feed the brain
        FeedItem it; toFeatures (raw, it.x);
        std::copy (raw.harm, raw.harm + kHarm, it.harm); std::copy (raw.noise, raw.noise + kNoise, it.noise);
        std::copy (cond.begin(), cond.end(), it.cond); it.voiced = raw.voicing;
        it.learn = params.learn && present && params.listen;
        it.hasSpat = stereoIn && present;
        std::copy (spatNow.begin(), spatNow.end(), it.spat);
        if (params.listen || present) ring->push (it);
        // carrier for imprinting
        analyser.window (carrierWin.data());

        // ---- inference
        const Snapshot& S = brain->acquire();
        float xs[kFeat]; S.feat.apply (it.x, xs); for (auto& v : xs) v = std::clamp (v, -6.0f, 6.0f);
        float zIn[kZ]; S.vae.encode (xs, cond.data(), zIn);
        // dream forward
        float zG[kZ], zE[kZ], zD[kZ];
        S.gru.step (zPrev, cond.data(), gruH, zG, params.temperature, rng);
        float ein[EchoState::I]; std::copy (zPrev, zPrev + kZ, ein); std::copy (cond.begin(), cond.end(), ein + kZ);
        if (! S.esn.W.empty()) { S.esn.advance (esnS.data(), ein); S.esn.readout (esnS.data(), zE); }
        else std::copy (zPrev, zPrev + kZ, zE);
        std::normal_distribution<float> nd;
        for (int j = 0; j < kZ; ++j)
        {
            zE[j] += params.temperature * 0.3f * nd (rng);
            zD[j] = params.dreamEngine == 0 ? zG[j] : params.dreamEngine == 1 ? zE[j] : 0.5f * (zG[j] + zE[j]);
        }
        // lucidity: how much the input steers (only when there is input); the notes' "dream" axis leans away from it
        const float lucid = std::clamp (params.lucid - 0.3f * axis (TextEmbed::Dream), 0.0f, 1.0f) * presence;
        float z[kZ];
        for (int j = 0; j < kZ; ++j) z[j] = lucid * zIn[j] + (1 - lucid) * zD[j];
        // probe / route gravity
        float zp[kZ]; bool havePull = false;
        if (params.route && route.size() >= 2)
        {
            routePos += params.routeSpeed / 60.0f * analyser.hopSize() / sr;
            const int segN = (int) route.size();
            const float p = std::fmod (routePos, (float) segN); const int a = (int) p; const float f = p - a;
            const int na = route[(size_t) a], nb = route[(size_t) ((a + 1) % segN)];
            for (int j = 0; j < kZ; ++j) zp[j] = S.som.w[(size_t) na * kZ + j] * (1 - f) + S.som.w[(size_t) nb * kZ + j] * f;
            havePull = true;
        }
        else if (params.gravity > 0.001f) { S.som.at (params.probeX, params.probeY, zp); havePull = true; }
        const float grav = params.route ? std::max (params.gravity, 0.85f) : params.gravity;
        float zBeforePull[kZ]; std::copy (z, z + kZ, zBeforePull);
        if (havePull) for (int j = 0; j < kZ; ++j) z[j] += grav * (zp[j] - z[j]);
        for (int j = 0; j < kZ; ++j) labPull[j] = z[j] - zBeforePull[j];
        labLucid = lucid; labGrav = havePull ? grav : 0.0f; labHavePull = havePull;
        // drift: an Ornstein–Uhlenbeck walk through latent space
        const float dr = std::clamp (params.drift + 0.3f * axis (TextEmbed::Motion), 0.0f, 1.0f);
        for (int j = 0; j < kZ; ++j) { driftV[j] = 0.97f * driftV[j] + dr * 0.12f * nd (rng); z[j] = std::clamp (z[j] + driftV[j], -5.0f, 5.0f); labDrift[j] = driftV[j]; }
        labDriftMix = dr;
        std::copy (z, z + kZ, zPrev);
        // decode
        float out[kOut]; S.vae.decode (z, cond.data(), out);
        float feat[kFeat]; S.feat.undo (out, feat);
        float noiseDb[kNoise]; S.noise.undo (out + kFeat + kHarm, noiseDb);
        const int bmu = S.som.bmu (z), inBmu = S.som.bmu (zIn);
        // the spatial network: what stereo image goes with this latent (mean and σ, back in plain units)
        {
            float m[kSpatial], lv[kSpatial];
            S.spatial.infer (z, cond.data(), m, lv);
            for (int j = 0; j < kSpatial; ++j)
            {
                const float sd = std::sqrt (S.spat.var[(size_t) j] + 1e-4f);
                spatPredNow[(size_t) j] = m[j] * sd + S.spat.mean[(size_t) j];
                spatSdNow[(size_t) j] = std::exp (0.5f * lv[j]) * sd;
            }
            spatTrained = S.spatSteps > 100;
            // continuous map position of the latent (softmin over the stars), for the base direction in space
            float best[3] = { 1e30f, 1e30f, 1e30f }; int bi[3] = { 0, 0, 0 };
            for (int nn = 0; nn < kSom * kSom; ++nn)
            {
                float d = 0; for (int j = 0; j < kZ; ++j) { const float e = S.som.w[(size_t) nn * kZ + j] - z[j]; d += e * e; }
                for (int k = 0; k < 3; ++k) if (d < best[k]) { for (int q = 2; q > k; --q) { best[q] = best[q - 1]; bi[q] = bi[q - 1]; } best[k] = d; bi[k] = nn; break; }
            }
            float u = 0, v = 0, ws = 0;
            for (int k = 0; k < 3; ++k) { const float w = std::exp (-(best[k] - best[0]) * 2); u += w * ((bi[k] % kSom) + 0.5f) / kSom; v += w * ((bi[k] / kSom) + 0.5f) / kSom; ws += w; }
            mapU = u / ws; mapV = v / ws;
            std::copy (z, z + kZ, zNow);
        }
        const bool trained = S.steps > 200;

        // ---- control frame
        Control c;
        const float tiltDb = (params.tilt + 0.6f * axis (TextEmbed::Bright)) * 12, warm = axis (TextEmbed::Warm);
        for (int b = 0; b < kMel; ++b)
        {
            const float octFrom1k = (float) b / kMel * 7 - 3.5f;   // rough position of the band in octaves
            c.mel[b] = (trained ? feat[b] : raw.mel[b]) + tiltDb * octFrom1k / 3.5f - warm * 6 * std::max (0.0f, octFrom1k) / 3.5f;
        }
        double cs = 0; for (int k = 0; k < kChroma; ++k) { c.chroma[k] = std::max (0.0f, trained ? feat[kMel + k] : raw.chroma[k]); cs += c.chroma[k]; }
        for (auto& v : c.chroma) v = cs > 0 ? v / (float) cs : 1.0f / kChroma;
        float mx = -1e9f; for (int k = 0; k < kHarm; ++k) mx = std::max (mx, out[kFeat + k]);
        double hs = 0; for (int k = 0; k < kHarm; ++k) { c.harm[k] = trained ? std::exp (out[kFeat + k] - mx) : raw.harm[k]; hs += c.harm[k]; }
        for (auto& v : c.harm) v = hs > 0 ? v / (float) hs : 0;
        for (int k = 0; k < kNoise; ++k) c.noise[k] = (trained ? noiseDb[k] : raw.noise[k]) + 6 * axis (TextEmbed::Rough);
        const float netF0 = 440.0f * std::pow (2.0f, std::clamp (feat[kF0], -4.0f, 3.0f));
        float f0 = netF0;
        if (params.pitchSource == 0 && raw.f0 > 0 && present) f0 = raw.f0;
        if (! held.empty()) f0 = 440.0f * std::pow (2.0f, (held.back() - 69 + bend * 2) / 12.0f);
        f0 *= std::pow (2.0f, params.transpose / 12.0f);
        c.f0 = std::clamp (f0, 25.0f, 4000.0f);
        c.voicing = std::clamp (trained ? feat[kVoicing] : raw.voicing, 0.0f, 1.0f);
        c.loud = std::clamp (trained ? feat[kLoud] : raw.loud, -80.0f, 0.0f);
        if (! held.empty() && params.keysOnly) c.loud = std::max (c.loud, -24.0f) + 6 * (noteVel - 0.7f);
        c.centroid = 1000.0f * std::pow (2.0f, feat[kCentroid]);
        c.flatness = std::pow (10.0f, std::clamp (feat[kFlatness], -4.0f, 0.0f)) + 0.2f * std::max (0.0f, axis (TextEmbed::Rough));
        c.flux = std::max (0.0f, feat[kFlux]);
        labNetF0 = netF0; labTrained = trained; labSteps = S.steps;
        labPitchFrom = ! held.empty() ? 2 : (params.pitchSource == 0 && raw.f0 > 0 && present) ? 1 : 0;
        c.onset = raw.onset;
        std::copy (z, z + kZ, c.z);
        c.gate = 1;
        ctl = c;
        synth.setControl (c);
        synth.grains.density *= std::max (0.2f, 1 + axis (TextEmbed::Dense));

        // morph source
        switch (params.morphSource)
        {
            case 1: morphNow = 0.5f + 0.5f * std::tanh (z[0] * 0.8f); break;
            case 2: morphNow = (float) (bmu % kSom) / (kSom - 1); break;
            case 3: lfo += params.morphRate * analyser.hopSize() / sr; morphNow = 0.5f + 0.5f * std::sin (kTwoPi * lfo); break;
            default: morphNow = params.morph; break;
        }

        // ---- remember this moment (corpus unit): at onsets and every 250 ms while listening
        if (params.listen && present && (raw.onset || hopCount - lastUnitHop >= 25) && corpus.written > (int64_t) (0.35f * sr))
        {
            Unit u; u.len = (int) (0.3f * sr); u.start = corpus.written - u.len;
            std::copy (zIn, zIn + kZ, u.z.begin()); melCepstrum (raw.mel, u.timbre.data());
            std::copy (raw.chroma, raw.chroma + kChroma, u.chroma.begin()); u.loud = raw.loud; u.f0 = raw.f0;
            corpus.add (u);
            lastUnitHop = hopCount;
        }
        // keep unit latents current as the network learns (a few per hop)
        for (int k = 0; k < 6 && ! corpus.units.empty(); ++k)
        {
            refreshIdx = (refreshIdx + 1) % (int) corpus.units.size();
            // re-encode from its stored timbre is not possible; nudge toward the encoder's view via the SOM node
            // that the unit currently maps to (keeps keys on the same manifold as the dream)
            auto& u = corpus.units[(size_t) refreshIdx];
            const int b = S.som.bmu (u.z.data());
            for (int j = 0; j < kZ; ++j) u.z[(size_t) j] += 0.02f * (S.som.w[(size_t) b * kZ + j] - u.z[(size_t) j]);
        }

        // ---- telemetry
        {
            std::lock_guard<std::mutex> l (telLock);
            std::copy (z, z + kZ, tel.z); std::copy (zIn, zIn + kZ, tel.zIn); std::copy (zD, zD + kZ, tel.zDream);
            tel.bmu = bmu; tel.inputBmu = inBmu; tel.inLoud = raw.loud; tel.outLoud = outLoud; tel.f0 = c.f0; tel.voicing = c.voicing;
            tel.presence = presence; tel.surprise = brain->stats.surprise;
            std::copy (labPull, labPull + kZ, tel.zPull); std::copy (labDrift, labDrift + kZ, tel.zDrift);
            tel.lucidMix = labLucid; tel.gravMix = labGrav; tel.driftMix = labDriftMix; tel.havePull = labHavePull; tel.netDrives = labTrained;
            std::copy (c.harm, c.harm + kHarm, tel.synHarm); std::copy (c.noise, c.noise + kNoise, tel.synNoise);
            tel.netF0 = labNetF0; tel.synF0 = c.f0; tel.synVoicing = c.voicing; tel.synLoud = c.loud; tel.synCentroid = c.centroid; tel.synFlat = c.flatness; tel.synFlux = c.flux;
            tel.pitchFrom = labPitchFrom; tel.netSteps = labSteps;
            std::copy (c.mel, c.mel + kMel, tel.mel); std::copy (raw.mel, raw.mel + kMel, tel.melIn); std::copy (c.chroma, c.chroma + kChroma, tel.chroma);
            std::copy (synth.lastW, synth.lastW + Synth::kLayers, tel.weights);
            // probe results: the nearest remembered moments ("signals") to where we are
            tel.nHits = 0;
            std::array<std::pair<float, int>, 8> best; int nb = 0;
            for (int k = 0; k < (int) corpus.units.size(); ++k)
            {
                const auto& u = corpus.units[(size_t) k];
                if (! corpus.valid (u)) continue;
                float d = 0; for (int j = 0; j < kZ; ++j) { const float e = u.z[(size_t) j] - z[j]; d += e * e; }
                if (nb < 8) best[(size_t) nb++] = { d, k };
                else { auto w = std::max_element (best.begin(), best.end()); if (d < w->first) *w = { d, k }; }
            }
            std::sort (best.begin(), best.begin() + nb);
            for (int k = 0; k < nb; ++k)
            {
                const auto& u = corpus.units[(size_t) best[(size_t) k].second];
                tel.hits[(size_t) k] = { best[(size_t) k].second, std::exp (-0.5f * best[(size_t) k].first), u.loud, u.f0 };
            }
            tel.nHits = nb; tel.corpusUnits = corpus.count(); tel.routePos = routePos; tel.hop = hopCount;
            std::copy (spatNow.begin(), spatNow.end(), tel.spatIn); std::copy (spatPredNow.begin(), spatPredNow.end(), tel.spatPred);
            std::copy (spatSdNow.begin(), spatSdNow.end(), tel.spatSd); tel.stereoInput = stereoIn; tel.spatSteps = S.spatSteps;
            for (int k = 0; k < 5; ++k) tel.sources[(size_t) k] = { srcCtl[(size_t) k].az, srcCtl[(size_t) k].el, srcCtl[(size_t) k].width, srcCtl[(size_t) k].dist, k < 4 ? synth.lastW[k] * (synth.layerLive (k) ? 1.0f : 0.0f) : 1 - params.mix };
            const auto& rd = scene.reflectionDirs(); tel.nRefl = (int) std::min<size_t> (6, rd.size());
            for (int k = 0; k < tel.nRefl; ++k) tel.reflections[(size_t) k] = rd[(size_t) k];
            tel.fieldAz = scene.fieldAz; tel.fieldEl = scene.fieldEl; tel.fieldDiffuse = scene.fieldDiffuse;
            for (int k = 0; k < 256; ++k) { tel.scopeL[(size_t) k] = scope[(size_t) ((scopePos + k) % 256)].first; tel.scopeR[(size_t) k] = scope[(size_t) ((scopePos + k) % 256)].second; }
        }
        brain->release();
    }

    std::shared_ptr<Brain> brain; std::shared_ptr<Brain::Ring> ring;
    float sr = 48000; int sinceHop = 0; uint64_t hopCount = 0, lastUnitHop = 0;
    Analyser analyser; Corpus corpus; Synth synth; Stargate stargate; Nebula nebula;
    std::vector<float> wetL, wetR, mono; std::array<float, 2048> carrierWin {};
    std::array<float, kCond> cond {}, pendingCond {}; std::atomic<bool> notesChanged { false }; std::mutex notesLock; std::vector<std::string> noteWords;
    std::vector<int> route, pendingRoute; std::atomic<bool> routeChanged { false }; std::mutex routeLock; float routePos = 0;
    float gruH[kGruH] {}; std::vector<float> esnS; float zPrev[kZ] {}, driftV[kZ] {};
    float presence = 0, morphNow = 0, lfo = 0, outLoud = -100, dcL = 0, dcR = 0, bend = 0, noteVel = 0.8f;
    std::vector<int> held; std::mt19937 rng { 7 };
    Control ctl; int refreshIdx = 0;
    std::mutex telLock; Telemetry tel;
    // the latent-to-synthesis path of the last hop (for the lab page)
    float labPull[kZ] {}, labDrift[kZ] {}, labLucid = 0, labGrav = 0, labDriftMix = 0, labNetF0 = 0; bool labHavePull = false, labTrained = false; int labPitchFrom = 0; uint64_t labSteps = 0;
    SpscRing<Chunk*, 64> inbox, done;
    StereoAnalyser stereo; std::array<float, kSpatial> spatNow {}, spatPredNow {}, spatSdNow {}; bool stereoIn = false, spatTrained = false;
    std::array<std::pair<float, float>, 256> scope {}; int scopePos = 0;
    SpatialScene scene; std::array<SourceCtl, SpatialScene::kSources> srcCtl {};
    std::array<std::vector<float>, 4> layL, layR; std::vector<float> dryL, dryR;
    float* const* spatialOut = nullptr; int spatialN = 0;
    float mapU = 0.5f, mapV = 0.5f, orbitPhase = 0, zNow[kZ] {};
};
} // namespace ns
