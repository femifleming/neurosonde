// Neurosonde — synthesis. The network decodes a Control frame every 10 ms (spectral envelope, harmonic distribution,
// noise envelope, pitch, loudness, chroma ...); each engine turns it into sound in its own way, after instruments and
// techniques from the user's world:
//   harmonic  DDSP harmonic + filtered noise (Engel et al. 2020)
//   phantom   magnitude-only inverse STFT with propagated/scattered phase (the artefacts of an unfinished network)
//   partials  inharmonic additive peaks (Partials)
//   grains    granular recall of remembered moments (Cloudfin) — modern Hopfield retrieval in latent space
//   mosaic    concatenative unit selection: point / ring / path (Coalescence)
//   ear tones difference-tone primaries: the ear makes the pitch (Oneirophone, after Chechile)
//   strings   sympathetic Karplus–Strong bank with jawari (Tantri)
//   voice     glottal source through moving formants (Spreekbuis)
//   gendy     Xenakis' dynamic stochastic synthesis (GENDY)
//   table     wavetable regenerated from the decoded spectrum (disting EX)
//   tape      skipping-CD micro-loops and dust (openprocess, after Oval)
#pragma once
#include "Analysis.h"

namespace ns::dsp
{
constexpr float kTwoPi = 6.283185307f;

/** what the network says, every hop, in plain units */
struct Control
{
    float mel[kMel] {};        // dB
    float chroma[kChroma] {};
    float harm[kHarm] {};      // distribution
    float noise[kNoise] {};    // dB
    float z[kZ] {};
    float f0 = 220, voicing = 0, loud = -60, centroid = 1000, flatness = 0.1f, flux = 0;
    bool onset = false;
    float gate = 1;            // MIDI / dynamics gate 0..1
};

inline float dbToGain (float db) { return std::pow (10.0f, db / 20.0f); }

/** Chamberlin-style state-variable bandpass (stable to ~fs/6 with oversampled-free tuning) — TPT/ZDF form. */
struct SVF
{
    float g = 0, k = 1, ic1 = 0, ic2 = 0, a1 = 0, a2 = 0, a3 = 0;
    void set (float fc, float q, float sr)
    {
        g = std::tan ((float) M_PI * std::clamp (fc, 10.0f, sr * 0.45f) / sr); k = 1.0f / std::max (0.05f, q);
        a1 = 1 / (1 + g * (g + k)); a2 = g * a1; a3 = g * a2;
    }
    float bp (float x)
    {
        const float v3 = x - ic2, v1 = a1 * ic1 + a2 * v3, v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2;
        return v1;
    }
    float lp (float x)
    {
        const float v3 = x - ic2, v1 = a1 * ic1 + a2 * v3, v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2;
        return v2;
    }
    void reset() { ic1 = ic2 = 0; }
};

struct Rand
{
    uint32_t s = 22222;
    float uni() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (float) (s & 0xffffff) / 16777216.0f; }
    float bi() { return 2 * uni() - 1; }
};

/** Mel-envelope lookup at an arbitrary frequency (dB), linear in mel. */
inline float envAt (const Control& c, float hz, float sr)
{
    const float mLo = hzToMel (30.0f), mHi = hzToMel (std::min (16000.0f, sr * 0.48f));
    const float pos = (hzToMel (std::max (30.0f, hz)) - mLo) / (mHi - mLo) * (kMel + 1) - 1;
    if (pos <= 0) return c.mel[0];
    if (pos >= kMel - 1) return c.mel[kMel - 1] - 6 * (pos - (kMel - 1));
    const int i = (int) pos; const float f = pos - i;
    return c.mel[i] * (1 - f) + c.mel[i + 1] * f;
}

/** A stored moment of sound (for grains, mosaic and tape). */
struct Unit { int64_t start = 0; int len = 0; std::array<float, kZ> z {}; std::array<float, 16> timbre {}; std::array<float, kChroma> chroma {}; float loud = -60, f0 = 0; };

/** Mono audio memory (a ring) + its units. Owned by one engine instance; written and read on the audio thread. */
struct Corpus
{
    std::vector<float> ring; int64_t written = 0; float sr = 48000;
    std::vector<Unit> units; int maxUnits = 2048;
    void prepare (float sampleRate, float seconds = 180)
    {
        sr = sampleRate; ring.assign ((size_t) (sampleRate * seconds), 0.0f); written = 0; units.clear(); units.reserve ((size_t) maxUnits);
    }
    void write (const float* x, int n)
    {
        for (int i = 0; i < n; ++i) ring[(size_t) ((written + i) % (int64_t) ring.size())] = x[i];
        written += n;
    }
    bool valid (const Unit& u) const { return u.start >= written - (int64_t) ring.size() + 4096 && u.start + u.len <= written; }
    float at (double pos) const
    {
        const int64_t i = (int64_t) std::floor (pos); const float f = (float) (pos - (double) i);
        const size_t n = ring.size();
        const float a = ring[(size_t) (((i % (int64_t) n) + (int64_t) n) % (int64_t) n)], b = ring[(size_t) ((((i + 1) % (int64_t) n) + (int64_t) n) % (int64_t) n)];
        return a + (b - a) * f;
    }
    void add (const Unit& u)
    {
        // evict invalid units first, then the oldest
        if ((int) units.size() >= maxUnits)
        {
            auto it = std::find_if (units.begin(), units.end(), [&] (const Unit& v) { return ! valid (v); });
            if (it == units.end()) it = units.begin();
            *it = u; return;
        }
        units.push_back (u);
    }
    int count() const { int n = 0; for (auto& u : units) n += valid (u) ? 1 : 0; return n; }
};

/** feature space for corpus similarity (after Coalescence's descriptor choice) */
enum class Space { Latent, Timbre, Chroma, Loudness };
inline float unitDistance (const Unit& u, const Control& c, const float* timbre, Space s)
{
    float d = 0;
    switch (s)
    {
        case Space::Latent: for (int j = 0; j < kZ; ++j) { const float e = u.z[(size_t) j] - c.z[j]; d += e * e; } break;
        case Space::Timbre: for (int j = 0; j < 16; ++j) { const float e = u.timbre[(size_t) j] - timbre[j]; d += e * e; } d *= 0.05f; break;
        case Space::Chroma: for (int j = 0; j < kChroma; ++j) { const float e = u.chroma[(size_t) j] - c.chroma[j]; d += e * e; } d *= 40.0f; break;
        case Space::Loudness: { const float e = (u.loud - c.loud) / 6.0f; d = e * e; } break;
    }
    return d;
}
/** 16 cepstral coefficients of the mel envelope (DCT-II): the timbre descriptor */
inline void melCepstrum (const float* mel, float* out)
{
    for (int k = 0; k < 16; ++k)
    {
        float s = 0; for (int i = 0; i < kMel; ++i) s += mel[i] * std::cos ((float) M_PI * k * (i + 0.5f) / kMel);
        out[k] = s / kMel;
    }
}

/** Interface: setControl() once per hop, render() adds into the output. */
struct Engine
{
    virtual ~Engine() = default;
    virtual void prepare (float sampleRate) { sr = sampleRate; }
    virtual void setControl (const Control& c, float character) = 0;
    virtual void render (float* L, float* R, int n) = 0;
    float sr = 48000;
};

// ---------------------------------------------------------------------------------------------------------------
struct HarmonicEngine : Engine
{
    double ph[kHarm] {}; float amp[kHarm] {}, tgt[kHarm] {}; float f0 = 220, f0t = 220, nGainT = 0;
    SVF bands[kNoise]; float bg[kNoise] {}, bgT[kNoise] {}; Rand rnd;
    void prepare (float s) override
    {
        Engine::prepare (s);
        for (int b = 0; b < kNoise; ++b) { const float lo = noiseBandEdge (b), hi = noiseBandEdge (b + 1); bands[b].set (std::sqrt (lo * hi), std::sqrt (lo * hi) / (hi - lo) * 1.2f, sr); }
    }
    void setControl (const Control& c, float character) override
    {
        f0t = c.f0;
        const float a = dbToGain (c.loud + 6) * c.gate;
        const float harmonic = std::clamp (c.voicing * 1.2f - 0.1f, 0.0f, 1.0f) * (1 - 0.6f * character);
        for (int k = 0; k < kHarm; ++k) tgt[k] = (c.f0 * (k + 1) < sr * 0.45f) ? a * harmonic * c.harm[k] * 1.4f : 0.0f;
        // noise: bands shaped by the decoded noise envelope, scaled to sit under the harmonics
        float mx = -200; for (int b = 0; b < kNoise; ++b) mx = std::max (mx, c.noise[b]);
        for (int b = 0; b < kNoise; ++b) bgT[b] = dbToGain (c.noise[b] - mx) * a * (1 - harmonic * 0.7f) * (0.35f + character) * 1.6f;
    }
    void render (float* L, float* R, int n) override
    {
        const float sm = 1 - std::exp (-1.0f / (0.01f * sr));
        for (int i = 0; i < n; ++i)
        {
            f0 += (f0t - f0) * sm;
            float s = 0; const double w = kTwoPi * f0 / sr;
            for (int k = 0; k < kHarm; ++k)
            {
                amp[k] += (tgt[k] - amp[k]) * sm;
                if (amp[k] < 1e-6f) { ph[k] += w * (k + 1); continue; }
                ph[k] += w * (k + 1); if (ph[k] > kTwoPi) ph[k] -= kTwoPi * std::floor (ph[k] / kTwoPi);
                s += amp[k] * (float) std::sin (ph[k]);
            }
            const float wn = rnd.bi(); float nz = 0;
            for (int b = 0; b < kNoise; ++b) { bg[b] += (bgT[b] - bg[b]) * sm; if (bg[b] > 1e-6f) nz += bg[b] * bands[b].bp (wn); }
            L[i] += s + nz; R[i] += s + nz;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** magnitude from the network, phase propagated (+ scattered by `character`), or imprinted on the live input */
struct PhantomEngine : Engine
{
    static constexpr int N = 2048;
    RealFFT fft; std::vector<float> win, ola, frame, inBuf; std::vector<std::complex<float>> X, Xin;
    std::vector<double> phase; int hop = 480; int64_t olaPos = 0; Rand rnd; float norm = 1;
    std::vector<float> out; int outRead = 0, outAvail = 0;
    const float* carrier = nullptr;   // live input window (N samples) when imprinting
    float imprint = 0;
    void prepare (float s) override
    {
        Engine::prepare (s); fft.init (N); hop = std::max (64, (int) std::lround (s * 0.01));
        win.resize (N); for (int i = 0; i < N; ++i) win[(size_t) i] = 0.5f - 0.5f * std::cos (kTwoPi * i / N);
        double ws = 0; for (int i = 0; i < N; ++i) ws += win[(size_t) i] * win[(size_t) i];
        norm = (float) (hop / ws);
        ola.assign ((size_t) N * 2, 0); frame.assign (N, 0); X.assign (N / 2 + 1, {}); Xin = X; phase.assign (N / 2 + 1, 0);
        out.assign ((size_t) hop * 4, 0); outRead = outAvail = 0; inBuf.assign (N, 0);
    }
    void setControl (const Control& c, float character) override
    {
        const float a = c.gate;
        if (carrier && imprint > 0) { for (int i = 0; i < N; ++i) inBuf[(size_t) i] = carrier[i] * win[(size_t) i]; fft.forward (inBuf.data(), Xin.data()); }
        for (int k = 1; k < N / 2; ++k)
        {
            const float hz = k * sr / N;
            const float mag = dbToGain (envAt (c, hz, sr) + 58) * a;     // mel band energy → per-bin magnitude (approx.)
            phase[(size_t) k] += kTwoPi * k * hop / N + character * kTwoPi * rnd.bi() * 0.5f;
            std::complex<float> p (std::cos ((float) phase[(size_t) k]), std::sin ((float) phase[(size_t) k]));
            if (carrier && imprint > 0)
            {
                const float m = std::abs (Xin[(size_t) k]);
                if (m > 1e-9f) p = (1 - imprint) * p + imprint * Xin[(size_t) k] / m;
            }
            X[(size_t) k] = mag * p;
        }
        X[0] = 0; X[N / 2] = 0;
        fft.inverse (X.data(), frame.data());
        // overlap-add
        for (int i = 0; i < N; ++i) ola[(size_t) i] += frame[(size_t) i] * win[(size_t) i] * norm;
        // emit hop samples
        for (int i = 0; i < hop; ++i) out[(size_t) ((outRead + outAvail + i) % (int) out.size())] = ola[(size_t) i];
        outAvail = std::min ((int) out.size(), outAvail + hop);
        std::move (ola.begin() + hop, ola.end(), ola.begin()); std::fill (ola.end() - hop, ola.end(), 0.0f);
    }
    void render (float* L, float* R, int n) override
    {
        for (int i = 0; i < n && outAvail > 0; ++i)
        {
            const float s = out[(size_t) outRead]; outRead = (outRead + 1) % (int) out.size(); --outAvail;
            L[i] += s; R[i] += s;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
struct PartialsEngine : Engine
{
    static constexpr int P = 32;
    float freq[P] {}, ftgt[P] {}, amp[P] {}, atgt[P] {}, pan[P] {}; double ph[P] {}; Rand rnd;
    void prepare (float s) override { Engine::prepare (s); for (int p = 0; p < P; ++p) { freq[p] = ftgt[p] = 200.0f * (p + 1); pan[p] = rnd.bi() * 0.6f; } }
    void setControl (const Control& c, float character) override
    {
        // peaks of the decoded envelope over a fine grid
        struct Pk { float hz, db; };
        Pk pk[96]; int np = 0;
        float prev2 = -200, prev = -200, prevHz = 0;
        for (int i = 0; i < 400 && np < 96; ++i)
        {
            const float hz = 40.0f * std::pow (400.0f, i / 399.0f), db = envAt (c, hz, sr);
            if (i > 1 && prev > prev2 && prev >= db) pk[np++] = { prevHz, prev };
            prev2 = prev; prev = db; prevHz = hz;
        }
        std::sort (pk, pk + np, [] (const Pk& a, const Pk& b) { return a.db > b.db; });
        // plus the harmonic series of the decoded pitch, stretched by `character` (inharmonicity)
        const float stretch = 1 + 0.08f * character;
        for (int p = 0; p < P; ++p)
        {
            float hz, db;
            if (p < np && p < P / 2) { hz = pk[p].hz; db = pk[p].db; }
            else { const int k = p - std::min (np, P / 2) + 1; hz = c.f0 * k * std::pow ((float) k, stretch - 1); db = envAt (c, hz, sr) + 20 * std::log10 (std::max (1e-4f, c.harm[std::min (kHarm - 1, k - 1)] * 4)); }
            ftgt[p] = std::clamp (hz * (1 + character * 0.01f * rnd.bi()), 30.0f, sr * 0.45f);
            atgt[p] = dbToGain (db + 14) * c.gate / P * 4;
        }
    }
    void render (float* L, float* R, int n) override
    {
        const float sm = 1 - std::exp (-1.0f / (0.02f * sr));
        for (int i = 0; i < n; ++i)
        {
            float l = 0, r = 0;
            for (int p = 0; p < P; ++p)
            {
                freq[p] += (ftgt[p] - freq[p]) * sm; amp[p] += (atgt[p] - amp[p]) * sm;
                ph[p] += kTwoPi * freq[p] / sr; if (ph[p] > kTwoPi) ph[p] -= kTwoPi;
                const float s = amp[p] * (float) std::sin (ph[p]);
                l += s * (1 - pan[p]); r += s * (1 + pan[p]);
            }
            L[i] += l; R[i] += r;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** grains recalled from memory: each grain picks a remembered unit by modern-Hopfield attention over latents */
struct GrainEngine : Engine
{
    Corpus* corpus = nullptr;
    struct G { double pos = 0, rate = 1; int len = 0, age = 0; float gain = 0, pan = 0; bool on = false; };
    std::array<G, 64> g {}; Rand rnd; std::mt19937 mt { 9 };
    float density = 12, sizeMs = 120, spray = 0.3f, beta = 6, pitchFollow = 0.5f, f0 = 220, amp = 0; double acc = 0;
    std::vector<std::array<float, kZ>> keys; std::vector<int> map; Control last;
    void setControl (const Control& c, float character) override
    {
        last = c; f0 = c.f0; amp = dbToGain (c.loud + 9) * c.gate;
        density = 4 + 60 * character * character + 30 * std::clamp ((c.loud + 50) / 40, 0.0f, 1.0f);
        // keys for recall (rebuilt per hop: units change)
        keys.clear(); map.clear();
        if (corpus) for (int i = 0; i < (int) corpus->units.size(); ++i) if (corpus->valid (corpus->units[(size_t) i])) { keys.push_back (corpus->units[(size_t) i].z); map.push_back (i); }
    }
    void spawn()
    {
        if (keys.empty()) return;
        for (auto& v : g) if (! v.on)
        {
            const int k = ml::hopfieldRecall (last.z, keys, beta, mt);
            // the keys are from the last hop: the corpus may have been cleared since (re-prepare)
            if (k < 0 || (size_t) k >= map.size() || (size_t) map[(size_t) k] >= corpus->units.size() || ! corpus->valid (corpus->units[(size_t) map[(size_t) k]])) { keys.clear(); map.clear(); return; }
            const Unit& u = corpus->units[(size_t) map[(size_t) k]];
            v.len = (int) (sizeMs * (0.7f + 0.6f * rnd.uni()) * sr / 1000);
            v.pos = (double) u.start + rnd.uni() * std::max (1, u.len - v.len / 2);
            v.rate = (u.f0 > 0 && f0 > 0) ? std::pow ((double) f0 / u.f0, (double) pitchFollow) : 1.0;
            v.rate = std::clamp (v.rate, 0.25, 4.0);
            v.age = 0; v.pan = spray * rnd.bi(); v.gain = amp; v.on = true;
            return;
        }
    }
    void render (float* L, float* R, int n) override
    {
        if (! corpus) return;
        for (int i = 0; i < n; ++i)
        {
            // grains per sample; never let a bad rate or density run away (an infinite accumulator would never drain)
            const double inc = sr > 0 && std::isfinite (density) ? std::clamp ((double) density / sr, 0.0, 1.0) : 0.0;
            acc += inc; if (! (acc < 4)) acc = 0;
            while (acc >= 1) { acc -= 1; spawn(); }
            float l = 0, r = 0;
            for (auto& v : g)
            {
                if (! v.on) continue;
                const float w = 0.5f - 0.5f * std::cos (kTwoPi * v.age / v.len);
                const float s = corpus->at (v.pos) * w * v.gain;
                l += s * (1 - v.pan); r += s * (1 + v.pan);
                v.pos += v.rate; if (++v.age >= v.len) v.on = false;
            }
            L[i] += l; R[i] += r;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** concatenative synthesis after Coalescence: point (nearest), ring (random among the near), path (prefer the unit
    that followed the last one in time, so recordings continue unless the target moves away) */
struct MosaicEngine : Engine
{
    Corpus* corpus = nullptr; Space space = Space::Latent; int mode = 0; float ringK = 8, pitchFollow = 0.0f;
    struct V { double pos = 0, rate = 1; int idx = -1, left = 0; float gain = 0; };
    V cur, old; int xfade = 0, xlen = 960; Rand rnd; float unitMs = 160, amp = 0, f0 = 220; Control last; float timbre[16] {};
    void setControl (const Control& c, float character) override
    {
        last = c; amp = dbToGain (c.loud + 9) * c.gate; f0 = c.f0;
        unitMs = 60 + 400 * (1 - character); melCepstrum (c.mel, timbre);
    }
    int choose()
    {
        if (! corpus) return -1;
        const auto& U = corpus->units;
        int best = -1; float bd = 1e30f;
        std::array<std::pair<float, int>, 16> near; int nn = 0;
        for (int i = 0; i < (int) U.size(); ++i)
        {
            if (! corpus->valid (U[(size_t) i])) continue;
            float d = unitDistance (U[(size_t) i], last, timbre, space);
            if (mode == 2 && cur.idx >= 0 && (size_t) cur.idx < U.size() && U[(size_t) i].start == U[(size_t) cur.idx].start + U[(size_t) cur.idx].len) d *= 0.35f;   // continuity
            if (d < bd) { bd = d; best = i; }
            if (nn < 16) near[(size_t) nn++] = { d, i };
            else { auto w = std::max_element (near.begin(), near.end()); if (d < w->first) *w = { d, i }; }
        }
        if (mode == 1 && nn > 0)
        {
            std::sort (near.begin(), near.begin() + nn);
            return near[(size_t) (rnd.uni() * std::min ((float) nn, ringK))].second;
        }
        return best;
    }
    void render (float* L, float* R, int n) override
    {
        if (! corpus) return;
        for (int i = 0; i < n; ++i)
        {
            if (cur.left <= 0)
            {
                const int k = choose();
                if (k >= 0)
                {
                    old = cur; xfade = xlen;
                    const Unit& u = corpus->units[(size_t) k];
                    cur.idx = k; cur.pos = (double) u.start; cur.left = (int) (unitMs * sr / 1000); cur.gain = amp;
                    cur.rate = (u.f0 > 0 && f0 > 0) ? std::clamp (std::pow ((double) f0 / u.f0, (double) pitchFollow), 0.25, 4.0) : 1.0;
                }
                else cur.left = 480;
            }
            float s = cur.idx >= 0 ? corpus->at (cur.pos) * cur.gain : 0;
            if (xfade > 0 && old.idx >= 0)
            {
                const float t = (float) xfade / xlen;
                s = s * (1 - t) + corpus->at (old.pos) * old.gain * t;
                old.pos += old.rate; --xfade;
            }
            cur.pos += cur.rate; --cur.left;
            L[i] += s; R[i] += s;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** difference tones: for each target pitch (f0, and the two strongest envelope peaks) two high primaries f1, f2 with
    f2 - f1 = target (quadratic DT) or 2f1 - f2 = target (cubic DT). The ear produces the tone. */
struct EarToneEngine : Engine
{
    static constexpr int T = 3;
    double ph[T][2] {}; float f[T][2] {}, ft[T][2] {}, a[T] {}, at[T] {};
    void setControl (const Control& c, float character) override
    {
        float targets[T] = { c.f0, 0, 0 };
        // the two strongest formant-ish peaks under 1.2 kHz
        float best[2] = { -200, -200 }; float bhz[2] = { c.f0 * 2, c.f0 * 3 };
        for (int i = 0; i < 120; ++i)
        {
            const float hz = 150.0f * std::pow (8.0f, i / 119.0f), d = envAt (c, hz, sr);
            if (d > best[0]) { best[1] = best[0]; bhz[1] = bhz[0]; best[0] = d; bhz[0] = hz; }
            else if (d > best[1] && std::abs (hz - bhz[0]) > 120) { best[1] = d; bhz[1] = hz; }
        }
        targets[1] = bhz[0]; targets[2] = bhz[1];
        const float base = 1600 + 1800 * character;   // primaries' region
        const float amp = dbToGain (std::max (c.loud, -40.0f) + 3) * c.gate * 0.5f;
        for (int t = 0; t < T; ++t)
        {
            const float tg = std::clamp (targets[t], 40.0f, 1200.0f);
            if (t == 1) { ft[t][0] = base * 1.1f; ft[t][1] = 2 * ft[t][0] - tg; }      // cubic: 2f1 - f2
            else { ft[t][0] = base * (1 + 0.13f * t); ft[t][1] = ft[t][0] + tg; }      // quadratic: f2 - f1
            at[t] = amp * (t == 0 ? 1.0f : 0.6f);
        }
    }
    void render (float* L, float* R, int n) override
    {
        const float sm = 1 - std::exp (-1.0f / (0.03f * sr));
        for (int i = 0; i < n; ++i)
        {
            float l = 0, r = 0;
            for (int t = 0; t < T; ++t)
            {
                a[t] += (at[t] - a[t]) * sm;
                for (int k = 0; k < 2; ++k)
                {
                    f[t][k] += (ft[t][k] - f[t][k]) * sm;
                    ph[t][k] += kTwoPi * f[t][k] / sr; if (ph[t][k] > kTwoPi) ph[t][k] -= kTwoPi;
                }
                const float s1 = (float) std::sin (ph[t][0]), s2 = (float) std::sin (ph[t][1]);
                l += a[t] * (s1 + 0.9f * s2) * 0.5f; r += a[t] * (0.9f * s1 + s2) * 0.5f;
            }
            L[i] += l; R[i] += r;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** 24 sympathetic strings tuned to the decoded chroma (two octaves above the decoded pitch class of f0),
    Karplus–Strong with a jawari bridge (a buzzing, displacement-dependent soft clip) */
struct StringsEngine : Engine
{
    static constexpr int S = 24;
    struct Str { std::vector<float> d; int w = 0, len = 100; float lp = 0, gain = 0, gt = 0, fb = 0.995f; };
    std::array<Str, S> s; Rand rnd; float exc = 0, excT = 0, jawari = 0.3f, rootHz = 110; float nz[kNoise] {}; bool onset = false;
    void prepare (float sr0) override { Engine::prepare (sr0); for (auto& x : s) { x.d.assign ((size_t) (sr0 / 30), 0); x.w = 0; } }
    void setControl (const Control& c, float character) override
    {
        jawari = character;
        if (c.f0 > 30) rootHz = c.f0;
        while (rootHz > 140) rootHz *= 0.5f; while (rootHz < 70) rootHz *= 2;
        const float rootMidi = 69 + 12 * std::log2 (rootHz / 440.0f);
        for (int i = 0; i < S; ++i)
        {
            const int pc = i % 12, oct = i / 12;
            const float midi = std::round (rootMidi) + ((pc - ((int) std::round (rootMidi) % 12) + 12) % 12) + 12 * (oct + 1);
            const float hz = 440.0f * std::pow (2.0f, (midi - 69) / 12);
            s[(size_t) i].len = std::clamp ((int) (sr / hz), 8, (int) s[(size_t) i].d.size() - 1);
            s[(size_t) i].gt = std::pow (c.chroma[pc], 1.5f) * 3;
            s[(size_t) i].fb = 0.993f + 0.006f * (1 - character * 0.5f);
        }
        excT = dbToGain (c.loud + 3) * c.gate;
        onset = c.onset;
    }
    void render (float* L, float* R, int n) override
    {
        if (onset) { for (auto& x : s) for (int k = 0; k < x.len; ++k) x.d[(size_t) ((x.w + k) % x.len)] += x.gt * excT * 0.4f * rnd.bi(); onset = false; }
        for (int i = 0; i < n; ++i)
        {
            exc += (excT - exc) * 0.001f;
            const float e = exc * rnd.bi() * 0.02f;   // a constant sympathetic stir
            float l = 0, r = 0;
            for (int k = 0; k < S; ++k)
            {
                auto& x = s[(size_t) k];
                x.gain += (x.gt - x.gain) * 0.0005f;
                const int a = x.w, b = (x.w + 1) % x.len;
                float y = 0.5f * (x.d[(size_t) a] + x.d[(size_t) b]);
                // jawari: the string grazes the curved bridge — soft clipping that grows with displacement
                y = y - jawari * 0.3f * y * std::abs (y);
                x.d[(size_t) a] = y * x.fb + e * x.gain;
                x.w = b;
                if (k % 2) l += y * x.gain; else r += y * x.gain;
            }
            L[i] += l * 1.6f; R[i] += r * 1.6f;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** glottal pulse + aspiration through five formants found in the decoded envelope */
struct VoiceEngine : Engine
{
    static constexpr int F = 5;
    SVF fm[F][2]; float fhz[F] { 500, 1500, 2500, 3500, 4500 }, ft[F] { 500, 1500, 2500, 3500, 4500 }, fg[F] {}, fgt[F] {};
    double ph = 0; float f0 = 120, f0t = 120, voiced = 1, amp = 0, ampT = 0, shift = 1; Rand rnd; SVF tilt;
    void prepare (float s) override { Engine::prepare (s); tilt.set (2500, 0.5f, s); }
    void setControl (const Control& c, float character) override
    {
        f0t = c.f0; voiced = std::clamp (c.voicing * 1.3f - 0.2f, 0.0f, 1.0f); ampT = dbToGain (c.loud + 10) * c.gate;
        shift = 0.7f + 0.6f * character;
        // find peaks of the envelope between 200 Hz and 5 kHz
        float pk[F] {}; float pd[F]; for (auto& v : pd) v = -300; int found = 0;
        float prev2 = -300, prev = -300, prevHz = 0;
        for (int i = 0; i < 200; ++i)
        {
            const float hz = 200.0f * std::pow (25.0f, i / 199.0f), d = envAt (c, hz, sr);
            if (i > 1 && prev > prev2 && prev >= d && found < F) { pk[found] = prevHz; pd[found] = prev; ++found; }
            prev2 = prev; prev = d; prevHz = hz;
        }
        const float def[F] = { 500, 1500, 2500, 3500, 4500 };
        float mx = -300; for (int k = 0; k < found; ++k) mx = std::max (mx, pd[k]);
        for (int k = 0; k < F; ++k) { ft[k] = (k < found ? pk[k] : def[k]) * shift; fgt[k] = k < found ? dbToGain ((pd[k] - mx) * 0.7f) : 0.15f; }
    }
    void render (float* L, float* R, int n) override
    {
        const float sm = 1 - std::exp (-1.0f / (0.015f * sr));
        for (int i = 0; i < n; ++i)
        {
            f0 += (f0t - f0) * sm; amp += (ampT - amp) * sm;
            if (i % 32 == 0) for (int k = 0; k < F; ++k) { fhz[k] += (ft[k] - fhz[k]) * 0.2f; fg[k] += (fgt[k] - fg[k]) * 0.2f; fm[k][0].set (fhz[k], 8 + k * 2, sr); fm[k][1].set (fhz[k] * 1.01f, 8 + k * 2, sr); }
            ph += f0 / sr; if (ph >= 1) ph -= 1;
            // Rosenberg-like glottal pulse: rising cosine, sharp closure
            const float t = (float) ph; const float glottal = t < 0.6f ? 0.5f - 0.5f * std::cos ((float) M_PI * t / 0.6f) : std::cos ((float) M_PI * (t - 0.6f) / 0.8f);
            const float src = voiced * (glottal - 0.5f) + (1.2f - voiced) * 0.6f * rnd.bi();
            const float x = tilt.lp (src);
            float l = 0, r = 0;
            for (int k = 0; k < F; ++k) { l += fg[k] * fm[k][0].bp (x); r += fg[k] * fm[k][1].bp (x); }
            L[i] += l * amp * 0.16f; R[i] += r * amp * 0.16f;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** GENDY: a polygon wave whose breakpoints random-walk in amplitude and duration, between mirrors (Xenakis) */
struct GendyEngine : Engine
{
    static constexpr int M = 12;
    float amp[M] {}, dur[M] {}; int seg = 0; float t = 0, a0 = 0, a1 = 0, f0 = 110, rough = 0.2f, level = 0; Rand rnd; SVF lp;
    void prepare (float s) override { Engine::prepare (s); for (int i = 0; i < M; ++i) { amp[i] = rnd.bi() * 0.5f; dur[i] = 1; } lp.set (4000, 0.7f, s); }
    void setControl (const Control& c, float character) override
    {
        f0 = std::clamp (c.f0 > 30 ? c.f0 : f0, 30.0f, 2000.0f);
        rough = 0.05f + 0.6f * character + 0.5f * std::clamp (c.flatness * 3, 0.0f, 1.0f);
        level = dbToGain (c.loud + 6) * c.gate;
        lp.set (std::clamp (c.centroid * 2.5f, 300.0f, sr * 0.4f), 0.7f, sr);
    }
    static float mirror (float x, float lo, float hi) { while (x < lo || x > hi) x = x < lo ? 2 * lo - x : 2 * hi - x; return x; }
    void render (float* L, float* R, int n) override
    {
        for (int i = 0; i < n; ++i)
        {
            float total = 0; for (float d : dur) total += d;
            const float segLen = dur[seg] / total * sr / f0;
            t += 1; if (t >= segLen)
            {
                t -= segLen; a0 = a1; seg = (seg + 1) % M;
                amp[seg] = mirror (amp[seg] + rough * 0.3f * rnd.bi(), -1, 1);
                dur[seg] = mirror (dur[seg] + rough * 0.4f * rnd.bi(), 0.2f, 3);
                a1 = amp[seg];
            }
            const float y = lp.lp (a0 + (a1 - a0) * (t / std::max (1.0f, segLen))) * level * 0.7f;
            L[i] += y; R[i] += y;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** a single-cycle table rebuilt each hop from the decoded harmonics, crossfaded, wavefolded by character */
struct TableEngine : Engine
{
    static constexpr int TN = 2048;
    std::vector<float> tab[2]; int curT = 0; float xf = 1; double ph = 0; float f0 = 220, f0t = 220, amp = 0, ampT = 0, fold = 0;
    void prepare (float s) override { Engine::prepare (s); tab[0].assign (TN + 1, 0); tab[1].assign (TN + 1, 0); }
    void setControl (const Control& c, float character) override
    {
        f0t = c.f0; ampT = dbToGain (c.loud + 6) * c.gate; fold = character;
        auto& t = tab[1 - curT];
        std::fill (t.begin(), t.end(), 0.0f);
        const int K = std::min (kHarm, (int) (sr * 0.45f / std::max (30.0f, c.f0)));
        for (int k = 0; k < K; ++k)
        {
            const float a = c.harm[k] * dbToGain ((envAt (c, c.f0 * (k + 1), sr) - envAt (c, c.f0, sr)) * 0.3f);
            if (a < 1e-4f) continue;
            for (int i = 0; i < TN; ++i) t[(size_t) i] += a * std::sin (kTwoPi * (k + 1) * i / TN + 0.3f * k * k);
        }
        float mx = 1e-6f; for (float v : t) mx = std::max (mx, std::abs (v));
        for (auto& v : t) v /= mx; t[TN] = t[0];
        curT = 1 - curT; xf = 0;
    }
    float read (const std::vector<float>& t, double p) const { const double x = p * TN; const int i = (int) x; const float f = (float) (x - i); return t[(size_t) i] + (t[(size_t) i + 1] - t[(size_t) i]) * f; }
    void render (float* L, float* R, int n) override
    {
        const float sm = 1 - std::exp (-1.0f / (0.01f * sr)), xs = 1.0f / (0.01f * sr);
        for (int i = 0; i < n; ++i)
        {
            f0 += (f0t - f0) * sm; amp += (ampT - amp) * sm; xf = std::min (1.0f, xf + xs);
            ph += f0 / sr; if (ph >= 1) ph -= 1;
            float y = read (tab[curT], ph) * xf + read (tab[1 - curT], ph) * (1 - xf);
            if (fold > 0.01f) { const float g = 1 + 4 * fold; y = std::sin (y * g * (float) M_PI * 0.5f); }
            y *= amp * 0.5f;
            L[i] += y; R[i] += y;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
/** openprocess / Oval: corpus fragments stuck in micro-loops like a skipping CD, plus dust */
struct TapeEngine : Engine
{
    Corpus* corpus = nullptr; Rand rnd; double pos = 0, loopStart = 0; int loopLen = 2400, loopLeft = 0, repeats = 0; float amp = 0, skip = 0.5f, dust = 0, gain = 0;
    Control last; float timbre[16] {};
    void setControl (const Control& c, float character) override { last = c; amp = dbToGain (c.loud + 9) * c.gate; skip = character; dust = std::clamp (c.flux / 6, 0.0f, 1.0f); melCepstrum (c.mel, timbre); }
    void render (float* L, float* R, int n) override
    {
        if (! corpus || corpus->units.empty()) return;
        for (int i = 0; i < n; ++i)
        {
            if (loopLeft <= 0)
            {
                if (repeats > 0) { --repeats; pos = loopStart; loopLeft = loopLen; }
                else
                {
                    // pick a unit near the current timbre
                    int best = -1; float bd = 1e30f;
                    for (int k = 0; k < 64 && ! corpus->units.empty(); ++k)
                    {
                        const int u = std::min ((int) corpus->units.size() - 1, (int) (rnd.uni() * corpus->units.size()));
                        if (! corpus->valid (corpus->units[(size_t) u])) continue;
                        const float d = unitDistance (corpus->units[(size_t) u], last, timbre, Space::Timbre);
                        if (d < bd) { bd = d; best = u; }
                    }
                    if (best < 0) { loopLeft = 480; continue; }
                    const Unit& un = corpus->units[(size_t) best];
                    loopStart = (double) un.start + rnd.uni() * un.len * 0.5;
                    loopLen = (int) ((0.01f + 0.07f * rnd.uni()) * sr);
                    repeats = rnd.uni() < skip ? 1 + (int) (rnd.uni() * 12 * skip) : 0;
                    pos = loopStart; loopLeft = loopLen * (repeats > 0 ? 1 : 4);
                    gain = amp;
                }
            }
            const float edge = std::min (1.0f, std::min ((float) (loopLen - loopLeft), (float) loopLeft) / 48.0f);
            float s = corpus->at (pos) * gain * (repeats > 0 ? std::max (0.3f, edge) : 1.0f);
            if (rnd.uni() < dust * 0.0008f) s += (rnd.uni() < 0.5f ? 0.3f : -0.3f) * amp;
            pos += 1; --loopLeft;
            L[i] += s; R[i] += s;
        }
    }
};

enum class Mode { Harmonic, Phantom, Partials, Grains, Mosaic, EarTones, Strings, Voice, Gendy, Table, Tape, Count };
inline const char* modeName (int m)
{
    static const char* n[] = { "harmonic", "phantom", "partials", "grains", "mosaic", "ear tones", "strings", "voice", "gendy", "table", "tape" };
    return n[m];
}
inline const char* modeLineage (int m)
{
    static const char* n[] = { "DDSP harmonic + noise", "magnitude-only resynthesis: the network's artefacts", "inharmonic additive peaks · Partials",
                               "granular recall by Hopfield attention · Cloudfin", "concatenative point / ring / path · Coalescence",
                               "difference-tone primaries · Oneirophone", "sympathetic strings + jawari · Tantri", "glottal source through formants · Spreekbuis",
                               "dynamic stochastic synthesis · Xenakis", "wavetable from the decoded spectrum · disting EX", "skipping micro-loops and dust · openprocess / Oval" };
    return n[m];
}

/** Spectral combination of up to four layer signals (one channel): log-magnitude interpolation ("merge") or the
    envelope of one layer pressed onto another's fine structure ("imprint"). STFT 1024, hop 256, Hann^2 OLA. */
struct SpectralCombiner
{
    static constexpr int N = 1024, H = 256, L = 4;
    RealFFT fft; std::vector<float> win, frame, ola, out; std::array<std::vector<float>, L> in; std::array<std::vector<std::complex<float>>, L> X;
    std::vector<std::complex<float>> Y; std::vector<float> envA, envB; int fill = 0, outPos = 0; float norm = 1;
    void prepare()
    {
        fft.init (N); win.resize (N); for (int i = 0; i < N; ++i) win[(size_t) i] = 0.5f - 0.5f * std::cos (kTwoPi * i / N);
        double ws = 0; for (float w : win) ws += w * w; norm = (float) (H / ws);
        frame.assign (N, 0); ola.assign (N, 0); out.assign (H, 0); for (auto& v : in) v.assign (N, 0); for (auto& v : X) v.assign (N / 2 + 1, {});
        Y.assign (N / 2 + 1, {}); envA.assign (N / 2 + 1, 0); envB = envA; m.assign (N / 2 + 1, 0); fill = 0; outPos = 0;
    }
    std::vector<float> m;
    void smoothEnv (const std::vector<std::complex<float>>& S, std::vector<float>& env)
    {
        // envelope: moving average of the magnitude over ±8 bins
        const int n = (int) S.size(); double acc = 0; int cnt = 0;
        m.resize ((size_t) n); for (int k = 0; k < n; ++k) m[(size_t) k] = std::abs (S[(size_t) k]);
        for (int k = 0; k < n; ++k)
        {
            const int lo = std::max (0, k - 8), hi = std::min (n - 1, k + 8);
            if (k == 0) { acc = 0; for (int j = lo; j <= hi; ++j) acc += m[(size_t) j]; cnt = hi - lo + 1; }
            else { if (k + 8 < n) { acc += m[(size_t) (k + 8)]; ++cnt; } if (k - 9 >= 0) { acc -= m[(size_t) (k - 9)]; --cnt; } }
            env[(size_t) k] = (float) (acc / std::max (1, cnt));
        }
    }
    /** push one sample per layer; returns the combined sample (latency N). weights: per-layer, sum to 1.
        mode 0 = merge (weighted log-magnitude, phase of the strongest), 1 = imprint (layer 0's envelope on the others) */
    float process (const float* x, const float* w, int mode)
    {
        for (int l = 0; l < L; ++l) in[(size_t) l][(size_t) (N - H + fill)] = x[l];
        const float y = out[(size_t) outPos];
        if (++outPos >= H) outPos = 0;
        if (++fill >= H)
        {
            fill = 0;
            for (int l = 0; l < L; ++l)
            {
                if (w[l] <= 0 && ! (mode == 1 && l == 0)) continue;
                for (int i = 0; i < N; ++i) frame[(size_t) i] = in[(size_t) l][(size_t) i] * win[(size_t) i];
                fft.forward (frame.data(), X[(size_t) l].data());
            }
            if (mode == 0)
            {
                for (int k = 0; k <= N / 2; ++k)
                {
                    double lm = 0; float best = -1; std::complex<float> ph (1, 0);
                    for (int l = 0; l < L; ++l)
                    {
                        if (w[l] <= 0) continue;
                        const float m = std::abs (X[(size_t) l][(size_t) k]);
                        lm += w[l] * std::log (m + 1e-7f);
                        if (m * w[l] > best) { best = m * w[l]; ph = m > 1e-12f ? X[(size_t) l][(size_t) k] / m : std::complex<float> (1, 0); }
                    }
                    Y[(size_t) k] = (float) std::exp (lm) * ph;
                }
            }
            else
            {
                smoothEnv (X[0], envA);
                std::fill (Y.begin(), Y.end(), std::complex<float> (0, 0));
                for (int l = 1; l < L; ++l)
                {
                    if (w[l] <= 0) continue;
                    smoothEnv (X[(size_t) l], envB);
                    for (int k = 0; k <= N / 2; ++k) Y[(size_t) k] += w[l] * X[(size_t) l][(size_t) k] * (envA[(size_t) k] / (envB[(size_t) k] + 1e-6f));
                }
            }
            fft.inverse (Y.data(), frame.data());
            for (int i = 0; i < N; ++i) ola[(size_t) i] += frame[(size_t) i] * win[(size_t) i] * norm * 2.0f;
            for (int i = 0; i < H; ++i) out[(size_t) i] = ola[(size_t) i];
            std::move (ola.begin() + H, ola.end(), ola.begin()); std::fill (ola.end() - H, ola.end(), 0.0f);
            for (auto& v : in) std::move (v.begin() + H, v.end(), v.begin());
        }
        return y;
    }
};

enum class Combine { Stack, Morph, Merge, Imprint };
inline const char* combineName (int c) { static const char* n[] = { "stack", "morph", "merge", "imprint" }; return n[c]; }

/** owns one of every engine and four layers that combine by stacking, morphing, spectral merging or imprinting.
    The same engine type can't run twice (each has one state), so a layer that repeats an engine above it is muted. */
struct Synth
{
    static constexpr int kLayers = 4;
    HarmonicEngine harmonic; PhantomEngine phantom; PartialsEngine partials; GrainEngine grains; MosaicEngine mosaic;
    EarToneEngine ear; StringsEngine strings; VoiceEngine voice; GendyEngine gendy; TableEngine table; TapeEngine tape;
    Engine* all[(int) Mode::Count] { &harmonic, &phantom, &partials, &grains, &mosaic, &ear, &strings, &voice, &gendy, &table, &tape };
    struct Layer { int mode = 0; bool on = false; float level = 1, pan = 0, character = 0.3f; };
    std::array<Layer, kLayers> layers;
    std::array<std::vector<float>, kLayers> bufL, bufR;
    SpectralCombiner specL, specR;
    float lastW[kLayers] {};
    void prepare (float sr, int maxBlock, Corpus* corpus)
    {
        for (auto* e : all) e->prepare (sr);
        grains.corpus = mosaic.corpus = tape.corpus = corpus;
        for (auto& b : bufL) b.assign ((size_t) maxBlock, 0); for (auto& b : bufR) b.assign ((size_t) maxBlock, 0);
        specL.prepare(); specR.prepare();
    }
    bool layerLive (int l) const
    {
        if (! layers[(size_t) l].on) return false;
        for (int k = 0; k < l; ++k) if (layers[(size_t) k].on && layers[(size_t) k].mode == layers[(size_t) l].mode) return false;
        return true;
    }
    void setControl (const Control& c)
    {
        for (int l = 0; l < kLayers; ++l) if (layerLive (l)) all[layers[(size_t) l].mode]->setControl (c, layers[(size_t) l].character);
    }
    /** morph weights: equal-power crossfade along the live layers at position m (0..1) */
    void morphWeights (float m, float* w) const
    {
        int live[kLayers]; int n = 0; for (int l = 0; l < kLayers; ++l) { w[l] = 0; if (layerLive (l)) live[n++] = l; }
        if (n == 0) return;
        if (n == 1) { w[live[0]] = 1; return; }
        const float p = std::clamp (m, 0.0f, 1.0f) * (n - 1); const int i = std::min (n - 2, (int) p); const float f = p - i;
        w[live[i]] = std::cos (f * (float) M_PI * 0.5f); w[live[i + 1]] = std::sin (f * (float) M_PI * 0.5f);
    }
    /** render n samples into outL/outR (added). latency of the spectral modes: SpectralCombiner::N samples */
    void render (float* outL, float* outR, int n, Combine mode, float morph)
    {
        for (int l = 0; l < kLayers; ++l)
        {
            std::fill (bufL[(size_t) l].begin(), bufL[(size_t) l].begin() + n, 0.0f); std::fill (bufR[(size_t) l].begin(), bufR[(size_t) l].begin() + n, 0.0f);
            if (! layerLive (l)) continue;
            all[layers[(size_t) l].mode]->render (bufL[(size_t) l].data(), bufR[(size_t) l].data(), n);
        }
        float w[kLayers];
        if (mode == Combine::Stack) for (int l = 0; l < kLayers; ++l) w[l] = layerLive (l) ? 1.0f : 0.0f;
        else morphWeights (morph, w);
        if (mode == Combine::Imprint)   // the first live layer is the envelope; the morph moves between the others
        {
            int first = -1; for (int l = 0; l < kLayers; ++l) if (layerLive (l)) { first = l; break; }
            float rest[kLayers] {}; int idx[kLayers]; int m = 0;
            for (int l = 0; l < kLayers; ++l) if (layerLive (l) && l != first) idx[m++] = l;
            if (m == 1) rest[idx[0]] = 1;
            else if (m > 1) { const float p = std::clamp (morph, 0.0f, 1.0f) * (m - 1); const int i = std::min (m - 2, (int) p); const float f = p - i; rest[idx[i]] = std::cos (f * 1.5707963f); rest[idx[i + 1]] = std::sin (f * 1.5707963f); }
            for (int l = 0; l < kLayers; ++l) w[l] = rest[l];
            if (first >= 0) w[first] = 0;
            // reorder so the envelope layer is slot 0 of the combiner
            for (int i = 0; i < n; ++i)
            {
                float xl[4] {}, xr[4] {}, ww[4] {};
                int k = 1;
                if (first >= 0) { xl[0] = bufL[(size_t) first][(size_t) i] * layers[(size_t) first].level; xr[0] = bufR[(size_t) first][(size_t) i] * layers[(size_t) first].level; }
                for (int l = 0; l < kLayers && k < 4; ++l)
                    if (l != first && layerLive (l)) { xl[k] = bufL[(size_t) l][(size_t) i] * layers[(size_t) l].level; xr[k] = bufR[(size_t) l][(size_t) i] * layers[(size_t) l].level; ww[k] = w[l]; ++k; }
                outL[i] += specL.process (xl, ww, 1); outR[i] += specR.process (xr, ww, 1);
            }
        }
        else if (mode == Combine::Merge)
        {
            for (int i = 0; i < n; ++i)
            {
                float xl[4], xr[4];
                for (int l = 0; l < kLayers; ++l) { xl[l] = bufL[(size_t) l][(size_t) i] * layers[(size_t) l].level; xr[l] = bufR[(size_t) l][(size_t) i] * layers[(size_t) l].level; }
                float ww[4]; float s = 0; for (int l = 0; l < kLayers; ++l) s += w[l] * w[l];
                for (int l = 0; l < kLayers; ++l) ww[l] = s > 0 ? w[l] * w[l] / s : 0;     // power weights sum to 1
                outL[i] += specL.process (xl, ww, 0); outR[i] += specR.process (xr, ww, 0);
            }
        }
        else
        {
            for (int l = 0; l < kLayers; ++l)
            {
                if (w[l] <= 0) continue;
                const float g = w[l] * layers[(size_t) l].level, pan = layers[(size_t) l].pan;
                const float gl = g * std::min (1.0f, 1 - pan), gr = g * std::min (1.0f, 1 + pan);
                for (int i = 0; i < n; ++i) { outL[i] += bufL[(size_t) l][(size_t) i] * gl; outR[i] += bufR[(size_t) l][(size_t) i] * gr; }
            }
        }
        for (int l = 0; l < kLayers; ++l) lastW[l] = w[l];
    }
    /** spatial output: each live layer (weighted by stack / morph and its level) into its own stereo pair, so it can be
        placed in space on its own. Merge / imprint make one sound: it goes to the first pair. */
    void renderLayers (float* const* Ls, float* const* Rs, int n, Combine mode, float morph)
    {
        for (int l = 0; l < kLayers; ++l) { std::fill (Ls[l], Ls[l] + n, 0.0f); std::fill (Rs[l], Rs[l] + n, 0.0f); }
        if (mode == Combine::Merge || mode == Combine::Imprint) { render (Ls[0], Rs[0], n, mode, morph); return; }
        for (int l = 0; l < kLayers; ++l)
        {
            std::fill (bufL[(size_t) l].begin(), bufL[(size_t) l].begin() + n, 0.0f); std::fill (bufR[(size_t) l].begin(), bufR[(size_t) l].begin() + n, 0.0f);
            if (layerLive (l)) all[layers[(size_t) l].mode]->render (bufL[(size_t) l].data(), bufR[(size_t) l].data(), n);
        }
        float w[kLayers];
        if (mode == Combine::Stack) for (int l = 0; l < kLayers; ++l) w[l] = layerLive (l) ? 1.0f : 0.0f;
        else morphWeights (morph, w);
        for (int l = 0; l < kLayers; ++l)
        {
            const float g = w[l] * layers[(size_t) l].level;
            if (g <= 0) continue;
            for (int i = 0; i < n; ++i) { Ls[l][i] = bufL[(size_t) l][(size_t) i] * g; Rs[l][i] = bufR[(size_t) l][(size_t) i] * g; }
        }
        for (int l = 0; l < kLayers; ++l) lastW[l] = w[l];
    }
};
} // namespace ns::dsp
