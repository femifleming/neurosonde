// Every synthesis engine on its own, driven by controls like the network's (a harmonic spectrum with formants, a
// noisy one, pitches from 55 Hz to 1.76 kHz, onsets), at character 0, 0.5 and 1: finite, bounded, audible, at a
// level near the harmonic engine's, and well inside real time. Then the output limiter: nothing above the ceiling,
// sample-exact latency, bit-transparent below the ceiling.
//   clang++ -std=c++20 -O2 -ISource tests/EngineCheck.cpp -framework Accelerate -o /tmp/enginecheck && /tmp/enginecheck
#include "dsp/Synths.h"
#include "dsp/Limiter.h"
#include <chrono>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace ns; using namespace ns::dsp;
static int fails = 0;
#define CHECK(c, ...) do { std::printf ((c) ? "  ok   " : "  FAIL "); std::printf (__VA_ARGS__); std::printf ("\n"); if (! (c)) ++fails; } while (0)

static Control makeControl (float f0, bool noisy, float loud, int hop)
{
    Control c;
    c.f0 = f0; c.voicing = noisy ? 0.1f : 0.9f; c.loud = loud; c.flatness = noisy ? 0.5f : 0.05f; c.flux = noisy ? 3 : 0.5f;
    c.centroid = noisy ? 4000 : f0 * 4; c.gate = 1; c.onset = hop % 40 == 0;
    for (int i = 0; i < kMel; ++i)
    {
        const float hz = melToHz (hzToMel (30) + (hzToMel (16000) - hzToMel (30)) * (i + 1) / (kMel + 1));
        // a spectral tilt with two formants (a vowel-ish envelope), or flat noise
        float db = noisy ? loud - 6 - 3 * std::log2 (hz / 1000 + 1) : loud - 6 * std::log2 (std::max (1.0f, hz / f0));
        if (! noisy) db += 12 * std::exp (-std::pow (std::log2 (hz / 700), 2) * 8) + 9 * std::exp (-std::pow (std::log2 (hz / 1800), 2) * 8);
        c.mel[i] = db;
    }
    float hs = 0; for (int k = 0; k < kHarm; ++k) { c.harm[k] = 1.0f / (k + 1); hs += c.harm[k]; } for (auto& h : c.harm) h /= hs;
    for (int b = 0; b < kNoise; ++b) c.noise[b] = loud - 31 - (noisy ? 2 : 30) - b * 0.3f;   // the analysis' scale: per-bin power, white noise ~ loud - 31
    for (int i = 0; i < kChroma; ++i) c.chroma[i] = i == 9 ? 0.5f : 0.05f;
    for (int j = 0; j < kZ; ++j) c.z[j] = std::sin (hop * 0.01f + j);
    return c;
}

int main()
{
    const float sr = 48000; const int hop = 480;
    // a corpus for grains / mosaic / tape: 20 s of a melody, units every 160 ms
    Corpus corpus; corpus.prepare (sr, 60);
    {
        std::vector<float> a ((size_t) (20 * sr));
        for (size_t i = 0; i < a.size(); ++i) { const double t = i / sr; const double f = 220 * std::pow (2.0, ((int) (t * 3) % 7) / 12.0); a[i] = (float) (0.3 * std::sin (2 * M_PI * f * t) * std::exp (-std::fmod (t, 0.333) * 4)); }
        corpus.write (a.data(), (int) a.size());
        for (int64_t s = 0; s + 7680 < (int64_t) a.size(); s += 7680)
        {
            Unit u; u.start = s; u.len = 7680; u.loud = -20; u.f0 = 220; for (int j = 0; j < kZ; ++j) u.z[(size_t) j] = std::sin (s * 1e-4f + j); corpus.add (u);
        }
    }
    std::printf ("engines, level-matched to the decoded loudness, -18 dBFS (5 s each, 4 pitches x tonal/noisy x character 0 / 0.5 / 1):\n");
    double refRms = 0;
    for (int m = 0; m < (int) Mode::Count; ++m)
    {
        auto synth = std::make_unique<Synth>();
        synth->prepare (sr, hop, &corpus);
        synth->matchLevels = std::getenv ("NS_RAW") == nullptr;
        double worstPk = 0, sumRms = 0, worstCpu = 0; int nRuns = 0, silent = 0; bool finite = true;
        for (float ch : { 0.0f, 0.5f, 1.0f })
            for (float f0 : { 55.0f, 220.0f, 880.0f, 1760.0f })
                for (bool noisy : { false, true })
                {
                    synth->layers = {}; synth->layers[0] = { m, true, 1.0f, 0.0f, ch };
                    std::vector<float> L (hop), R (hop);
                    double e = 0, pk = 0; long n = 0;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int h = 0; h < (int) (5 * sr / hop); ++h)
                    {
                        synth->setControl (makeControl (f0, noisy, -18, h));
                        std::fill (L.begin(), L.end(), 0.0f); std::fill (R.begin(), R.end(), 0.0f);
                        synth->render (L.data(), R.data(), hop, Combine::Stack, 0);
                        for (int i = 0; i < hop; ++i)
                        {
                            if (! std::isfinite (L[(size_t) i]) || ! std::isfinite (R[(size_t) i])) finite = false;
                            if (h > 50) { e += 0.5 * (L[(size_t) i] * L[(size_t) i] + R[(size_t) i] * R[(size_t) i]); pk = std::max (pk, (double) std::max (std::abs (L[(size_t) i]), std::abs (R[(size_t) i]))); ++n; }
                        }
                    }
                    const double cpu = std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count() / 5.0;
                    const double rms = std::sqrt (e / std::max (1L, n));
                    worstPk = std::max (worstPk, pk); worstCpu = std::max (worstCpu, cpu); sumRms += 20 * std::log10 (rms + 1e-9); ++nRuns;
                    if (rms < 1e-4) ++silent;
                    if (std::getenv ("NS_VERBOSE") && std::string (std::getenv ("NS_VERBOSE")) == modeName (m))
                        std::printf ("      ch %.1f f0 %6.0f %s  rms %6.1f dB  peak %6.2f\n", ch, f0, noisy ? "noisy" : "tonal", 20 * std::log10 (rms + 1e-9), pk);
                }
        const double meanDb = sumRms / nRuns;
        if (m == 0) refRms = meanDb;
        const bool ok = finite && worstPk < 4.0 && silent <= 4 && worstCpu < 0.15 && (std::getenv ("NS_RAW") || std::abs (meanDb + 18) < 4.5);
        CHECK (ok, "%-9s mean level %6.1f dB (%+5.1f vs harmonic)  peak %5.2f  silent runs %d/24  cpu %4.1f%% of real time", modeName (m), meanDb, meanDb - refRms, worstPk, silent, worstCpu * 100);
    }

    std::printf ("pitch (autocorrelation, tonal, character 0.3):\n");
    for (int m : { (int) Mode::Harmonic, (int) Mode::Table, (int) Mode::Voice, (int) Mode::FM, (int) Mode::Bowed, (int) Mode::Scanned, (int) Mode::Pulsar })
    {
        float worst = 0;
        for (float f0 : { 110.0f, 220.0f, 440.0f, 880.0f })
        {
            auto synth = std::make_unique<Synth>(); synth->prepare (sr, hop, &corpus);
            synth->layers = {}; synth->layers[0] = { m, true, 1.0f, 0.0f, 0.3f };
            std::vector<float> L (hop), R (hop), rec;
            for (int h = 0; h < 200; ++h)
            {
                auto c = makeControl (f0, false, -18, h); c.onset = false;
                synth->setControl (c);
                std::fill (L.begin(), L.end(), 0.0f); std::fill (R.begin(), R.end(), 0.0f);
                synth->render (L.data(), R.data(), hop, Combine::Stack, 0);
                if (h >= 150) for (int i = 0; i < hop; ++i) rec.push_back (L[(size_t) i] + R[(size_t) i]);
            }
            // the strongest autocorrelation peak within 2.2x either side of the intended period (an octave error either
            // way shows up), then the shortest lag that is nearly as strong
            const double T = sr / f0;
            const int lo = std::max (2, (int) (T / 2.2)), hi = (int) (T * 2.2) + 2, n = (int) rec.size() - hi - 2;
            std::vector<double> ac ((size_t) hi + 3);
            double e0 = 0; for (int i = 0; i < n; ++i) e0 += rec[(size_t) i] * rec[(size_t) i];
            for (int lag = lo - 1; lag <= hi + 1; ++lag) { double a2 = 0; for (int i = 0; i < n; ++i) a2 += rec[(size_t) i] * rec[(size_t) (i + lag)]; ac[(size_t) lag] = a2 / (e0 + 1e-12); }
            double best = -1; int bl = 0;
            for (int lag = lo; lag <= hi; ++lag) if (ac[(size_t) lag] > best && ac[(size_t) lag] >= ac[(size_t) lag - 1] && ac[(size_t) lag] >= ac[(size_t) lag + 1]) { best = ac[(size_t) lag]; bl = lag; }
            {
                const int c0 = (int) std::lround (bl / 2.0); int cb = 0; double cv = -1;
                for (int lag = std::max (lo, c0 - 2); lag <= std::min (hi, c0 + 2); ++lag) if (ac[(size_t) lag] > cv) { cv = ac[(size_t) lag]; cb = lag; }
                if (cb > 0 && cv > 0.9 * best) bl = cb;
            }
            float est = 0;
            if (bl > 0) { const double a = ac[(size_t) bl - 1], b = ac[(size_t) bl], c = ac[(size_t) bl + 1]; const double d = 0.5 * (a - c) / (a - 2 * b + c); est = (float) (sr / (bl + d)); }
            const float cents = est > 0 ? 1200 * std::log2 (est / f0) : 9999;
            // an octave error counts as wrong
            worst = std::max (worst, std::abs (cents));
            if (std::getenv ("NS_PITCH")) std::printf ("      %-9s %6.1f Hz -> %7.1f Hz (%+.1f cents, ac %.2f)\n", modeName (m), f0, est, cents, best);
        }
        CHECK (worst < 25, "%-9s worst pitch error %.1f cents (110 / 220 / 440 / 880 Hz)", modeName (m), worst);
    }

    std::printf ("limiter:\n");
    {
        Limiter lim; lim.prepare (sr, 2);
        std::mt19937 rng (5); std::normal_distribution<float> nd;
        const float ceil = std::pow (10.0f, -1.0f / 20);
        std::vector<float> a (48000), b (48000);
        for (size_t i = 0; i < a.size(); ++i) { const float burst = (i / 4800) % 2 ? 8.0f : 0.3f; a[i] = nd (rng) * burst; b[i] = std::sin (i * 0.05f) * burst * 1.3f; }
        float* ch[2] = { a.data(), b.data() };
        lim.process (ch, 2, (int) a.size(), ceil, Limiter::Clean);
        float pk = 0; for (size_t i = 0; i < a.size(); ++i) pk = std::max (pk, std::max (std::abs (a[i]), std::abs (b[i])));
        CHECK (pk <= ceil + 1e-6f && lim.outOvers == 0, "clean: input up to +24 dBFS, output peak %.4f <= ceiling %.4f, %ld samples were above the ceiling going in", pk, ceil, lim.overs);
        Limiter soft; soft.prepare (sr, 2);
        for (size_t i = 0; i < a.size(); ++i) { a[i] = nd (rng) * 6; b[i] = nd (rng) * 6; }
        soft.process (ch, 2, (int) a.size(), ceil, Limiter::Soft);
        pk = 0; for (size_t i = 0; i < a.size(); ++i) pk = std::max (pk, std::max (std::abs (a[i]), std::abs (b[i])));
        CHECK (pk <= ceil + 1e-6f, "soft: output peak %.4f <= ceiling", pk);
        // latency and transparency: a quiet signal comes out exactly, delayed by latency()
        Limiter t; t.prepare (sr, 2);
        std::vector<float> x (4096), y (4096), x0;
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.25f * std::sin (i * 0.031f) + (i == 1000 ? 0.4f : 0.0f);
        x0 = x; y = x; float* c2[2] = { x.data(), y.data() };
        t.process (c2, 2, (int) x.size(), ceil, Limiter::Clean);
        const int L = t.latency(); float err = 0;
        for (size_t i = (size_t) L; i < x.size(); ++i) err = std::max (err, std::abs (x[i] - x0[i - (size_t) L]));
        CHECK (err == 0.0f, "transparent below the ceiling, delayed by exactly %d samples (%.2f ms): max difference %.1e", L, 1000.0 * L / sr, err);
        // 96 kHz, 64 channels (Neurosonde Space)
        Limiter big; big.prepare (96000, 64);
        std::vector<std::vector<float>> bufs (64, std::vector<float> (9600));
        std::vector<float*> ptr; for (auto& v : bufs) { for (auto& s2 : v) s2 = nd (rng) * 3; ptr.push_back (v.data()); }
        big.process (ptr.data(), 64, 9600, 1.0f, Limiter::Clean);
        pk = 0; for (auto& v : bufs) for (float s2 : v) pk = std::max (pk, std::abs (s2));
        CHECK (pk <= 1.0f && big.latency() > 100, "64 channels at 96 kHz, linked: peak %.4f, look-ahead %d samples", pk, big.latency());
    }
    std::printf (fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
