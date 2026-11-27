#include "../Source/Core.h"
#include "TestDir.h"
#include <cstdio>
#include <chrono>
using namespace ns;
static int fails = 0;
#define CHECK(c, ...) do { std::printf ((c) ? "  ok   " : "  FAIL "); std::printf (__VA_ARGS__); std::printf ("\n"); if (! (c)) ++fails; } while (0)
int main()
{
    auto brain = std::make_shared<Brain> (testDir ("ns_core_test"));
    brain->settings.intensive = true;
    Instance inst (brain);
    const int B = 256; const double sr = 48000;
    inst.prepare (sr, B);
    inst.setNotes ("warm dusty warehouse, golden light, slow drift");
    // a little piece: a melody of tones with noise breaths
    auto music = [] (int64_t i) {
        const double t = i / 48000.0; const int note = (int) (t * 2) % 5; const double f = 220 * std::pow (2.0, (std::array<int, 5> { 0, 3, 7, 10, 12 }[(size_t) note]) / 12.0);
        const double env = std::exp (-std::fmod (t, 0.5) * 3);
        return (float) (0.3 * env * (std::sin (2 * M_PI * f * t) + 0.3 * std::sin (4 * M_PI * f * t)) + 0.01 * std::sin (t * 1e5));
    };
    std::vector<float> L (B), R (B), oL (B), oR (B);
    int64_t pos = 0;
    auto run = [&] (double seconds, bool withInput, double* peak = nullptr, double* rms = nullptr, double* cpu = nullptr)
    {
        double e = 0, pk = 0; int64_t cnt = 0; bool finite = true;
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t k = 0; k < (int64_t) (seconds * sr / B); ++k)
        {
            for (int i = 0; i < B; ++i) L[(size_t) i] = R[(size_t) i] = withInput ? music (pos + i) : 0.0f;
            pos += B;
            inst.process (L.data(), R.data(), oL.data(), oR.data(), B);
            for (int i = 0; i < B; ++i) { if (! std::isfinite (oL[(size_t) i])) finite = false; e += oL[(size_t) i] * oL[(size_t) i]; pk = std::max (pk, (double) std::abs (oL[(size_t) i])); ++cnt; }
        }
        const double el = std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count();
        if (peak) *peak = pk; if (rms) *rms = std::sqrt (e / std::max<int64_t> (1, cnt)); if (cpu) *cpu = el / seconds;
        return finite;
    };
    inst.params.layer[0] = { true, 0, 1, 0, 0.3f };
    run (20, true);
    brain->trainFor (200);
    CHECK (brain->stats.vaeSteps > 100, "learned while playing: %llu steps, recon %.3f, corpus units %d", (unsigned long long) brain->stats.vaeSteps.load(), brain->stats.recon.load(), inst.telemetry().corpusUnits);
    for (int m = 0; m < (int) Mode::Count; ++m)
    {
        inst.params.layer = {}; inst.params.layer[0] = { true, m, 1, 0, 0.4f }; inst.params.combine = 0;
        double pk, rms, cpu;
        const bool fin = run (3, true, &pk, &rms, &cpu);
        CHECK (fin && rms > 1e-4 && pk < 4, "%-10s rms %.4f peak %.2f  cpu %.1f%% of real time", modeName (m), rms, pk, cpu * 100);
    }
    // dreaming without input
    inst.params.layer = {}; inst.params.layer[0] = { true, 0, 1, 0, 0.3f }; inst.params.lucid = 0.0f;
    double pk, rms, cpu; run (4, false, &pk, &rms, &cpu);
    CHECK (rms > 1e-4, "dreams on after the input stops (rms %.4f)", rms);
    // combining: stack / morph / merge / imprint with three layers
    for (int c = 0; c < 4; ++c)
    {
        inst.params.layer = {};
        inst.params.layer[0] = { true, 0, 1, -0.3f, 0.3f }; inst.params.layer[1] = { true, 3, 1, 0.3f, 0.5f }; inst.params.layer[2] = { true, 6, 1, 0, 0.4f };
        inst.params.combine = c; inst.params.morphSource = 3; inst.params.morphRate = 0.5f; inst.params.lucid = 0.6f;
        const bool fin = run (3, true, &pk, &rms, &cpu);
        CHECK (fin && rms > 1e-4 && pk < 4, "%-7s harmonic + grains + strings: rms %.4f, cpu %.1f%%", combineName (c), rms, cpu * 100);
    }
    // ---- spatial: two timbres, each with its own stereo image; the spatial network should learn which goes with which
    {
        auto b2 = std::make_shared<Brain> (testDir ("ns_spat_test")); b2->settings.intensive = true;
        Instance s2 (b2); s2.prepare (sr, B); s2.params.layer[0] = { true, 0, 1, 0, 0.3f };
        std::mt19937 rng (4); std::normal_distribution<float> nd;
        int64_t p2 = 0;
        auto feed = [&] (double seconds)
        {
            for (int64_t k = 0; k < (int64_t) (seconds * sr / B); ++k)
            {
                const bool low = ((p2 / 48000) % 2) == 0;   // alternate each second
                for (int i = 0; i < B; ++i)
                {
                    const double t = (p2 + i) / sr;
                    if (low) { const float x = 0.3f * (float) std::sin (2 * M_PI * 110 * t); L[(size_t) i] = x; R[(size_t) i] = 0.1f * x; }           // dark tone, left, narrow
                    else { L[(size_t) i] = 0.15f * nd (rng); R[(size_t) i] = 0.15f * nd (rng); }                                                   // bright noise, wide
                }
                p2 += B;
                s2.process (L.data(), R.data(), oL.data(), oR.data(), B);
            }
        };
        feed (40); b2->trainFor (3000); feed (20); b2->trainFor (3000);
        // probe: ask what the network predicts while each kind is playing
        auto probe = [&] (bool low)
        {
            p2 = low ? 0 : 48000; float bal = 0, wid = 0; int n = 0;
            for (int k = 0; k < 150; ++k)
            {
                for (int i = 0; i < B; ++i) { const double t = (p2 + i) / sr; if (low) { const float x = 0.3f * (float) std::sin (2 * M_PI * 110 * t); L[(size_t) i] = x; R[(size_t) i] = 0.1f * x; } else { L[(size_t) i] = 0.15f * nd (rng); R[(size_t) i] = 0.15f * nd (rng); } }
                p2 = (low ? 0 : 48000) + ((k + 1) * B) % 40000;   // continuous audio, staying in the same kind
                s2.process (L.data(), R.data(), oL.data(), oR.data(), B);
                if (k > 50) { const auto t = s2.telemetry(); bal += t.spatPred[19]; wid += t.spatPred[20]; ++n; }
            }
            return std::pair<float, float> { bal / n, wid / n };
        };
        s2.params.lucid = 1.0f; s2.params.drift = 0.0f;   // follow the input exactly while probing
        const auto lo = probe (true), hi = probe (false);
        const auto tel = s2.telemetry();
        CHECK (b2->stats.spatCoverage > 0.45f && b2->stats.spatCoverage < 0.95f, "spatial σ is calibrated on held-out frames: %.0f%% inside ±1σ (Gaussian: 68%%), held-out NLL %.2f", b2->stats.spatCoverage * 100.0f, b2->stats.spatValNll.load());
        CHECK (tel.spatSteps > 100, "spatial network trained (%llu steps, NLL %.2f)", (unsigned long long) tel.spatSteps, b2->stats.spatNll.load());
        CHECK (lo.first > hi.first + 0.2f && hi.second > lo.second + 0.3f,
               "it learned which image goes with which timbre: tone → azimuth %+.2f diffuseness %.2f (measured +0.84, 0), noise → azimuth %+.2f diffuseness %.2f (true 0, ≈1)", lo.first, lo.second, hi.first, hi.second);
        // spatial render: 7th-order AmbiX and binaural, finite and non-silent
        for (int fmt : { 0, 1 })
        {
            s2.params.spatial = true; s2.params.order = 7; s2.params.format = fmt;
            const int nOut = fmt == 0 ? 64 : 2;
            std::vector<std::vector<float>> o ((size_t) nOut, std::vector<float> ((size_t) B)); std::vector<float*> op; for (auto& v : o) op.push_back (v.data());
            s2.setSpatialOutput (op.data(), nOut);
            double e = 0; bool fin = true; const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 400; ++k)
            {
                for (int i = 0; i < B; ++i) L[(size_t) i] = R[(size_t) i] = 0.2f * (float) std::sin (i * 0.05);
                s2.process (L.data(), R.data(), oL.data(), oR.data(), B);
                for (auto& v : o) for (float x : v) { if (! std::isfinite (x)) fin = false; e += x * x; }
            }
            const double cpu = std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count() / (400.0 * B / sr);
            CHECK (fin && e > 1e-3, "%s order 7: finite, energy %.2f, cpu %.1f%%", fmt == 0 ? "ambix" : "binaural", e, cpu * 100);
            s2.setSpatialOutput (nullptr, 0);
        }
    }
    std::printf (fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails;
}
