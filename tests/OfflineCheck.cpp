// offline training: epochs, a block split, early stopping, best-weights restore, and that it really learns
#include "../Source/ml/Brain.h"
#include <memory>
#include "TestDir.h"
#include <cstdio>
using namespace ns::ml; using namespace ns::dsp;
static int fails = 0;
#define CHECK(c, ...) do { std::printf ((c) ? "  ok   " : "  FAIL "); std::printf (__VA_ARGS__); std::printf ("\n"); if (! (c)) ++fails; } while (0)
static std::vector<FeedItem> feed (std::function<float (int)> gen, int seconds, const char* notes)
{
    Analyser a; a.prepare (48000);
    std::vector<FeedItem> out; std::vector<float> x ((size_t) (48000 * seconds));
    for (size_t i = 0; i < x.size(); ++i) x[i] = gen ((int) i);
    const auto c = TextEmbed::embed (notes);
    a.push (x.data(), (int) x.size(), [&] (const RawFrame& r)
    {
        FeedItem it; toFeatures (r, it.x); std::copy (r.harm, r.harm + kHarm, it.harm); std::copy (r.noise, r.noise + kNoise, it.noise);
        std::copy (c.begin(), c.end(), it.cond); it.voiced = r.voicing; it.learn = r.loud > -60; out.push_back (it);
    });
    return out;
}
int main()
{
    testDir ("ns_offline_test"); testDir ("ns_offline_test2");
    std::mt19937 rng (3); std::uniform_real_distribution<float> u (-1, 1);
    // a small "album": a melody, a drone with beating, filtered noise swells
    std::vector<std::vector<FeedItem>> files;
    files.push_back (feed ([] (int i) { const double t = i / 48000.0; const int n = (int) (t * 3) % 6; const double f = 220 * std::pow (2.0, (std::array<int, 6> { 0, 2, 4, 7, 9, 12 }[(size_t) n]) / 12.0);
                                         return (float) (0.3 * std::exp (-std::fmod (t, 1 / 3.0) * 4) * std::sin (2 * M_PI * f * t)); }, 20, "a small melody"));
    files.push_back (feed ([] (int i) { const double t = i / 48000.0; return (float) (0.2 * std::sin (2 * M_PI * 110 * t) + 0.2 * std::sin (2 * M_PI * 110.7 * t) + 0.05 * std::sin (2 * M_PI * 330 * t)); }, 20, "drone, beating"));
    files.push_back (feed ([&] (int i) { const double t = i / 48000.0; return (float) (0.25 * (0.5 + 0.5 * std::sin (t * 1.3)) * u (rng)); }, 20, "noise swells"));
    auto bp = std::make_unique<Brain> (testDir ("ns_offline_test", false)); Brain& b = *bp;   // (heap: a Brain is ~0.5 MB)
    b.settings.frozen = true;   // offline mode: nothing learned from playback
    const auto t0 = std::chrono::steady_clock::now();
    b.trainOffline (files, 40, 5);
    const double secs = std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count();
    auto& o = b.offline;
    CHECK (o.state == 3, "finished (state %d) in %.1f s: %d epochs, best at epoch %d", o.state.load(), secs, o.epoch.load(), o.bestEpoch.load());
    CHECK (o.valFrames > 300 && o.trainFrames > 9 * o.valFrames / 2, "block split: %d training frames, %d validation frames", o.trainFrames.load(), o.valFrames.load());
    CHECK (o.valCurve[0] > 1.5f * o.bestVal, "validation loss fell: epoch 1 %.4f -> best %.4f", o.valCurve[0], o.bestVal.load());
    CHECK (o.bestVal < 0.2f && o.bestVal < 3 * o.trainLoss + 0.03f, "it generalises: best validation %.4f, train %.4f", o.bestVal.load(), o.trainLoss.load());
    CHECK (o.epoch == 40 || o.epoch - o.bestEpoch >= 5, "early stopping honoured (stopped at %d, best %d)", o.epoch.load(), o.bestEpoch.load());
    // the restored weights are the best ones: evaluate the validation loss on the published snapshot and compare
    {
        const Snapshot& S = b.acquire();
        double va = 0; int n = 0;
        int block = 0;
        for (auto& fl : files) { for (size_t i = 0; i < fl.size(); ++i) { if (i > 0 && i % 200 == 0) ++block; if (block % 10 != 9 || ! fl[i].learn) continue;
            float x[kFeat], mu[kZ], out[kOut]; S.feat.apply (fl[i].x, x); for (auto& v : x) v = std::clamp (v, -6.0f, 6.0f); S.vae.encode (x, fl[i].cond, mu); S.vae.decode (mu, fl[i].cond, out);
            double e = 0; for (int k = 0; k < kFeat; ++k) e += (out[k] - x[k]) * (out[k] - x[k]); va += e / kFeat; ++n; } ++block; }
        b.release();
        CHECK (n > 0 && std::abs (va / n - o.bestVal) < 0.25 * o.bestVal + 0.005, "the published network is the best epoch's (validation %.4f vs recorded best %.4f)", va / n, o.bestVal.load());
    }
    // frozen: playback does not change the network
    const auto steps = b.stats.vaeSteps.load();
    auto ring = b.addFeeder(); for (int k = 0; k < 400; ++k) ring->push (files[0][(size_t) k]);
    for (int k = 0; k < 50; ++k) std::this_thread::sleep_for (std::chrono::milliseconds (5));
    CHECK (b.stats.vaeSteps == steps, "offline mode is frozen while playing (%llu steps before and after)", (unsigned long long) steps);
    // cancel
    auto cp = std::make_unique<Brain> (testDir ("ns_offline_test2", false)); Brain& c = *cp;
    std::thread t ([&] { std::this_thread::sleep_for (std::chrono::milliseconds (300)); c.offline.cancel = true; });
    c.trainOffline (files, 200, 50); t.join();
    CHECK (c.offline.state == 4 && c.offline.epoch < 200, "cancel stops it (epoch %d, state %d)", c.offline.epoch.load(), c.offline.state.load());
    std::printf (fails ? "%d FAILED\n" : "all passed\n", fails);
    testDir ("ns_offline_test"); testDir ("ns_offline_test2");
    return fails;
}
