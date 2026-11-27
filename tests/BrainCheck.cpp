#include "../Source/ml/Brain.h"
#include <cstdio>
using namespace ns::ml; using namespace ns::dsp;
static int fails = 0;
#define CHECK(c, ...) do { std::printf ((c) ? "  ok   " : "  FAIL "); std::printf (__VA_ARGS__); std::printf ("\n"); if (! (c)) ++fails; } while (0)

// analyse a signal into feed items
static std::vector<FeedItem> feed (std::function<float (int)> gen, int seconds, const char* notes)
{
    Analyser a; a.prepare (48000);
    std::vector<FeedItem> out; std::vector<float> x (48000 * seconds);
    for (size_t i = 0; i < x.size(); ++i) x[i] = gen ((int) i);
    const auto c = TextEmbed::embed (notes);
    a.push (x.data(), (int) x.size(), [&] (const RawFrame& r)
    {
        FeedItem it; toFeatures (r, it.x); std::copy (r.harm, r.harm + kHarm, it.harm); std::copy (r.noise, r.noise + kNoise, it.noise);
        std::copy (c.begin(), c.end(), it.cond); it.voiced = r.voicing; out.push_back (it);
    });
    return out;
}

int main()
{
    const std::string dir = "/private/tmp/ns_brain_test";
    std::system (("rm -rf " + dir).c_str());
    std::mt19937 rng (1); std::uniform_real_distribution<float> u (-1, 1);
    auto tone = feed ([] (int i) { double t = i / 48000.0; return (float) (0.3 * std::sin (2 * M_PI * 220 * t) + 0.15 * std::sin (2 * M_PI * 440 * t)); }, 6, "warm tungsten organ");
    auto hiss = feed ([&] (int) { return 0.2f * u (rng); }, 6, "dust and rain");
    {
        Brain b (dir);
        auto ring = b.addFeeder();
        // interleave the two sounds the way a performance would
        size_t i = 0, j = 0;
        for (int round = 0; round < 40; ++round)
        {
            for (int k = 0; k < 30 && i < tone.size(); ++k) ring->push (tone[i++ % tone.size()]);
            for (int k = 0; k < 30 && j < hiss.size(); ++k) ring->push (hiss[j++ % hiss.size()]);
            b.trainFor (30);
            if (i >= tone.size()) i = 0; if (j >= hiss.size()) j = 0;
        }
        b.settings.intensive = true; b.trainFor (2000);
        CHECK (b.stats.vaeSteps > 1500, "trained %llu VAE steps, %llu GRU steps, %llu frames learned", (unsigned long long) b.stats.vaeSteps.load(), (unsigned long long) b.stats.gruSteps.load(), (unsigned long long) b.stats.framesLearned.load());
        CHECK (b.stats.recon < 0.35f, "reconstruction error of standardised features %.3f (1.0 = knows nothing)", b.stats.recon.load());
        CHECK (b.stats.holdout > 50 && b.stats.valRecon > 0 && b.stats.valRecon < 0.35f && b.stats.valRecon < 3 * b.stats.trainRecon + 0.02f,
               "held out %d frames: eval recon train %.4f, validation %.4f (it generalises, not memorises)", b.stats.holdout.load(), b.stats.trainRecon.load(), b.stats.valRecon.load());
        CHECK (b.stats.activeUnits >= 1 && b.stats.activeUnits <= kZ, "%d of %d latent units active (Cov E_q[z] > 0.01; two sounds need few)", b.stats.activeUnits.load(), kZ);
        // inference snapshot separates the two sounds
        const Snapshot& s = b.acquire();
        auto enc = [&] (const FeedItem& it) { std::array<float, kZ> z; float x[kFeat]; s.feat.apply (it.x, x); s.vae.encode (x, it.cond, z.data()); return z; };
        auto za = enc (tone[200]), zb = enc (hiss[200]), za2 = enc (tone[300]);
        double dab = 0, daa = 0; for (int k = 0; k < kZ; ++k) { dab += (za[k] - zb[k]) * (za[k] - zb[k]); daa += (za[k] - za2[k]) * (za[k] - za2[k]); }
        CHECK (dab > 10 * daa, "snapshot: tone vs hiss latent distance %.2f, tone vs tone %.3f", dab, daa);
        const int na = s.som.bmu (za.data()), nb = s.som.bmu (zb.data());
        CHECK (na != nb, "constellation: tone and hiss land on different stars (%d, %d)", na, nb);
        b.release();
        b.setNoteWords ({ "tungsten" });
        for (int k = 0; k < 50; ++k) ring->push (tone[(size_t) k + 100]);
        b.trainFor (5);
        auto w = b.starWords (b.stats.lastBmu);
        CHECK (! w.empty() && w[0].first == "tungsten", "stars are named after the patch notes ('%s')", w.empty() ? "" : w[0].first.c_str());
        b.remember ("first night");
        b.sync();
    }
    // two "processes" sharing the folder: both start from the file, learn different things, both sync
    {
        Brain a (dir), b (dir);
        const uint64_t start = a.stats.vaeSteps;
        CHECK (start > 1500, "a new brain loads the shared file (%llu steps)", (unsigned long long) start);
        auto ra = a.addFeeder(), rb = b.addFeeder();
        for (int k = 0; k < 400; ++k) { ra->push (tone[(size_t) k]); rb->push (hiss[(size_t) k]); }
        a.settings.intensive = true; b.settings.intensive = true;
        a.trainFor (300); b.trainFor (300);
        a.sync(); b.sync();
        const uint64_t merged = b.stats.vaeSteps;
        CHECK (merged >= start + 550, "federated merge: both instances' learning is in the file (%llu -> %llu steps)", (unsigned long long) start, (unsigned long long) merged);
        CHECK (b.stats.merges >= 1, "merges counted (%d)", b.stats.merges.load());
        auto mem = a.memories();
        CHECK (mem.size() == 1 && mem[0].label == "first night", "memories are listed (%zu, '%s')", mem.size(), mem.empty() ? "" : mem[0].label.c_str());
        CHECK (a.recall (mem[0].file) && a.stats.vaeSteps < merged, "recall goes back to an earlier brain (%llu steps)", (unsigned long long) a.stats.vaeSteps.load());
    }
    std::printf (fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails;
}
