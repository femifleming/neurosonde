#include "../Source/dsp/Spatial.h"
#include <cstdio>
using namespace ns::dsp;
static int fails = 0;
#define CHECK(c, ...) do { std::printf ((c) ? "  ok   " : "  FAIL "); std::printf (__VA_ARGS__); std::printf ("\n"); if (! (c)) ++fails; } while (0)
struct Out { std::vector<std::vector<float>> ch; };
static Out render (OutFormat f, int order, float azDeg, float elDeg, int nOut, float early = 0, float late = 0, float width = 0.0f)
{
    SpatialScene sc; sc.setOrder (order, f, true); sc.prepare (48000, 512); sc.setOrder (order, f, true);
    sc.early = early; sc.late = late;
    std::mt19937 rng (3); std::normal_distribution<float> nd;
    std::vector<float> L (512), R (512);
    std::array<SourceCtl, SpatialScene::kSources> ctl {}; ctl[0].az = azDeg * (float) M_PI / 180; ctl[0].el = elDeg * (float) M_PI / 180; ctl[0].dist = 1; ctl[0].width = width; ctl[0].send = late > 0 ? 0.5f : 0.0f;
    Out o; o.ch.assign ((size_t) nOut, {});
    std::vector<std::vector<float>> blk ((size_t) nOut, std::vector<float> (512)); std::vector<float*> ptr; for (auto& b : blk) ptr.push_back (b.data());
    for (int b = 0; b < 60; ++b)
    {
        for (int i = 0; i < 512; ++i) L[(size_t) i] = R[(size_t) i] = 0.2f * nd (rng);
        std::array<const float*, SpatialScene::kSources> l { L.data() }, r { R.data() };
        sc.process (l, r, ctl, ptr.data(), nOut, 512);
        if (b >= 10) for (int c = 0; c < nOut; ++c) o.ch[(size_t) c].insert (o.ch[(size_t) c].end(), blk[(size_t) c].begin(), blk[(size_t) c].end());
    }
    return o;
}
static double rms (const std::vector<float>& x) { double e = 0; for (float v : x) e += v * v; return std::sqrt (e / x.size()); }
static double db (double x) { return 20 * std::log10 (x + 1e-12); }
// lag (samples) of y relative to x by cross-correlation (positive: y later)
static int lag (const std::vector<float>& x, const std::vector<float>& y)
{
    int best = 0; double bv = -1e30;
    for (int l = -40; l <= 40; ++l) { double s = 0; for (size_t i = 64; i + 64 < x.size(); ++i) s += x[i] * y[(size_t) ((int) i + l)]; if (s > bv) { bv = s; best = l; } }
    return best;
}
int main()
{
    // binaural: a source on the left is louder and earlier in the left ear
    auto left = render (OutFormat::Binaural, 3, 90, 0, 2), right = render (OutFormat::Binaural, 3, -90, 0, 2), front = render (OutFormat::Binaural, 3, 0, 0, 2);
    const double ildL = db (rms (left.ch[0])) - db (rms (left.ch[1])), ildR = db (rms (right.ch[0])) - db (rms (right.ch[1])), ildF = db (rms (front.ch[0])) - db (rms (front.ch[1]));
    CHECK (ildL > 3 && ildR < -3 && std::abs (ildF) < 1.0, "binaural ILD: left source %+.1f dB, right %+.1f dB, front %+.1f dB", ildL, ildR, ildF);
    CHECK (std::abs (db (rms (front.ch[0])) - db (0.2)) < 4.0, "binaural level: a frontal source reaches each ear at %+.1f dB re. the input", db (rms (front.ch[0])) - db (0.2));
    const int itdL = lag (left.ch[0], left.ch[1]);
    CHECK (itdL > 5 && itdL < 40, "binaural ITD: a left source reaches the right ear %d samples later (%.2f ms; Woodworth ~0.66 ms max)", itdL, itdL / 48.0);
    // ambix: first order, ACN 1 = Y (left), ACN 3 = X (front)
    auto ax = render (OutFormat::Ambix, 1, 90, 0, 4);
    double c1 = 0, c3 = 0; for (size_t i = 0; i < ax.ch[0].size(); ++i) { c1 += ax.ch[0][i] * ax.ch[1][i]; c3 += ax.ch[0][i] * ax.ch[3][i]; }
    CHECK (c1 > 0 && std::abs (c3) < std::abs (c1) * 0.05, "ambix: a left source is in Y (ACN 1), not X (corr W·Y %+.3g, W·X %+.3g)", c1, c3);
    // stereo decode: a source at +30° is in the left speaker
    auto st = render (OutFormat::Stereo, 3, 30, 0, 2);
    CHECK (db (rms (st.ch[0])) - db (rms (st.ch[1])) > 6, "stereo decode: +30° source %.1f dB louder on the left", db (rms (st.ch[0])) - db (rms (st.ch[1])));
    // max-rE decoding keeps the loudness steady as a source travels round (order 3, 8 directions)
    double lo = 1e9, hi = -1e9;
    for (int a = 0; a < 360; a += 45) { auto q = render (OutFormat::Binaural, 3, (float) a, 0, 2); const double e = db (std::sqrt (rms (q.ch[0]) * rms (q.ch[0]) + rms (q.ch[1]) * rms (q.ch[1]))); lo = std::min (lo, e); hi = std::max (hi, e); }
    CHECK (hi - lo < 4.5, "binaural loudness round the head varies %.1f dB (head shadow: measured HRTFs vary 3-4 dB broadband)", hi - lo);
    // 7th order 7.1.4
    auto s714 = render (OutFormat::Surround714, 7, 45, 45, 12);
    int loud = 0; double m = 0; for (int c = 0; c < 12; ++c) if (rms (s714.ch[(size_t) c]) > m) { m = rms (s714.ch[(size_t) c]); loud = c; }
    CHECK (loud == 8 && rms (s714.ch[3]) == 0, "7.1.4: a source up-left-front is loudest in Ltf (channel %d), LFE silent", loud);
    // room: reflections and tail add energy after the direct sound stops
    auto dry = render (OutFormat::Binaural, 3, 0, 0, 2, 0, 0), wet = render (OutFormat::Binaural, 3, 0, 0, 2, 0.6f, 0.6f);
    CHECK (rms (wet.ch[0]) > rms (dry.ch[0]) * 1.12, "early reflections + late tail (send 0.5) add energy (%.1f dB)", db (rms (wet.ch[0])) - db (rms (dry.ch[0])));
    std::printf (fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails;
}
// (timing harness, compiled with -DTIMING)
