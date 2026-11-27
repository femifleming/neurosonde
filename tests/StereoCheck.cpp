#include "../Source/dsp/Stereo.h"
#include <cstdio>
using namespace ns::dsp;
int main()
{
    int fails = 0;
    auto run = [&] (const char* name, auto gen, auto check)
    {
        StereoAnalyser a; a.prepare (48000); std::vector<float> l (48000), r (48000);
        for (int i = 0; i < 48000; ++i) gen (i, l[(size_t) i], r[(size_t) i]);
        std::array<float, kSpat> f {}; a.push (l.data(), r.data(), 48000, [&] (const auto& x) { f = x; });
        const bool ok = check (f);
        std::printf ("  %s %-14s width %.2f corr %+.2f bal %+.1f az %+.2f diffuse %.2f coh(b3) %.2f ild(b3) %+.1f\n", ok ? "ok  " : "FAIL", name, f[16], f[17], f[18], f[19], f[20], f[11], f[3]);
        fails += ok ? 0 : 1;
    };
    std::mt19937 rng (1); std::normal_distribution<float> nd;
    run ("mono centre", [&] (int, float& l, float& r) { l = r = 0.3f * nd (rng); }, [] (auto& f) { return f[16] < 0.01f && f[17] > 0.99f && std::abs (f[19]) < 0.02f && f[20] < 0.05f; });
    run ("hard left", [&] (int, float& l, float& r) { l = 0.3f * nd (rng); r = 0.03f * l; }, [] (auto& f) { return f[18] > 15 && f[19] > 0.8f && f[3] > 15; });
    run ("panned right", [&] (int, float& l, float& r) { const float x = 0.3f * nd (rng); l = 0.5f * x; r = x; }, [] (auto& f) { return f[19] < -0.2f && f[18] < -4; });
    run ("decorrelated", [&] (int, float& l, float& r) { l = 0.3f * nd (rng); r = 0.3f * nd (rng); }, [] (auto& f) { return f[16] > 0.4f && std::abs (f[17]) < 0.1f && f[20] > 0.7f; });
    std::printf (fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails;
}
