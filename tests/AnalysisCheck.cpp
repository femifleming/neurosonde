#include "../Source/dsp/Analysis.h"
#include <cstdio>
using namespace ns::dsp;
int main()
{
    int fails = 0;
    auto run = [&] (const char* name, auto gen, auto check)
    {
        Analyser a; a.prepare (48000.0);
        std::vector<float> x (48000); for (int i = 0; i < 48000; ++i) x[(size_t) i] = gen (i);
        RawFrame last; a.push (x.data(), (int) x.size(), [&] (const RawFrame& f) { last = f; });
        const bool ok = check (last);
        std::printf ("  %s %-10s f0 %.1f voicing %.2f loud %.1f centroid %.0f flat %.3f harm1 %.2f harm2 %.2f\n", ok ? "ok  " : "FAIL", name, last.f0, last.voicing, last.loud, last.centroid, last.flatness, last.harm[0], last.harm[1]);
        fails += ok ? 0 : 1;
    };
    run ("sine 220", [] (int i) { return 0.5f * std::sin (2 * M_PI * 220 * i / 48000.0); },
         [] (const RawFrame& f) { return std::abs (f.f0 - 220) < 2 && f.voicing > 0.9f && f.harm[0] > 0.8f; });
    run ("saw 110", [] (int i) { double ph = std::fmod (110.0 * i / 48000.0, 1.0); return (float) (0.4 * (2 * ph - 1)); },
         [] (const RawFrame& f) { return std::abs (f.f0 - 110) < 1.5 && f.harm[0] > f.harm[1] && f.harm[1] > f.harm[3]; });
    std::mt19937 rng (1); std::uniform_real_distribution<float> u (-1, 1);
    run ("noise", [&] (int) { return 0.3f * u (rng); },
         [] (const RawFrame& f) { return f.f0 == 0 && f.flatness > 0.3f; });
    run ("a440 chord", [] (int i) { double t = i / 48000.0; return (float) (0.2 * (std::sin (2 * M_PI * 440 * t) + std::sin (2 * M_PI * 554.37 * t) + std::sin (2 * M_PI * 659.26 * t))); },
         [] (const RawFrame& f) { for (int i = 0; i < 12; ++i) std::printf ("%.2f ", f.chroma[i]); std::printf ("\n"); return f.chroma[9] > 0.2f && f.chroma[1] > 0.15f && f.chroma[4] > 0.15f; });
    std::printf (fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails;
}
