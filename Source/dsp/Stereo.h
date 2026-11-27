// Neurosonde — what the stereo image of the input is like, every 10 ms (24 spatial features):
//   0..7   interaural/inter-channel level difference per band (dB, 8 log bands 60 Hz .. 16 kHz)
//   8..15  inter-channel coherence per band (magnitude-squared coherence, smoothed cross-spectra)
//   16     width: side energy / (mid + side)          17  correlation (Pearson, time domain)
//   18     balance (dB, L − R)                         19  azimuth estimate (tangent pan law, −1 right .. +1 left)
//   20     diffuseness (1 − energy-weighted coherence)  21  motion (how fast the azimuth moves)
//   22     width tilt (high-band minus low-band width)  23  phase spread (mean |inter-channel phase|, low bands)
#pragma once
#include "Analysis.h"

namespace ns::dsp
{
constexpr int kSpat = 24;
static_assert (kSpat == ns::ml::kSpatial, "spatial feature count");
constexpr int kSpatBands = 8;
inline float spatBandEdge (int b) { return 60.0f * std::pow (16000.0f / 60.0f, (float) b / kSpatBands); }

class StereoAnalyser
{
public:
    static constexpr int N = 2048;
    void prepare (double sampleRate)
    {
        sr = (float) sampleRate; hop = std::max (64, (int) std::lround (sampleRate * 0.01));
        fft.init (N); win.resize (N); for (int i = 0; i < N; ++i) win[(size_t) i] = 0.5f - 0.5f * std::cos (2 * (float) M_PI * i / (N - 1));
        bl.assign (N, 0); br.assign (N, 0); xl.assign (N, 0); xr.assign (N, 0); XL.assign (N / 2 + 1, {}); XR = XL;
        Sll.fill (1e-9); Srr.fill (1e-9); Slr.fill ({ 0, 0 });
        pos = sinceHop = 0; prevAz = 0; motion = 0;
    }
    template <typename Cb> void push (const float* l, const float* r, int n, Cb&& onFrame)
    {
        for (int i = 0; i < n; ++i)
        {
            bl[(size_t) pos] = l[i]; br[(size_t) pos] = r[i]; pos = (pos + 1) % N;
            if (++sinceHop >= hop) { sinceHop = 0; analyse(); onFrame (f); }
        }
    }
    const std::array<float, kSpat>& features() const { return f; }

private:
    void analyse()
    {
        double sm = 0, ss = 0, slr = 0, sll = 0, srr = 0;
        for (int i = 0; i < N; ++i)
        {
            const float a = bl[(size_t) ((pos + i) % N)], b = br[(size_t) ((pos + i) % N)];
            xl[(size_t) i] = a * win[(size_t) i]; xr[(size_t) i] = b * win[(size_t) i];
            const double m = 0.5 * (a + b), s = 0.5 * (a - b);
            sm += m * m; ss += s * s; slr += (double) a * b; sll += (double) a * a; srr += (double) b * b;
        }
        fft.forward (xl.data(), XL.data()); fft.forward (xr.data(), XR.data());
        const float binHz = sr / N;
        double cohW = 0, eW = 0, wLo = 0, wHi = 0, eLo = 0, eHi = 0, ph = 0, phW = 0;
        const double a = 0.25;   // cross-spectral smoothing over frames (coherence needs averaging to be meaningful)
        for (int b = 0; b < kSpatBands; ++b)
        {
            const int k0 = std::max (1, (int) (spatBandEdge (b) / binHz)), k1 = std::min (N / 2, std::max (k0 + 1, (int) (spatBandEdge (b + 1) / binHz)));
            double el = 0, er = 0; std::complex<double> c (0, 0); double bandPh = 0, bandE = 0;
            for (int k = k0; k < k1; ++k)
            {
                const std::complex<double> L = XL[(size_t) k], R = XR[(size_t) k];
                el += std::norm (L); er += std::norm (R); c += L * std::conj (R);
                const double e = std::abs (L) * std::abs (R); bandPh += e * std::abs (std::arg (L * std::conj (R))); bandE += e;
            }
            Sll[(size_t) b] = (1 - a) * Sll[(size_t) b] + a * el; Srr[(size_t) b] = (1 - a) * Srr[(size_t) b] + a * er; Slr[(size_t) b] = (1 - a) * Slr[(size_t) b] + a * c;
            f[(size_t) b] = (float) std::clamp (10.0 * std::log10 ((el + 1e-12) / (er + 1e-12)), -24.0, 24.0);
            const double coh = std::norm (Slr[(size_t) b]) / (Sll[(size_t) b] * Srr[(size_t) b] + 1e-24);
            f[(size_t) (8 + b)] = (float) std::clamp (coh, 0.0, 1.0);
            cohW += coh * (el + er); eW += el + er;
            // band width: 1 − |coherence| weighted, split low/high
            const double w = 1.0 - std::sqrt (std::clamp (coh, 0.0, 1.0));
            if (b < 4) { wLo += w * (el + er); eLo += el + er; } else { wHi += w * (el + er); eHi += el + er; }
            if (b < 4) { ph += bandPh; phW += bandE; }
        }
        f[16] = (float) (ss / (sm + ss + 1e-12));
        f[17] = (float) (slr / std::sqrt (sll * srr + 1e-24));
        f[18] = (float) std::clamp (10.0 * std::log10 ((sll + 1e-12) / (srr + 1e-12)), -30.0, 30.0);
        // tangent law: tan(θ)/tan(θ0) = (gL − gR)/(gL + gR), θ0 = 30°
        const double gl = std::sqrt (sll), gr = std::sqrt (srr);
        const double t = (gl - gr) / (gl + gr + 1e-12);
        const float az = (float) (std::atan (t * std::tan (M_PI / 6)) / (M_PI / 6));
        f[19] = std::clamp (az, -1.0f, 1.0f);
        f[20] = (float) std::clamp (1.0 - cohW / (eW + 1e-12), 0.0, 1.0);
        motion = 0.9f * motion + 0.1f * std::abs (az - prevAz) * 100.0f; prevAz = az;
        f[21] = std::min (1.0f, motion);
        f[22] = (float) (wHi / (eHi + 1e-12) - wLo / (eLo + 1e-12));
        f[23] = (float) (ph / (phW + 1e-12) / M_PI);
        if (sll + srr < 1e-10) { for (int b = 0; b < kSpatBands; ++b) { f[(size_t) b] = 0; f[(size_t) (8 + b)] = 1; } f[16] = 0; f[17] = 1; f[18] = 0; f[19] = 0; f[20] = 0; f[22] = 0; f[23] = 0; }
    }
    float sr = 48000; int hop = 480, pos = 0, sinceHop = 0;
    RealFFT fft; std::vector<float> win, bl, br, xl, xr; std::vector<std::complex<float>> XL, XR;
    std::array<double, kSpatBands> Sll {}, Srr {}; std::array<std::complex<double>, kSpatBands> Slr {};
    std::array<float, kSpat> f {}; float prevAz = 0, motion = 0;
};
} // namespace ns::dsp
