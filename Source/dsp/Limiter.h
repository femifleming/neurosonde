// Neurosonde — gain staging: the output limiter and the meters.
//
// A look-ahead peak limiter, linked across every channel (2 in Neurosonde, up to 64 in Space). For each sample the
// gain that would keep the loudest channel under the ceiling is taken; a minimum filter over the look-ahead window,
// a release, and a moving average over the same window give the gain applied to the signal delayed by that window.
// Because every gain the average sees around a peak is already at or below what that peak needs, no sample leaves
// above the ceiling, and the gain never steps (no clicks). Latency: the window, about 1.3 ms.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace ns::dsp
{
struct Limiter
{
    enum Mode { Clean = 0, Soft = 1 };     // always on: no sample leaves above the ceiling
    int L = 64, nCh = 2, pos = 0;
    std::vector<float> delay;          // nCh x L
    std::vector<float> req;            // L
    float env = 1, boxSum = 0, relCoef = 0.0005f;
    std::vector<float> box;            // L
    int boxPos = 0;
    // what the meters read, since the last take()
    float inPeak = 0, outPeak = 0, minGain = 1; long overs = 0, outOvers = 0;

    void prepare (float sr, int channels)
    {
        L = std::max (8, (int) std::lround (0.00133 * sr));
        nCh = std::max (1, channels);
        delay.assign ((size_t) nCh * (size_t) L, 0.0f);
        req.assign ((size_t) L, 1.0f); box.assign ((size_t) L, 1.0f); boxSum = (float) L;
        pos = 0; boxPos = 0; env = 1;
        relCoef = 1 - std::exp (-1.0f / (0.08f * sr));          // 80 ms release
    }
    int latency() const { return L - 1; }

    /** ch[c][i], c < channels, n samples, in place. ceiling in linear gain. */
    void process (float* const* ch, int channels, int n, float ceiling, int mode)
    {
        channels = std::min (channels, nCh);
        const float soft = mode == Soft ? ceiling : 0;
        for (int i = 0; i < n; ++i)
        {
            float pk = 0;
            for (int c = 0; c < channels; ++c)
            {
                float x = ch[c][i];
                if (! std::isfinite (x)) x = 0;
                if (soft > 0) x = soft * std::tanh (x / soft);     // gentle saturation into the ceiling
                ch[c][i] = x;
                pk = std::max (pk, std::abs (x));
            }
            inPeak = std::max (inPeak, pk);
            if (pk > ceiling) ++overs;
            // the gain this sample needs, the smallest over the window, released, then averaged over the window
            req[(size_t) pos] = pk > ceiling ? ceiling / pk : 1.0f;
            float m = 1; for (int k = 0; k < L; ++k) m = std::min (m, req[(size_t) k]);
            env = m < env ? m : env + (m - env) * relCoef;
            boxSum += env - box[(size_t) boxPos]; box[(size_t) boxPos] = env; boxPos = (boxPos + 1) % L;
            float g = boxSum / (float) L;
            minGain = std::min (minGain, g);
            // the delayed sample: L - 1 samples ago
            const int rd = (pos + 1) % L;
            for (int c = 0; c < channels; ++c)
            {
                float* d = &delay[(size_t) c * (size_t) L];
                const float x = ch[c][i];
                float y = d[rd] * g;
                d[pos] = x;
                y = std::clamp (y, -ceiling, ceiling);    // rounding only: the average already holds it
                if (std::abs (y) > 1.0f) ++outOvers;
                outPeak = std::max (outPeak, std::abs (y));
                ch[c][i] = y;
            }
            pos = (pos + 1) % L;
            // the running sum drifts with rounding: rebuild it now and then
            if (boxPos == 0) { double s = 0; for (float v : box) s += v; boxSum = (float) s; }
        }
    }
};

/** the meters the gain page reads: peaks and loudness at every stage, gain reduction, overs */
struct GainMeters
{
    float inPeak = 0, inSq = 0;        // the input, after input gain
    float wetPeak = 0;                 // the synthesis, after output gain and auto gain, before the mix
    float preLimPeak = 0;              // what reaches the limiter (wet + dry)
    float outPeak = 0, outSq = 0;      // what leaves
    float minGain = 1;                 // the limiter's deepest gain
    long overs = 0, outOvers = 0;      // samples above the ceiling before the limiter / above 0 dBFS after it
    long n = 0;
    void reset() { *this = GainMeters(); }
};
} // namespace ns::dsp
