// Neurosonde — frame analysis: 10 ms hop, 2048-point STFT. Produces the 84 features the network learns from
// (64 log-mel bands, 12 chroma, 8 scalars) and the DDSP-style targets for its synthesis heads: a harmonic
// distribution sampled at k·f0 and a 32-band noise residual.
#pragma once
#include "../ml/Models.h"
#include <complex>

namespace ns::dsp
{
using namespace ns::ml;

/** Real FFT of size N (power of two) through vDSP. */
struct RealFFT
{
    int N = 0, log2N = 0;
    FFTSetup setup = nullptr;
    std::vector<float> re, im;
    void init (int n)
    {
        N = n; log2N = (int) std::round (std::log2 ((double) n));
        if (setup) vDSP_destroy_fftsetup (setup);
        setup = vDSP_create_fftsetup ((vDSP_Length) log2N, kFFTRadix2);
        re.assign ((size_t) N / 2, 0); im.assign ((size_t) N / 2, 0);
    }
    ~RealFFT() { if (setup) vDSP_destroy_fftsetup (setup); }
    /** x[N] -> power spectrum pw[N/2+1] (unnormalised |X|^2 / N^2 scale) */
    void power (const float* x, float* pw)
    {
        DSPSplitComplex sc { re.data(), im.data() };
        vDSP_ctoz ((const DSPComplex*) x, 2, &sc, 1, (vDSP_Length) N / 2);
        vDSP_fft_zrip (setup, &sc, 1, (vDSP_Length) log2N, FFT_FORWARD);
        const float s = 1.0f / (float) N;     // vDSP scales the forward by 2
        pw[0] = (re[0] * s) * (re[0] * s);
        pw[N / 2] = (im[0] * s) * (im[0] * s);
        for (int k = 1; k < N / 2; ++k) { const float a = re[(size_t) k] * s, b = im[(size_t) k] * s; pw[k] = a * a + b * b; }
    }
    /** complex forward: X[k] for k=0..N/2 (DC/Nyquist real) */
    void forward (const float* x, std::complex<float>* X)
    {
        DSPSplitComplex sc { re.data(), im.data() };
        vDSP_ctoz ((const DSPComplex*) x, 2, &sc, 1, (vDSP_Length) N / 2);
        vDSP_fft_zrip (setup, &sc, 1, (vDSP_Length) log2N, FFT_FORWARD);
        const float s = 0.5f;
        X[0] = { re[0] * s, 0 }; X[N / 2] = { im[0] * s, 0 };
        for (int k = 1; k < N / 2; ++k) X[k] = { re[(size_t) k] * s, im[(size_t) k] * s };
    }
    /** complex inverse: X[0..N/2] -> x[N] (real), scaled so inverse(forward(x)) == x */
    void inverse (const std::complex<float>* X, float* x)
    {
        re[0] = X[0].real(); im[0] = X[N / 2].real();
        for (int k = 1; k < N / 2; ++k) { re[(size_t) k] = X[k].real(); im[(size_t) k] = X[k].imag(); }
        DSPSplitComplex sc { re.data(), im.data() };
        vDSP_fft_zrip (setup, &sc, 1, (vDSP_Length) log2N, FFT_INVERSE);
        vDSP_ztoc (&sc, 1, (DSPComplex*) x, 2, (vDSP_Length) N / 2);
        const float s = 1.0f / (float) N;
        vDSP_vsmul (x, 1, &s, x, 1, (vDSP_Length) N);
    }
};

inline float hzToMel (float f) { return 2595.0f * std::log10 (1.0f + f / 700.0f); }
inline float melToHz (float m) { return 700.0f * (std::pow (10.0f, m / 2595.0f) - 1.0f); }

/** Everything measured for one frame, in plain units. */
struct RawFrame
{
    float mel[kMel];            // dB
    float chroma[kChroma];      // sums to 1
    float f0 = 0;               // Hz (0 if unvoiced)
    float voicing = 0;          // 0..1
    float loud = -100;          // dBFS
    float centroid = 1000, spread = 500, flatness = 0, flux = 0, rolloff = 4000;   // Hz / ratio
    float harm[kHarm];          // distribution (sums to 1)
    float noise[kNoise];        // dB
    bool onset = false;
};

/** Converts a RawFrame into the network's feature vector (before standardisation) and back. */
inline void toFeatures (const RawFrame& r, float* x)
{
    for (int i = 0; i < kMel; ++i) x[i] = r.mel[i];
    for (int i = 0; i < kChroma; ++i) x[kMel + i] = r.chroma[i];
    x[kF0] = r.f0 > 0 ? std::log2 (r.f0 / 440.0f) : -3.0f;
    x[kVoicing] = r.voicing;
    x[kLoud] = r.loud;
    x[kCentroid] = std::log2 (std::max (20.0f, r.centroid) / 1000.0f);
    x[kSpread] = std::log2 (std::max (20.0f, r.spread) / 1000.0f);
    x[kFlatness] = std::log10 (std::max (1e-4f, r.flatness));
    x[kFlux] = r.flux;
    x[kRolloff] = std::log2 (std::max (20.0f, r.rolloff) / 1000.0f);
}

/** Bands for the 32-band noise envelope: log-spaced 40 Hz .. 20 kHz. */
inline float noiseBandEdge (int i) { return 40.0f * std::pow (500.0f, (float) i / kNoise); }

class Analyser
{
public:
    static constexpr int N = 2048;
    void prepare (double sampleRate)
    {
        sr = (float) sampleRate;
        hop = std::max (64, (int) std::lround (sampleRate * 0.01));
        fft.init (N); acf.init (2 * N);
        win.resize (N); for (int i = 0; i < N; ++i) win[(size_t) i] = 0.5f - 0.5f * std::cos (2 * (float) M_PI * i / (N - 1));
        pw.assign (N / 2 + 1, 0); prevMel.assign (kMel, -100);
        buf.assign (N, 0);
        // mel filterbank (triangular, 30 Hz .. min(16k, nyquist))
        melLo.resize (kMel); melHi.resize (kMel); melW.assign ((size_t) kMel, {});
        const float mLo = hzToMel (30.0f), mHi = hzToMel (std::min (16000.0f, sr * 0.48f));
        std::vector<float> c (kMel + 2);
        for (int i = 0; i < kMel + 2; ++i) c[(size_t) i] = melToHz (mLo + (mHi - mLo) * i / (kMel + 1));
        for (int b = 0; b < kMel; ++b)
        {
            const int k0 = std::max (1, (int) std::floor (c[(size_t) b] / sr * N)), k2 = std::min (N / 2, (int) std::ceil (c[(size_t) b + 2] / sr * N));
            melLo[(size_t) b] = k0; melHi[(size_t) b] = std::max (k0 + 1, k2);
            for (int k = k0; k < melHi[(size_t) b]; ++k)
            {
                const float f = k * sr / N;
                float w = f < c[(size_t) b + 1] ? (f - c[(size_t) b]) / (c[(size_t) b + 1] - c[(size_t) b]) : (c[(size_t) b + 2] - f) / (c[(size_t) b + 2] - c[(size_t) b + 1]);
                melW[(size_t) b].push_back (std::max (0.0f, w));
            }
        }
        lastF0 = 0; written = 0;
        resid.assign (N / 2 + 1, 0); sA.assign ((size_t) 2 * N, 0); sP.assign ((size_t) N + 1, 0);
        sX.assign ((size_t) N + 1, {}); sR.assign ((size_t) 2 * N, 0); sCm.assign ((size_t) N / 2 + 2, 1.0f);
    }
    int hopSize() const { return hop; }
    float sampleRate() const { return sr; }

    /** push mono samples; returns true each time a new hop completed (call frame() to read it) */
    template <typename Cb>
    void push (const float* x, int n, Cb&& onFrame)
    {
        for (int i = 0; i < n; ++i)
        {
            buf[(size_t) pos] = x[i]; pos = (pos + 1) % N;
            if (++sinceHop >= hop) { sinceHop = 0; analyse(); onFrame (cur); }
        }
    }
    const RawFrame& frame() const { return cur; }
    /** the current analysis window, oldest sample first (for grain capture etc.) */
    void window (float* out) const { for (int i = 0; i < N; ++i) out[i] = buf[(size_t) ((pos + i) % N)]; }

private:
    void analyse()
    {
        float x[N]; double ms = 0;
        for (int i = 0; i < N; ++i) { const float v = buf[(size_t) ((pos + i) % N)]; ms += (double) v * v; x[i] = v * win[(size_t) i]; }
        // loudness over the most recent ~20 ms (responsive) mixed with the window
        double rec = 0; const int R = std::min (N, 2 * hop);
        for (int i = N - R; i < N; ++i) { const float v = buf[(size_t) ((pos + i) % N)]; rec += (double) v * v; }
        cur.loud = (float) (10.0 * std::log10 (0.5 * (ms / N + rec / R) + 1e-12));
        fft.power (x, pw.data());
        // mel + flux
        float flux = 0;
        for (int b = 0; b < kMel; ++b)
        {
            double e = 0;
            for (int k = melLo[(size_t) b], j = 0; k < melHi[(size_t) b]; ++k, ++j) e += pw[(size_t) k] * melW[(size_t) b][(size_t) j];
            cur.mel[b] = (float) (10.0 * std::log10 (e + 1e-12));
            flux += std::max (0.0f, cur.mel[b] - prevMel[(size_t) b]);
            prevMel[(size_t) b] = cur.mel[b];
        }
        cur.flux = flux / kMel;
        cur.onset = cur.flux > onsetThresh && cur.loud > -55;
        onsetThresh = 0.97f * onsetThresh + 0.03f * std::max (2.0f, cur.flux * 1.6f);
        // centroid, spread, flatness, rolloff
        double sum = 0, wsum = 0, lsum = 0; int nb = 0;
        for (int k = 1; k <= N / 2; ++k) { const double p = pw[(size_t) k]; sum += p; wsum += p * k; lsum += std::log (p + 1e-14); ++nb; }
        const float binHz = sr / N;
        cur.centroid = sum > 0 ? (float) (wsum / sum) * binHz : 1000;
        double sp = 0; for (int k = 1; k <= N / 2; ++k) { const double d = k * binHz - cur.centroid; sp += pw[(size_t) k] * d * d; }
        cur.spread = sum > 0 ? (float) std::sqrt (sp / sum) : 500;
        cur.flatness = sum > 0 ? (float) (std::exp (lsum / nb) / (sum / nb)) : 1;
        double acc = 0; cur.rolloff = sr / 2;
        for (int k = 1; k <= N / 2; ++k) { acc += pw[(size_t) k]; if (acc >= 0.85 * sum) { cur.rolloff = k * binHz; break; } }
        pitch();
        // chroma (60 Hz .. 5 kHz)
        std::fill (cur.chroma, cur.chroma + kChroma, 0.0f);
        double cs = 0;
        float pmax = 0; for (int k = 2; k < N / 2; ++k) pmax = std::max (pmax, pw[(size_t) k]);
        for (int k = std::max (2, (int) (60 / binHz)); k <= std::min (N / 2 - 1, (int) (5000 / binHz)); ++k)
        {
            const float pk = pw[(size_t) k];
            if (! (pk > pw[(size_t) k - 1] && pk >= pw[(size_t) k + 1] && pk > pmax * 1e-4f)) continue;
            // parabolic interpolation of the peak on log power
            const float l0 = std::log (pw[(size_t) k - 1] + 1e-20f), l1 = std::log (pk + 1e-20f), l2 = std::log (pw[(size_t) k + 1] + 1e-20f);
            const float den = l0 - 2 * l1 + l2, d = std::abs (den) > 1e-9f ? 0.5f * (l0 - l2) / den : 0.0f;
            const float f = (k + std::clamp (d, -0.5f, 0.5f)) * binHz;
            const float midi = 69 + 12 * std::log2 (f / 440.0f);
            const int pc = ((int) std::lround (midi) % 12 + 12) % 12;
            const float m = std::sqrt (pk);
            cur.chroma[pc] += m; cs += m;
        }
        for (auto& c : cur.chroma) c = cs > 0 ? c / (float) cs : 1.0f / kChroma;
        // harmonic distribution + noise residual
        std::copy (pw.begin(), pw.end(), resid.begin());
        std::fill (cur.harm, cur.harm + kHarm, 0.0f);
        if (cur.f0 > 0)
        {
            double hs = 0;
            for (int h = 1; h <= kHarm; ++h)
            {
                const float fk = cur.f0 * h; if (fk >= sr * 0.48f) break;
                const int k = (int) std::lround (fk / binHz);
                float best = 0; int bk = k;
                for (int j = std::max (1, k - 2); j <= std::min (N / 2, k + 2); ++j) if (pw[(size_t) j] > best) { best = pw[(size_t) j]; bk = j; }
                cur.harm[h - 1] = std::sqrt (best); hs += cur.harm[h - 1];
                for (int j = std::max (1, bk - 3); j <= std::min (N / 2, bk + 3); ++j) resid[(size_t) j] *= 0.02f;
            }
            for (auto& h : cur.harm) h = hs > 0 ? h / (float) hs : 1.0f / kHarm;
        }
        else for (auto& h : cur.harm) h = 1.0f / kHarm;
        for (int b = 0; b < kNoise; ++b)
        {
            const int k0 = std::max (1, (int) (noiseBandEdge (b) / binHz)), k1 = std::min (N / 2, std::max (k0 + 1, (int) (noiseBandEdge (b + 1) / binHz)));
            double e = 0; for (int k = k0; k < k1; ++k) e += resid[(size_t) k];
            cur.noise[b] = (float) (10.0 * std::log10 (e / (k1 - k0) + 1e-14));
        }
        ++written;
    }
    /** YIN (de Cheveigné & Kawahara 2002) on the autocorrelation computed by FFT. */
    void pitch()
    {
        const int W = N;   // window length
        auto& a = sA; auto& p = sP; auto& X = sX; auto& r = sR; auto& cm = sCm;
        std::fill (a.begin(), a.end(), 0.0f);
        for (int i = 0; i < W; ++i) a[(size_t) i] = buf[(size_t) ((pos + i) % N)];
        acf.power (a.data(), p.data());
        // inverse of the power spectrum = autocorrelation (via a real IFFT through the complex path)
        for (int k = 0; k <= N; ++k) X[(size_t) k] = { p[(size_t) k], 0 };
        acf.inverse (X.data(), r.data());
        const int tMin = std::max (2, (int) (sr / 1000.0f)), tMax = std::min (N / 2, (int) (sr / 50.0f));
        // d(t) ≈ 2 (r0 - r(t)) with energy correction; cumulative mean normalised difference
        std::fill (cm.begin(), cm.end(), 1.0f);
        double run = 0; const float r0 = r[0];
        if (r0 <= 1e-12f) { cur.f0 = 0; cur.voicing = 0; return; }
        float best = 1; int bt = -1;
        for (int t = 1; t <= tMax; ++t)
        {
            const float d = std::max (0.0f, 2 * (r0 - r[(size_t) t] * (float) W / (float) (W - t)));
            run += d;
            cm[(size_t) t] = run > 0 ? d * t / (float) run : 1;
            if (t >= tMin && bt < 0 && cm[(size_t) t] < 0.15f)
            {
                while (t + 1 <= tMax)
                {
                    const float dn = std::max (0.0f, 2 * (r0 - r[(size_t) t + 1] * (float) W / (float) (W - t - 1)));
                    const float cn = (run + dn) > 0 ? dn * (t + 1) / (float) (run + dn) : 1;
                    if (cn < cm[(size_t) t]) { ++t; run += dn; cm[(size_t) t] = cn; } else break;
                }
                bt = t; best = cm[(size_t) t];
            }
        }
        if (bt < 0)
        {
            for (int t = tMin; t <= tMax; ++t) if (cm[(size_t) t] < best) { best = cm[(size_t) t]; bt = t; }
        }
        cur.voicing = std::clamp (1.0f - best, 0.0f, 1.0f);
        if (bt > 1 && bt < tMax && cur.voicing > 0.6f && cur.loud > -60)
        {
            const float a0 = cm[(size_t) bt - 1], a1 = cm[(size_t) bt], a2 = cm[(size_t) bt + 1];
            const float den = a0 - 2 * a1 + a2;
            const float shift = std::abs (den) > 1e-9f ? 0.5f * (a0 - a2) / den : 0;
            cur.f0 = sr / (bt + std::clamp (shift, -0.5f, 0.5f));
            lastF0 = cur.f0;
        }
        else { cur.f0 = 0; cur.voicing = std::min (cur.voicing, 0.5f); }
    }

    float sr = 48000; int hop = 480;
    RealFFT fft, acf;
    std::vector<float> win, pw, prevMel, buf, resid, sA, sP, sR, sCm;
    std::vector<std::complex<float>> sX;
    std::vector<int> melLo, melHi; std::vector<std::vector<float>> melW;
    int pos = 0, sinceHop = 0; long written = 0;
    float lastF0 = 0, onsetThresh = 3;
    RawFrame cur;
};
} // namespace ns::dsp
