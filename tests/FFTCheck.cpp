// The portable real FFT (Windows, Linux) against the vDSP one (the Mac) — and, where there is no vDSP, against a
// direct double-precision DFT: forward, power and inverse must agree, so the analysis, phantom and the spectral
// combiner sound the same on every platform.
//   clang++ -std=c++20 -O2 -ISource tests/FFTCheck.cpp -framework Accelerate -o /tmp/fftcheck && /tmp/fftcheck
#include "dsp/Analysis.h"
#include <cstdio>
#include <random>
using namespace ns::dsp;

#if ! defined(__APPLE__)
/** the reference where there is no vDSP: the DFT by its definition, in double precision, the same conventions */
struct RealFFTvDSP
{
    int N = 0;
    void init (int n) { N = n; }
    void forward (const float* x, std::complex<float>* X)
    {
        for (int k = 0; k <= N / 2; ++k)
        {
            double re = 0, im = 0;
            for (int n = 0; n < N; ++n) { const double w = -2 * M_PI * (double) k * n / N; re += x[n] * std::cos (w); im += x[n] * std::sin (w); }
            X[k] = { (float) re, (k == 0 || k == N / 2) ? 0.0f : (float) im };
        }
    }
    void power (const float* x, float* pw)
    {
        std::vector<std::complex<float>> X ((size_t) N / 2 + 1); forward (x, X.data());
        for (int k = 0; k <= N / 2; ++k) { const double a = std::abs (X[(size_t) k]) * 2.0 / N; pw[k] = (float) (a * a); }
    }
    void inverse (const std::complex<float>* X, float* x)
    {
        for (int n = 0; n < N; ++n)
        {
            double s = X[0].real() + X[N / 2].real() * ((n % 2) ? -1.0 : 1.0);
            for (int k = 1; k < N / 2; ++k) { const double w = 2 * M_PI * (double) k * n / N; s += 2 * (X[k].real() * std::cos (w) - X[k].imag() * std::sin (w)); }
            x[n] = (float) (s / N);
        }
    }
};
#endif

int main()
{
    int fails = 0;
    std::mt19937 rng (1);
    std::normal_distribution<float> nd;
    #if defined(__APPLE__)
    const int kTrials = 20;
#else
    const int kTrials = 3;
#endif
    for (int N : { 64, 256, 1024, 2048, 4096 })
    {
        RealFFTvDSP a; RealFFTPortable b; a.init (N); b.init (N);
        std::vector<float> x ((size_t) N), pa ((size_t) N / 2 + 1), pb = pa, ya ((size_t) N), yb = ya;
        std::vector<std::complex<float>> Xa ((size_t) N / 2 + 1), Xb = Xa, R = Xa;
        double ef = 0, ep = 0, ei = 0, eRound = 0, mag = 0, pmag = 0;
        for (int trial = 0; trial < kTrials; ++trial)
        {
            for (auto& v : x) v = nd (rng);
            a.forward (x.data(), Xa.data()); b.forward (x.data(), Xb.data());
            for (int k = 0; k <= N / 2; ++k) { ef = std::max (ef, (double) std::abs (Xa[(size_t) k] - Xb[(size_t) k])); mag = std::max (mag, (double) std::abs (Xa[(size_t) k])); }
            a.power (x.data(), pa.data()); b.power (x.data(), pb.data());
            for (int k = 0; k <= N / 2; ++k) { ep = std::max (ep, (double) std::abs (pa[(size_t) k] - pb[(size_t) k])); pmag = std::max (pmag, (double) pa[(size_t) k]); }
            // an arbitrary Hermitian spectrum (what phantom and the combiner build)
            for (auto& v : R) v = { nd (rng), nd (rng) };
            a.inverse (R.data(), ya.data()); b.inverse (R.data(), yb.data());
            for (int n = 0; n < N; ++n) ei = std::max (ei, (double) std::abs (ya[(size_t) n] - yb[(size_t) n]));
            // round trip
            b.forward (x.data(), Xb.data()); b.inverse (Xb.data(), yb.data());
            for (int n = 0; n < N; ++n) eRound = std::max (eRound, (double) std::abs (yb[(size_t) n] - x[(size_t) n]));
        }
        const bool ok = ef / mag < 1e-5 && ep / pmag < 1e-5 && ei < 1e-5 && eRound < 1e-5;
        std::printf ("%s N %4d  forward %.1e (rel)  power %.1e (rel)  inverse %.1e  round trip %.1e\n", ok ? "ok  " : "FAIL", N, ef / mag, ep / pmag, ei, eRound);
        fails += ok ? 0 : 1;
    }
    std::printf (fails ? "FAILED\n" : "all passed\n");
    return fails;
}
