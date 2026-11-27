// Neurosonde — a small neural-network library for online learning on the CPU.
//
// Everything here is real backpropagation: dense layers, FiLM conditioning (Perez et al. 2018), a GRU (Cho et al.
// 2014) with truncated BPTT, Adam (Kingma & Ba 2014) with global-norm clipping, and helpers. Matrices are row-major
// float vectors; batched products go through Accelerate (cblas_sgemm) on the Mac, a portable loop elsewhere. No allocation happens in forward/backward
// once a Workspace has been sized, so the trainer thread can run steadily.
#pragma once
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <random>
#include <functional>
// every standard header the project uses, here at the root of the include chain: MSVC's library pulls in less
// through other headers than libc++ or libstdc++ do
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#ifndef M_PI      // MSVC defines it only with _USE_MATH_DEFINES
 #define M_PI 3.14159265358979323846
#endif

#if defined(__APPLE__)
 // the classic CBLAS interface: there on every macOS this runs on (the "new LAPACK" one needs macOS 13.3)
 #include <Accelerate/Accelerate.h>
 #define NS_HAVE_BLAS 1
#else
 #define NS_HAVE_BLAS 0
#endif

namespace ns::ml
{
using Vec = std::vector<float>;

// C[MxN] (+)= A[MxK] * B[KxN], with optional transposes. Row-major. The portable version (Windows, Linux) runs
// row-wise so the inner loops run over contiguous memory and vectorise; tests/GemmCheck.cpp holds it to Accelerate's.
inline void gemmPortable (bool transA, bool transB, int M, int N, int K, float alpha, const float* A, const float* B,
                          float beta, float* C)
{
    thread_local std::vector<float> acol;
    for (int i = 0; i < M; ++i)
    {
        float* c = C + (size_t) i * N;
        if (beta == 0) std::fill (c, c + N, 0.0f); else if (beta != 1) for (int j = 0; j < N; ++j) c[j] *= beta;
        const float* arow = A + (size_t) i * K;
        if (transA) { acol.resize ((size_t) K); for (int k = 0; k < K; ++k) acol[(size_t) k] = A[(size_t) k * M + i]; arow = acol.data(); }
        if (! transB)
        {
            for (int k = 0; k < K; ++k)
            {
                const float a = alpha * arow[k];
                if (a == 0) continue;
                const float* b = B + (size_t) k * N;
                for (int j = 0; j < N; ++j) c[j] += a * b[j];
            }
        }
        else
        {
            for (int j = 0; j < N; ++j)
            {
                const float* b = B + (size_t) j * K;
                float s = 0; for (int k = 0; k < K; ++k) s += arow[k] * b[k];
                c[j] += alpha * s;
            }
        }
    }
}

inline void gemm (bool transA, bool transB, int M, int N, int K, float alpha, const float* A, const float* B,
                  float beta, float* C)
{
   #if NS_HAVE_BLAS
    cblas_sgemm (CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans, M, N, K, alpha,
                 A, transA ? M : K, B, transB ? K : N, beta, C, N);
   #else
    gemmPortable (transA, transB, M, N, K, alpha, A, B, beta, C);
   #endif
}

enum class Act { Linear, Tanh, LeakyRelu, Sigmoid };

inline float actF (Act a, float x)
{
    switch (a)
    {
        case Act::Tanh: return std::tanh (x);
        case Act::LeakyRelu: return x > 0 ? x : 0.1f * x;
        case Act::Sigmoid: return 1.0f / (1.0f + std::exp (-x));
        default: return x;
    }
}
// derivative expressed through the output y = f(x)
inline float actD (Act a, float y)
{
    switch (a)
    {
        case Act::Tanh: return 1.0f - y * y;
        case Act::LeakyRelu: return y > 0 ? 1.0f : 0.1f;
        case Act::Sigmoid: return y * (1.0f - y);
        default: return 1.0f;
    }
}

/** A parameter tensor with its gradient and Adam moments. */
struct Param
{
    Vec w, g, m, v;
    void resize (size_t n) { w.assign (n, 0); g.assign (n, 0); m.assign (n, 0); v.assign (n, 0); }
    size_t size() const { return w.size(); }
};

/** Registry of every parameter of a model, for the optimiser, serialisation and federated merging. */
struct ParamSet
{
    std::vector<Param*> list;
    void add (Param& p) { list.push_back (&p); }
    size_t count() const { size_t n = 0; for (auto* p : list) n += p->size(); return n; }
    void zeroGrad() { for (auto* p : list) std::fill (p->g.begin(), p->g.end(), 0.0f); }
    double gradNorm() const
    {
        double s = 0; for (auto* p : list) for (float x : p->g) s += (double) x * x;
        return std::sqrt (s);
    }
    bool finite() const
    {
        for (auto* p : list) for (float x : p->w) if (! std::isfinite (x)) return false;
        return true;
    }
    void copyWeightsTo (Vec& out) const
    {
        out.resize (count()); size_t o = 0;
        for (auto* p : list) { std::copy (p->w.begin(), p->w.end(), out.begin() + (long) o); o += p->size(); }
    }
    void setWeightsFrom (const Vec& in)
    {
        size_t o = 0;
        for (auto* p : list) { std::copy (in.begin() + (long) o, in.begin() + (long) (o + p->size()), p->w.begin()); o += p->size(); }
    }
};

/** Adam with decoupled weight decay (AdamW), global-norm gradient clipping and a learning-rate floor. */
struct Adam
{
    float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 1e-5f, clip = 1.0f;
    long t = 0;
    /** returns the pre-clip gradient norm */
    double step (ParamSet& ps, float scale = 1.0f)
    {
        const double gn = ps.gradNorm() * scale;
        const float k = (float) (scale * (gn > clip ? clip / gn : 1.0));
        ++t;
        const float c1 = 1.0f - std::pow (b1, (float) t), c2 = 1.0f - std::pow (b2, (float) t);
        for (auto* p : ps.list)
            for (size_t i = 0; i < p->size(); ++i)
            {
                const float g = p->g[i] * k;
                p->m[i] = b1 * p->m[i] + (1 - b1) * g;
                p->v[i] = b2 * p->v[i] + (1 - b2) * g * g;
                const float mh = p->m[i] / c1, vh = p->v[i] / c2;
                p->w[i] -= lr * (mh / (std::sqrt (vh) + eps) + wd * p->w[i]);
            }
        return gn;
    }
    void reset (ParamSet& ps) { t = 0; for (auto* p : ps.list) { std::fill (p->m.begin(), p->m.end(), 0.0f); std::fill (p->v.begin(), p->v.end(), 0.0f); } }
};

inline void initGlorot (Param& p, int fanIn, int fanOut, std::mt19937& rng, float gain = 1.0f)
{
    std::normal_distribution<float> nd (0.0f, gain * std::sqrt (2.0f / (float) (fanIn + fanOut)));
    for (auto& x : p.w) x = nd (rng);
}

/** y = act(x W^T + b). Batched: X is [B x in]. Keeps its own activations for backward. */
struct Dense
{
    int in = 0, out = 0; Act act = Act::Linear;
    Param W, b;
    Vec X, Y;          // cached input / output for backward
    int B = 0;

    void init (int nIn, int nOut, Act a, std::mt19937& rng, float gain = 1.0f)
    {
        in = nIn; out = nOut; act = a;
        W.resize ((size_t) in * out); b.resize ((size_t) out);
        initGlorot (W, in, out, rng, gain);
    }
    void reg (ParamSet& ps) { ps.add (W); ps.add (b); }

    const float* forward (const float* x, int batch)
    {
        B = batch;
        X.assign (x, x + (size_t) B * in);
        Y.resize ((size_t) B * out);
        for (int r = 0; r < B; ++r) std::copy (b.w.begin(), b.w.end(), Y.begin() + (long) r * out);
        gemm (false, true, B, out, in, 1.0f, X.data(), W.w.data(), 1.0f, Y.data());
        if (act != Act::Linear) for (auto& y : Y) y = actF (act, y);
        return Y.data();
    }
    /** dY [B x out] (modified in place: through the activation). Writes dX [B x in] when dx != nullptr. */
    void backward (float* dY, float* dX)
    {
        if (act != Act::Linear)
            for (size_t i = 0; i < Y.size(); ++i) dY[i] *= actD (act, Y[i]);
        gemm (true, false, out, in, B, 1.0f, dY, X.data(), 1.0f, W.g.data());      // dW += dY^T X
        for (int r = 0; r < B; ++r) for (int j = 0; j < out; ++j) b.g[(size_t) j] += dY[(size_t) r * out + j];
        if (dX) gemm (false, false, B, in, out, 1.0f, dY, W.w.data(), 0.0f, dX);    // dX = dY W
    }
    /** single-vector inference without caching (audio thread) */
    void infer (const float* x, float* y) const
    {
        for (int j = 0; j < out; ++j)
        {
            const float* wr = W.w.data() + (size_t) j * in;
            float s = b.w[(size_t) j];
            for (int i = 0; i < in; ++i) s += wr[i] * x[i];
            y[j] = actF (act, s);
        }
    }
};

/** FiLM: h' = h * (1 + gamma(c)) + beta(c), gamma/beta a linear map of the conditioning vector. */
struct FiLM
{
    int H = 0, C = 0;
    Dense map;           // C -> 2H (gamma | beta)
    Vec Hin, GB;
    int B = 0;
    void init (int hidden, int cond, std::mt19937& rng)
    {
        H = hidden; C = cond;
        map.init (C, 2 * H, Act::Linear, rng, 0.1f);   // starts near identity
    }
    void reg (ParamSet& ps) { map.reg (ps); }
    void forward (float* h, const float* c, int batch)
    {
        B = batch;
        Hin.assign (h, h + (size_t) B * H);
        const float* gb = map.forward (c, B);
        GB.assign (gb, gb + (size_t) B * 2 * H);
        for (int r = 0; r < B; ++r)
            for (int j = 0; j < H; ++j)
                h[(size_t) r * H + j] = Hin[(size_t) r * H + j] * (1.0f + GB[(size_t) r * 2 * H + j]) + GB[(size_t) r * 2 * H + H + j];
    }
    /** dh in place: from d(h') to d(h). Accumulates the map's gradients (no gradient into c). */
    void backward (float* dh)
    {
        Vec dGB ((size_t) B * 2 * H);
        for (int r = 0; r < B; ++r)
            for (int j = 0; j < H; ++j)
            {
                const float d = dh[(size_t) r * H + j];
                dGB[(size_t) r * 2 * H + j] = d * Hin[(size_t) r * H + j];
                dGB[(size_t) r * 2 * H + H + j] = d;
                dh[(size_t) r * H + j] = d * (1.0f + GB[(size_t) r * 2 * H + j]);
            }
        map.backward (dGB.data(), nullptr);
    }
    void infer (float* h, const float* c) const
    {
        float gb[512];
        map.infer (c, gb);
        for (int j = 0; j < H; ++j) h[j] = h[j] * (1.0f + gb[j]) + gb[H + j];
    }
};

/** GRU cell (Cho et al. 2014), batched over B sequences, unrolled for truncated BPTT over T steps.
    z = σ(Wz x + Uz h + bz), r = σ(Wr x + Ur h + br), n = tanh(Wn x + bn + r ⊙ (Un h + bhn)), h' = (1-z) ⊙ n + z ⊙ h */
struct GRU
{
    int I = 0, H = 0;
    Param Wx, Uh, bx, bh;     // Wx: [3H x I] (z|r|n), Uh: [3H x H], bx/bh: [3H]
    // caches per time step
    std::vector<Vec> xs, hs, zs, rs, ns, uhn;   // uhn = Un h + bhn (pre-reset)
    int B = 0, T = 0;

    void init (int nIn, int nHidden, std::mt19937& rng)
    {
        I = nIn; H = nHidden;
        Wx.resize ((size_t) 3 * H * I); Uh.resize ((size_t) 3 * H * H); bx.resize ((size_t) 3 * H); bh.resize ((size_t) 3 * H);
        initGlorot (Wx, I, H, rng); initGlorot (Uh, H, H, rng);
        for (int j = 0; j < H; ++j) bx.w[(size_t) j] = 1.0f;   // update gate biased to remember
    }
    void reg (ParamSet& ps) { ps.add (Wx); ps.add (Uh); ps.add (bx); ps.add (bh); }

    /** x: [T][B x I], h0: [B x H]. Returns hidden states [T][B x H] (hs[1..T]). */
    void forward (const std::vector<Vec>& x, const Vec& h0, int batch)
    {
        B = batch; T = (int) x.size();
        xs = x; hs.assign ((size_t) T + 1, Vec()); hs[0] = h0;
        zs.assign ((size_t) T, Vec()); rs = zs; ns = zs; uhn = zs;
        Vec gx ((size_t) B * 3 * H), gh ((size_t) B * 3 * H);
        for (int t = 0; t < T; ++t)
        {
            for (int r = 0; r < B; ++r)
            {
                std::copy (bx.w.begin(), bx.w.end(), gx.begin() + (long) r * 3 * H);
                std::copy (bh.w.begin(), bh.w.end(), gh.begin() + (long) r * 3 * H);
            }
            gemm (false, true, B, 3 * H, I, 1.0f, xs[(size_t) t].data(), Wx.w.data(), 1.0f, gx.data());
            gemm (false, true, B, 3 * H, H, 1.0f, hs[(size_t) t].data(), Uh.w.data(), 1.0f, gh.data());
            Vec& z = zs[(size_t) t]; Vec& rr = rs[(size_t) t]; Vec& n = ns[(size_t) t]; Vec& u = uhn[(size_t) t];
            z.resize ((size_t) B * H); rr.resize ((size_t) B * H); n.resize ((size_t) B * H); u.resize ((size_t) B * H);
            Vec& hn = hs[(size_t) t + 1]; hn.resize ((size_t) B * H);
            for (int r = 0; r < B; ++r)
                for (int j = 0; j < H; ++j)
                {
                    const size_t o = (size_t) r * 3 * H, k = (size_t) r * H + j;
                    z[k] = actF (Act::Sigmoid, gx[o + j] + gh[o + j]);
                    rr[k] = actF (Act::Sigmoid, gx[o + H + j] + gh[o + H + j]);
                    u[k] = gh[o + 2 * H + j];
                    n[k] = std::tanh (gx[o + 2 * H + j] + rr[k] * u[k]);
                    hn[k] = (1 - z[k]) * n[k] + z[k] * hs[(size_t) t][k];
                }
        }
    }
    /** dH: [T][B x H] gradients w.r.t. each hidden output. Accumulates parameter gradients; optional dX. */
    void backward (const std::vector<Vec>& dH, std::vector<Vec>* dX = nullptr)
    {
        Vec dh ((size_t) B * H, 0.0f), dgx ((size_t) B * 3 * H), dgh ((size_t) B * 3 * H), dhPrev ((size_t) B * H);
        if (dX) dX->assign ((size_t) T, Vec ((size_t) B * I, 0.0f));
        for (int t = T - 1; t >= 0; --t)
        {
            const Vec& z = zs[(size_t) t]; const Vec& rr = rs[(size_t) t]; const Vec& n = ns[(size_t) t];
            const Vec& u = uhn[(size_t) t]; const Vec& hp = hs[(size_t) t];
            for (size_t k = 0; k < dh.size(); ++k) dh[k] += dH[(size_t) t][k];
            for (int r = 0; r < B; ++r)
                for (int j = 0; j < H; ++j)
                {
                    const size_t k = (size_t) r * H + j, o = (size_t) r * 3 * H;
                    const float d = dh[k];
                    const float dz = d * (hp[k] - n[k]) * z[k] * (1 - z[k]);
                    const float dn = d * (1 - z[k]) * (1 - n[k] * n[k]);
                    const float dr = dn * u[k] * rr[k] * (1 - rr[k]);
                    dgx[o + j] = dz; dgx[o + H + j] = dr; dgx[o + 2 * H + j] = dn;
                    dgh[o + j] = dz; dgh[o + H + j] = dr; dgh[o + 2 * H + j] = dn * rr[k];
                    dhPrev[k] = d * z[k];
                }
            gemm (true, false, 3 * H, I, B, 1.0f, dgx.data(), xs[(size_t) t].data(), 1.0f, Wx.g.data());
            gemm (true, false, 3 * H, H, B, 1.0f, dgh.data(), hp.data(), 1.0f, Uh.g.data());
            for (int r = 0; r < B; ++r)
                for (int j = 0; j < 3 * H; ++j) { bx.g[(size_t) j] += dgx[(size_t) r * 3 * H + j]; bh.g[(size_t) j] += dgh[(size_t) r * 3 * H + j]; }
            gemm (false, false, B, H, 3 * H, 1.0f, dgh.data(), Uh.w.data(), 1.0f, dhPrev.data());
            if (dX) gemm (false, false, B, I, 3 * H, 1.0f, dgx.data(), Wx.w.data(), 0.0f, (*dX)[(size_t) t].data());
            dh = dhPrev;
        }
    }
    /** one step for a single sequence (audio/inference thread) */
    void step (const float* x, float* h) const
    {
        float gx[3 * 256], gh[3 * 256];
        for (int j = 0; j < 3 * H; ++j)
        {
            float s = bx.w[(size_t) j], q = bh.w[(size_t) j];
            const float* wr = Wx.w.data() + (size_t) j * I; for (int i = 0; i < I; ++i) s += wr[i] * x[i];
            const float* ur = Uh.w.data() + (size_t) j * H; for (int i = 0; i < H; ++i) q += ur[i] * h[i];
            gx[j] = s; gh[j] = q;
        }
        for (int j = 0; j < H; ++j)
        {
            const float z = actF (Act::Sigmoid, gx[j] + gh[j]);
            const float r = actF (Act::Sigmoid, gx[H + j] + gh[H + j]);
            const float n = std::tanh (gx[2 * H + j] + r * gh[2 * H + j]);
            h[j] = (1 - z) * n + z * h[j];
        }
    }
};

/** Welford-style running mean/variance with an exponential window (input standardisation). */
struct RunningNorm
{
    Vec mean, var; double count = 0; float alpha = 1e-3f;
    void init (int n) { mean.assign ((size_t) n, 0.0f); var.assign ((size_t) n, 1.0f); count = 0; }
    void update (const float* x)
    {
        count += 1;
        const float a = (float) std::max ((double) alpha, 1.0 / count);
        for (size_t i = 0; i < mean.size(); ++i)
        {
            const float d = x[i] - mean[i];
            mean[i] += a * d;
            var[i] = (1 - a) * (var[i] + a * d * d);
        }
    }
    void apply (const float* x, float* y) const
    {
        for (size_t i = 0; i < mean.size(); ++i) y[i] = (x[i] - mean[i]) / std::sqrt (var[i] + 1e-4f);
    }
    void undo (const float* y, float* x) const
    {
        for (size_t i = 0; i < mean.size(); ++i) x[i] = y[i] * std::sqrt (var[i] + 1e-4f) + mean[i];
    }
};

/** Reservoir sampling (Vitter 1985): a uniform sample of everything ever seen, the replay memory that protects the
    network from catastrophic forgetting. */
template <typename Item>
struct Reservoir
{
    std::vector<Item> items; size_t capacity = 0; uint64_t seen = 0;
    std::mt19937_64 rng { 0x5eed };
    void init (size_t cap) { capacity = cap; items.clear(); items.reserve (cap); seen = 0; }
    void add (const Item& it)
    {
        ++seen;
        if (items.size() < capacity) { items.push_back (it); return; }
        const uint64_t j = std::uniform_int_distribution<uint64_t> (0, seen - 1) (rng);
        if (j < capacity) items[(size_t) j] = it;
    }
    const Item& sample() { return items[std::uniform_int_distribution<size_t> (0, items.size() - 1) (rng)]; }
    bool empty() const { return items.empty(); }
};
} // namespace ns::ml
