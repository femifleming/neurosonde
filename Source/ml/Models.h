// Neurosonde — the models: a conditional VAE over spectral frames (with DDSP-style synthesis heads), a latent GRU
// that dreams forward, an echo state network with an RLS readout, a self-organising map (the constellation), a modern
// Hopfield recall over remembered moments, and a hashed n-gram text embedding for the patch notes.
#pragma once
#include "NN.h"
#include <string>
#include <map>
#include <array>

namespace ns::ml
{
// ---- dimensions --------------------------------------------------------------------------------------------------
constexpr int kMel = 64, kChroma = 12, kScalars = 8;
constexpr int kFeat = kMel + kChroma + kScalars;      // 84 analysed features per 10 ms frame
constexpr int kHarm = 48;                              // harmonic distribution (DDSP-style)
constexpr int kNoise = 32;                             // noise band envelope
constexpr int kOut = kFeat + kHarm + kNoise;           // decoder output
constexpr int kZ = 16;                                 // latent size
constexpr int kCond = 32;                              // conditioning (patch notes)
constexpr int kGruH = 96;
constexpr int kSom = 16;                               // 16 x 16 constellation
constexpr int kSpatial = 24;                           // stereo / spatial features (dsp/Stereo.h)
// scalar slots inside the feature vector
enum Scalar { kF0 = kMel + kChroma, kVoicing, kLoud, kCentroid, kSpread, kFlatness, kFlux, kRolloff };

/** A training example: standardised features, harmonic target distribution, standardised noise bands, voicing
    weight, and the conditioning vector (the patch notes active when it was heard). */
struct Frame
{
    std::array<float, kFeat> x;
    std::array<float, kHarm> harm;
    std::array<float, kNoise> noise;
    std::array<float, kCond> cond;
    float voiced = 0;
    std::array<float, kSpatial> spat {};   // standardised spatial features (when the input was stereo)
    float hasSpat = 0;
};

// ---- conditional VAE ---------------------------------------------------------------------------------------------
struct VAE
{
    Dense e1, e2, eh, d1, d2, dout;
    FiLM fe, fd;
    ParamSet ps;
    float beta = 0.0f;            // KL weight (warmed up by the trainer)
    // per-batch scratch
    Vec mu, lv, eps, z, out; std::vector<bool> lvIn;

    void init (uint32_t seed)
    {
        std::mt19937 rng (seed);
        e1.init (kFeat, 128, Act::LeakyRelu, rng); fe.init (128, kCond, rng);
        e2.init (128, 64, Act::LeakyRelu, rng);
        eh.init (64, 2 * kZ, Act::Linear, rng, 0.5f);
        d1.init (kZ, 128, Act::LeakyRelu, rng); fd.init (128, kCond, rng);
        d2.init (128, 128, Act::LeakyRelu, rng);
        dout.init (128, kOut, Act::Linear, rng);
        ps = {};
        e1.reg (ps); fe.reg (ps); e2.reg (ps); eh.reg (ps); d1.reg (ps); fd.reg (ps); d2.reg (ps); dout.reg (ps);
    }

    struct Losses { double total = 0, recon = 0, harm = 0, noise = 0, kl = 0; };

    /** one batch forward + backward (gradients accumulated into ps; caller zeroes and steps). */
    Losses trainBatch (const std::vector<const Frame*>& batch, std::mt19937& rng)
    {
        const int B = (int) batch.size();
        Vec X ((size_t) B * kFeat), C ((size_t) B * kCond);
        for (int r = 0; r < B; ++r)
        {
            std::copy (batch[(size_t) r]->x.begin(), batch[(size_t) r]->x.end(), X.begin() + (long) r * kFeat);
            std::copy (batch[(size_t) r]->cond.begin(), batch[(size_t) r]->cond.end(), C.begin() + (long) r * kCond);
        }
        // encoder
        e1.forward (X.data(), B);
        Vec h1 (e1.Y);
        fe.forward (h1.data(), C.data(), B);
        const float* h2 = e2.forward (h1.data(), B);
        const float* hd = eh.forward (h2, B);
        mu.resize ((size_t) B * kZ); lv.resize ((size_t) B * kZ); lvIn.assign ((size_t) B * kZ, true); eps.resize ((size_t) B * kZ); z.resize ((size_t) B * kZ);
        std::normal_distribution<float> nd;
        for (int r = 0; r < B; ++r)
            for (int j = 0; j < kZ; ++j)
            {
                const size_t k = (size_t) r * kZ + j;
                mu[k] = hd[(size_t) r * 2 * kZ + j];
                const float rawLv = hd[(size_t) r * 2 * kZ + kZ + j];
                lv[k] = std::clamp (rawLv, -8.0f, 4.0f);
                lvIn[k] = rawLv > -8.0f && rawLv < 4.0f;
                eps[k] = nd (rng);
                z[k] = mu[k] + std::exp (0.5f * lv[k]) * eps[k];
            }
        // decoder
        d1.forward (z.data(), B);
        Vec g1 (d1.Y);
        fd.forward (g1.data(), C.data(), B);
        const float* g2 = d2.forward (g1.data(), B);
        const float* o = dout.forward (g2, B);

        // losses and dOut
        Losses L;
        Vec dO ((size_t) B * kOut, 0.0f);
        const float invB = 1.0f / (float) B;
        for (int r = 0; r < B; ++r)
        {
            const Frame& f = *batch[(size_t) r];
            const float* orow = o + (size_t) r * kOut; float* drow = dO.data() + (size_t) r * kOut;
            for (int i = 0; i < kFeat; ++i)
            {
                const float w = i < kMel ? 1.0f : i < kMel + kChroma ? 0.5f : 1.0f;
                const float e = orow[i] - f.x[(size_t) i];
                L.recon += w * e * e / kFeat; drow[i] = 2 * w * e / kFeat * invB;
            }
            // harmonic distribution: softmax cross-entropy, weighted by voicing
            float mx = -1e9f; for (int i = 0; i < kHarm; ++i) mx = std::max (mx, orow[kFeat + i]);
            float p[kHarm]; float s = 0;
            for (int i = 0; i < kHarm; ++i) { p[i] = std::exp (orow[kFeat + i] - mx); s += p[i]; }
            for (int i = 0; i < kHarm; ++i)
            {
                p[i] /= s;
                L.harm -= f.voiced * f.harm[(size_t) i] * std::log (p[i] + 1e-9f);
                drow[kFeat + i] = f.voiced * (p[i] - f.harm[(size_t) i]) * 0.25f * invB;
            }
            for (int i = 0; i < kNoise; ++i)
            {
                const float e = orow[kFeat + kHarm + i] - f.noise[(size_t) i];
                L.noise += e * e / kNoise; drow[kFeat + kHarm + i] = 2 * e / kNoise * invB;
            }
        }
        // backward through the decoder
        Vec dg2 ((size_t) B * 128), dg1 ((size_t) B * 128), dz ((size_t) B * kZ);
        dout.backward (dO.data(), dg2.data());
        d2.backward (dg2.data(), dg1.data());
        fd.backward (dg1.data());
        d1.backward (dg1.data(), dz.data());
        // reparameterisation + KL
        Vec dHd ((size_t) B * 2 * kZ);
        for (int r = 0; r < B; ++r)
            for (int j = 0; j < kZ; ++j)
            {
                const size_t k = (size_t) r * kZ + j;
                const float sd = std::exp (0.5f * lv[k]);
                L.kl += -0.5 * (1 + lv[k] - mu[k] * mu[k] - sd * sd) / kZ;
                const float dmu = dz[k] + beta * mu[k] / kZ * invB;
                const float dlv = dz[k] * eps[k] * 0.5f * sd + beta * 0.5f * (sd * sd - 1) / kZ * invB;
                dHd[(size_t) r * 2 * kZ + j] = dmu;
                dHd[(size_t) r * 2 * kZ + kZ + j] = lvIn[k] ? dlv : 0.0f;
            }
        Vec dh2 ((size_t) B * 64), dh1 ((size_t) B * 128);
        eh.backward (dHd.data(), dh2.data());
        e2.backward (dh2.data(), dh1.data());
        fe.backward (dh1.data());
        e1.backward (dh1.data(), nullptr);
        L.recon /= B; L.harm /= B; L.noise /= B; L.kl /= B;
        L.total = L.recon + 0.25 * L.harm + L.noise + beta * L.kl;
        return L;
    }

    // ---- single-frame inference (audio/control thread) ----
    void encode (const float* x, const float* c, float* zMu, float* zLv = nullptr) const
    {
        float h1[128], h2[64], hd[2 * kZ];
        e1.infer (x, h1); fe.infer (h1, c); e2.infer (h1, h2); eh.infer (h2, hd);
        for (int j = 0; j < kZ; ++j) { zMu[j] = hd[j]; if (zLv) zLv[j] = std::clamp (hd[kZ + j], -8.0f, 4.0f); }
    }
    /** out: kOut (features | harmonic logits | noise) */
    void decode (const float* zz, const float* c, float* outv) const
    {
        float g1[128], g2[128];
        d1.infer (zz, g1); fd.infer (g1, c); d2.infer (g1, g2); dout.infer (g2, outv);
    }
};

// ---- latent GRU: p(z_{t+1} | z_<=t, c), diagonal Gaussian output -------------------------------------------------
struct LatentGRU
{
    GRU gru; Dense head; ParamSet ps;
    void init (uint32_t seed)
    {
        std::mt19937 rng (seed);
        gru.init (kZ + kCond, kGruH, rng); head.init (kGruH, 2 * kZ, Act::Linear, rng, 0.5f);
        ps = {}; gru.reg (ps); head.reg (ps);
    }
    /** seqZ: [B][T+1][kZ] latents, cond: [B][kCond]. Returns mean NLL. */
    double trainBatch (const std::vector<std::vector<std::array<float, kZ>>>& seqZ, const std::vector<std::array<float, kCond>>& cond)
    {
        const int B = (int) seqZ.size(), T = (int) seqZ[0].size() - 1;
        std::vector<Vec> x ((size_t) T, Vec ((size_t) B * (kZ + kCond)));
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < B; ++r)
            {
                float* row = x[(size_t) t].data() + (size_t) r * (kZ + kCond);
                std::copy (seqZ[(size_t) r][(size_t) t].begin(), seqZ[(size_t) r][(size_t) t].end(), row);
                std::copy (cond[(size_t) r].begin(), cond[(size_t) r].end(), row + kZ);
            }
        gru.forward (x, Vec ((size_t) B * kGruH, 0.0f), B);
        std::vector<Vec> dH ((size_t) T, Vec ((size_t) B * kGruH, 0.0f));
        double nll = 0;
        const float k = 1.0f / (float) (B * T);
        for (int t = 0; t < T; ++t)
        {
            const float* o = head.forward (gru.hs[(size_t) t + 1].data(), B);
            Vec dO ((size_t) B * 2 * kZ);
            for (int r = 0; r < B; ++r)
                for (int j = 0; j < kZ; ++j)
                {
                    const float m = o[(size_t) r * 2 * kZ + j];
                    const float raw = o[(size_t) r * 2 * kZ + kZ + j], l = std::clamp (raw, -6.0f, 2.0f);
                    const float y = seqZ[(size_t) r][(size_t) t + 1][(size_t) j];
                    const float iv = std::exp (-l), e = y - m;
                    nll += 0.5 * (l + e * e * iv) / kZ;
                    dO[(size_t) r * 2 * kZ + j] = -e * iv * k / kZ;
                    dO[(size_t) r * 2 * kZ + kZ + j] = (raw > -6.0f && raw < 2.0f) ? 0.5f * (1 - e * e * iv) * k / kZ : 0.0f;
                }
            head.backward (dO.data(), dH[(size_t) t].data());
        }
        gru.backward (dH);
        return nll / (B * T);
    }
    /** one generative step: h updated, next latent sampled at temperature tau */
    void step (const float* zIn, const float* c, float* h, float* zNext, float tau, std::mt19937& rng) const
    {
        float in[kZ + kCond]; std::copy (zIn, zIn + kZ, in); std::copy (c, c + kCond, in + kZ);
        gru.step (in, h);
        float o[2 * kZ]; head.infer (h, o);
        std::normal_distribution<float> nd;
        for (int j = 0; j < kZ; ++j)
            zNext[j] = o[j] + tau * std::exp (0.5f * std::clamp (o[kZ + j], -6.0f, 2.0f)) * nd (rng);
    }
};

// ---- spatial network: p(spatial image | timbre latent, notes), a diagonal Gaussian ----------------------------------
/** Learns how the input's stereo image (per-band level differences and coherence, width, balance, diffuseness,
    motion) goes with its timbre. Its input is the encoder's latent mean, detached (no gradient flows back), so
    learning space never bends the timbre model. Trained by Gaussian negative log-likelihood, so the variance it
    predicts is its real uncertainty. */
struct SpatialNet
{
    static constexpr float kVarFloor = 1e-3f;   // in standardised units: σ ≥ 0.03
    float beta = 0.5f;                          // β-NLL (Seitzer et al. 2022); 0 = plain Gaussian NLL
    Dense h1, h2, out; ParamSet ps;
    void init (uint32_t seed)
    {
        std::mt19937 rng (seed);
        h1.init (kZ + kCond, 64, Act::LeakyRelu, rng); h2.init (64, 64, Act::LeakyRelu, rng); out.init (64, 2 * kSpatial, Act::Linear, rng, 0.5f);
        ps = {}; h1.reg (ps); h2.reg (ps); out.reg (ps);
    }
    /** zs: [B x kZ], cs: [B x kCond], targets: [B x kSpatial] (standardised). Returns mean NLL per dim. */
    double trainBatch (const Vec& zs, const Vec& cs, const Vec& targets, int B)
    {
        Vec x ((size_t) B * (kZ + kCond));
        for (int r = 0; r < B; ++r)
        {
            std::copy (zs.begin() + (long) r * kZ, zs.begin() + (long) (r + 1) * kZ, x.begin() + (long) r * (kZ + kCond));
            std::copy (cs.begin() + (long) r * kCond, cs.begin() + (long) (r + 1) * kCond, x.begin() + (long) r * (kZ + kCond) + kZ);
        }
        const float* a1 = h1.forward (x.data(), B); Vec A1 (a1, a1 + (size_t) B * 64);
        const float* a2 = h2.forward (A1.data(), B); Vec A2 (a2, a2 + (size_t) B * 64);
        const float* o = out.forward (A2.data(), B);
        Vec dO ((size_t) B * 2 * kSpatial); double nll = 0; const float k = 1.0f / (float) (B * kSpatial);
        for (int r = 0; r < B; ++r)
            for (int j = 0; j < kSpatial; ++j)
            {
                // variance = floor + softplus(raw): smooth, never collapses; β-NLL weights each term by var^β
                // (detached), which stops the mean from stalling where the data is nearly deterministic
                const float m = o[(size_t) r * 2 * kSpatial + j], raw = o[(size_t) r * 2 * kSpatial + kSpatial + j];
                const float sp = raw > 20 ? raw : std::log1p (std::exp (raw)), sig = 1.0f / (1.0f + std::exp (-raw));
                const float var = kVarFloor + sp, y = targets[(size_t) r * kSpatial + j], e = y - m;
                const float w = beta > 0 ? std::pow (var, beta) : 1.0f;
                nll += 0.5 * (std::log (var) + e * e / var);
                dO[(size_t) r * 2 * kSpatial + j] = -w * e / var * k;
                dO[(size_t) r * 2 * kSpatial + kSpatial + j] = w * 0.5f * (1.0f / var - e * e / (var * var)) * sig * k;
            }
        Vec d2 ((size_t) B * 64), d1 ((size_t) B * 64);
        out.backward (dO.data(), d2.data()); h2.backward (d2.data(), d1.data()); h1.backward (d1.data(), nullptr);
        return nll / (B * kSpatial);
    }
    void infer (const float* z, const float* c, float* mean, float* logvar) const
    {
        float x[kZ + kCond], a[64], b[64], o[2 * kSpatial];
        std::copy (z, z + kZ, x); std::copy (c, c + kCond, x + kZ);
        h1.infer (x, a); h2.infer (a, b); out.infer (b, o);
        for (int j = 0; j < kSpatial; ++j)
        {
            mean[j] = o[j];
            const float raw = o[kSpatial + j];
            if (logvar) logvar[j] = std::log (kVarFloor + (raw > 20 ? raw : std::log1p (std::exp (raw))));
        }
    }
};

// ---- echo state network with recursive-least-squares readout (Jaeger 2001) ------------------------------------
struct EchoState
{
    static constexpr int N = 256, I = kZ + kCond;
    Vec Win, W;              // input and recurrent weights (fixed, random)
    Vec Wout;                // [kZ x (N+1)] learned
    Vec P;                   // RLS inverse correlation [(N+1) x (N+1)]
    float leak = 0.3f, lambda = 0.9995f;
    uint64_t updates = 0;

    void init (uint32_t seed)
    {
        std::mt19937 rng (seed);
        std::uniform_real_distribution<float> u (-1.0f, 1.0f);
        Win.resize ((size_t) N * I); for (auto& w : Win) w = 0.5f * u (rng);
        W.assign ((size_t) N * N, 0.0f);
        std::uniform_real_distribution<float> p01 (0, 1);
        for (auto& w : W) if (p01 (rng) < 0.05f) w = u (rng);
        // scale to spectral radius 0.95 (power iteration)
        Vec v ((size_t) N, 1.0f), t ((size_t) N);
        float rho = 1;
        for (int it = 0; it < 60; ++it)
        {
            for (int i = 0; i < N; ++i) { float s = 0; for (int j = 0; j < N; ++j) s += W[(size_t) i * N + j] * v[(size_t) j]; t[(size_t) i] = s; }
            float n = 0; for (float x : t) n += x * x; n = std::sqrt (n); rho = n / std::sqrt ((float) N) * std::sqrt ((float) N) / std::max (1e-9f, std::sqrt (std::inner_product (v.begin(), v.end(), v.begin(), 0.0f)));
            for (int i = 0; i < N; ++i) v[(size_t) i] = t[(size_t) i] / std::max (1e-9f, n);
        }
        for (auto& w : W) w *= 0.95f / std::max (1e-6f, rho);
        Wout.assign ((size_t) kZ * (N + 1), 0.0f);
        P.assign ((size_t) (N + 1) * (N + 1), 0.0f);
        for (int i = 0; i <= N; ++i) P[(size_t) i * (N + 1) + i] = 100.0f;
        updates = 0;
    }
    void advance (float* s, const float* in) const
    {
        float nx[N];
        for (int i = 0; i < N; ++i)
        {
            float a = 0;
            const float* wi = Win.data() + (size_t) i * I; for (int j = 0; j < I; ++j) a += wi[j] * in[j];
            const float* wr = W.data() + (size_t) i * N; for (int j = 0; j < N; ++j) a += wr[j] * s[j];
            nx[i] = (1 - leak) * s[i] + leak * std::tanh (a);
        }
        std::copy (nx, nx + N, s);
    }
    void readout (const float* s, float* y) const
    {
        for (int k = 0; k < kZ; ++k)
        {
            const float* w = Wout.data() + (size_t) k * (N + 1);
            float a = w[N]; for (int j = 0; j < N; ++j) a += w[j] * s[j];
            y[k] = a;
        }
    }
    /** RLS update of the readout towards target y for state s (O(N^2)). */
    void learn (const float* s, const float* target)
    {
        const int M = N + 1;
        float phi[N + 1]; std::copy (s, s + N, phi); phi[N] = 1.0f;
        Vec Pphi ((size_t) M, 0.0f);
        for (int i = 0; i < M; ++i) { float a = 0; for (int j = 0; j < M; ++j) a += P[(size_t) i * M + j] * phi[j]; Pphi[(size_t) i] = a; }
        float den = lambda; for (int i = 0; i < M; ++i) den += phi[i] * Pphi[(size_t) i];
        Vec k ((size_t) M); for (int i = 0; i < M; ++i) k[(size_t) i] = Pphi[(size_t) i] / den;
        float y[kZ]; readout (s, y);
        for (int o = 0; o < kZ; ++o)
        {
            const float e = target[o] - y[o];
            float* w = Wout.data() + (size_t) o * M;
            for (int i = 0; i < M; ++i) w[i] += k[(size_t) i] * e;
        }
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                P[(size_t) i * M + j] = (P[(size_t) i * M + j] - k[(size_t) i] * Pphi[(size_t) j]) / lambda;
        ++updates;
    }
};

// ---- self-organising map: the constellation (Kohonen 1982) --------------------------------------------------------
struct SOM
{
    Vec w;                         // [kSom*kSom x kZ]
    Vec hits;                      // decaying hit counts per node
    uint64_t steps = 0;
    void init (uint32_t seed)
    {
        std::mt19937 rng (seed); std::normal_distribution<float> nd (0.0f, 0.3f);
        w.resize ((size_t) kSom * kSom * kZ);
        for (int r = 0; r < kSom; ++r)
            for (int c = 0; c < kSom; ++c)
                for (int j = 0; j < kZ; ++j)   // start as a gentle 2-D sheet over the first two latent dims
                    w[((size_t) r * kSom + c) * kZ + j] = j == 0 ? (c / (kSom - 1.0f) - 0.5f) * 2 : j == 1 ? (r / (kSom - 1.0f) - 0.5f) * 2 : nd (rng);
        hits.assign ((size_t) kSom * kSom, 0.0f); steps = 0;
    }
    int bmu (const float* z) const
    {
        int best = 0; float bd = 1e30f;
        for (int n = 0; n < kSom * kSom; ++n)
        {
            float d = 0; const float* wn = w.data() + (size_t) n * kZ;
            for (int j = 0; j < kZ; ++j) { const float e = wn[j] - z[j]; d += e * e; }
            if (d < bd) { bd = d; best = n; }
        }
        return best;
    }
    float sigma() const { return std::max (1.0f, 4.0f * std::exp (-(float) steps / 40000.0f)); }
    float eta() const { return std::max (0.02f, 0.2f * std::exp (-(float) steps / 40000.0f)); }
    int learn (const float* z)
    {
        const int b = bmu (z); const int br = b / kSom, bc = b % kSom;
        const float s = sigma(), e = eta();
        for (int r = 0; r < kSom; ++r)
            for (int c = 0; c < kSom; ++c)
            {
                const float d2 = (float) ((r - br) * (r - br) + (c - bc) * (c - bc));
                const float h = e * std::exp (-d2 / (2 * s * s));
                if (h < 1e-4f) continue;
                float* wn = w.data() + ((size_t) r * kSom + c) * kZ;
                for (int j = 0; j < kZ; ++j) wn[j] += h * (z[j] - wn[j]);
            }
        for (auto& h : hits) h *= 0.9999f;
        hits[(size_t) b] += 1.0f;
        ++steps;
        return b;
    }
    /** bilinear interpolation of codebook vectors at a continuous map position (0..1, 0..1) */
    void at (float x, float y, float* z) const
    {
        const float fx = std::clamp (x, 0.0f, 1.0f) * (kSom - 1), fy = std::clamp (y, 0.0f, 1.0f) * (kSom - 1);
        const int x0 = (int) fx, y0 = (int) fy, x1 = std::min (kSom - 1, x0 + 1), y1 = std::min (kSom - 1, y0 + 1);
        const float ax = fx - x0, ay = fy - y0;
        for (int j = 0; j < kZ; ++j)
        {
            auto W = [&] (int r, int c) { return w[((size_t) r * kSom + c) * kZ + j]; };
            z[j] = (1 - ay) * ((1 - ax) * W (y0, x0) + ax * W (y0, x1)) + ay * ((1 - ax) * W (y1, x0) + ax * W (y1, x1));
        }
    }
};

// ---- modern Hopfield recall (Ramsauer et al. 2020): softmax attention over stored patterns ------------------------
/** p_i = softmax(-beta/2 * |q - k_i|^2); returns a sampled index (or the argmax when sample == false). */
inline int hopfieldRecall (const float* q, const std::vector<std::array<float, kZ>>& keys, float beta, std::mt19937& rng,
                           bool sample = true, int exclude = -1)
{
    if (keys.empty()) return -1;
    std::vector<float> e (keys.size());
    float mx = -1e30f;
    for (size_t i = 0; i < keys.size(); ++i)
    {
        float d = 0; for (int j = 0; j < kZ; ++j) { const float t = q[j] - keys[i][(size_t) j]; d += t * t; }
        e[i] = (int) i == exclude ? -1e30f : -0.5f * beta * d;
        mx = std::max (mx, e[i]);
    }
    if (! sample) return (int) (std::max_element (e.begin(), e.end()) - e.begin());
    double s = 0; for (auto& v : e) { v = std::exp (v - mx); s += v; }
    double u = std::uniform_real_distribution<double> (0, s) (rng);
    for (size_t i = 0; i < e.size(); ++i) { u -= e[i]; if (u <= 0) return (int) i; }
    return (int) e.size() - 1;
}

// ---- patch notes → conditioning vector ---------------------------------------------------------------------------
/** fastText-style hashed character n-grams (3–5, with word boundaries) and words, each bucket mapped to a seeded
    Gaussian random projection (no stored table: Johnson–Lindenstrauss), averaged and normalised to 24 dims, plus
    8 interpretable lexicon axes (with simple negation, after VADER). */
struct TextEmbed
{
    static constexpr int kHash = 24, kAxes = 8;
    enum Axis { Bright, Warm, Dense, Motion, Rough, Wide, Wet, Dream };
    static const char* axisName (int a)
    {
        static const char* n[] = { "bright", "warm", "dense", "motion", "rough", "wide", "wet", "dream" };
        return n[a];
    }
    static uint64_t fnv (const std::string& s)
    {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char ch : s) { h ^= ch; h *= 1099511628211ull; }
        return h;
    }
    static void bucketVec (uint64_t h, float* v)
    {
        std::mt19937 r ((uint32_t) (h ^ (h >> 32)));
        std::normal_distribution<float> nd;
        for (int i = 0; i < kHash; ++i) v[i] = nd (r);
    }
    static const std::map<std::string, std::array<float, kAxes>>& lexicon()
    {
        //                                            bright warm dense motion rough wide wet dream
        static const std::map<std::string, std::array<float, kAxes>> L = {
            { "bright", {1, 0, 0, 0, 0, 0, 0, 0} }, { "light", {.8f, .2f, -.2f, 0, -.2f, .2f, 0, .2f} },
            { "glass", {.9f, -.3f, -.3f, 0, -.2f, .3f, 0, .3f} }, { "shimmer", {.9f, 0, 0, .4f, 0, .5f, .2f, .5f} },
            { "sun", {.7f, .8f, 0, 0, 0, .5f, -.5f, 0} }, { "golden", {.4f, .9f, 0, 0, 0, .3f, 0, .3f} },
            { "dark", {-1, 0, 0, 0, 0, 0, 0, .3f} }, { "night", {-.7f, -.2f, 0, -.3f, 0, .3f, 0, .7f} },
            { "tungsten", {-.2f, 1, 0, 0, .2f, -.2f, 0, .3f} }, { "sepia", {-.4f, .8f, 0, -.3f, .3f, 0, 0, .6f} },
            { "warm", {0, 1, 0, 0, 0, 0, 0, 0} }, { "cold", {.3f, -1, 0, 0, 0, .3f, 0, 0} }, { "ice", {.6f, -1, 0, -.4f, 0, .3f, .3f, 0} },
            { "dust", {-.2f, .3f, .2f, 0, .8f, 0, -.6f, .3f} }, { "sand", {0, .4f, .4f, .2f, .7f, .3f, -.8f, 0} },
            { "desert", {.2f, .6f, -.5f, -.3f, .4f, 1, -1, .4f} }, { "warehouse", {-.2f, 0, -.3f, 0, .4f, .8f, 0, .3f} },
            { "metal", {.6f, -.4f, .2f, 0, .6f, 0, 0, 0} }, { "wood", {-.2f, .7f, 0, 0, .3f, -.2f, 0, 0} },
            { "rust", {-.3f, .5f, .2f, 0, .9f, 0, .2f, 0} }, { "broken", {0, 0, 0, .3f, .9f, 0, 0, .3f} },
            { "creak", {0, .2f, 0, .3f, .8f, -.2f, 0, 0} }, { "machine", {.2f, -.3f, .5f, .6f, .5f, 0, 0, 0} },
            { "dense", {0, 0, 1, 0, 0, 0, 0, 0} }, { "thick", {-.3f, .3f, 1, 0, .2f, 0, 0, 0} }, { "cloud", {0, .2f, .6f, -.2f, -.3f, .6f, .4f, .7f} },
            { "sparse", {0, 0, -1, 0, 0, .3f, 0, 0} }, { "empty", {0, -.3f, -1, -.5f, 0, .8f, 0, .5f} }, { "void", {-.5f, -.5f, -1, -.6f, 0, 1, 0, .8f} },
            { "fast", {0, 0, .3f, 1, 0, 0, 0, 0} }, { "rush", {0, 0, .5f, 1, .3f, 0, 0, 0} }, { "pulse", {0, 0, .2f, .7f, 0, 0, 0, 0} },
            { "slow", {0, 0, 0, -1, 0, 0, 0, .3f} }, { "still", {0, 0, -.3f, -1, -.3f, 0, 0, .5f} }, { "drift", {0, 0, 0, -.4f, 0, .4f, .3f, .7f} },
            { "rough", {0, 0, 0, 0, 1, 0, 0, 0} }, { "grain", {.2f, 0, .3f, .3f, .7f, 0, 0, 0} }, { "noise", {.3f, 0, .5f, .2f, .8f, 0, 0, 0} },
            { "smooth", {0, .3f, 0, 0, -1, 0, 0, 0} }, { "soft", {-.3f, .5f, 0, -.3f, -.8f, 0, 0, .3f} }, { "pure", {.3f, 0, -.5f, 0, -1, 0, 0, 0} },
            { "wide", {0, 0, 0, 0, 0, 1, 0, 0} }, { "vast", {0, 0, 0, -.2f, 0, 1, 0, .4f} }, { "infinite", {0, 0, 0, -.3f, 0, 1, 0, .7f} },
            { "close", {0, .3f, 0, 0, 0, -1, 0, 0} }, { "room", {0, .2f, 0, 0, 0, -.3f, 0, .3f} }, { "small", {.2f, 0, -.3f, 0, 0, -.8f, 0, 0} },
            { "water", {0, 0, 0, .2f, 0, .3f, 1, .3f} }, { "rain", {.3f, -.2f, .4f, .4f, .3f, .4f, 1, .3f} }, { "fog", {-.4f, 0, .3f, -.4f, -.3f, .6f, .7f, .8f} },
            { "dry", {0, .2f, 0, 0, .2f, -.3f, -1, 0} }, { "dream", {0, 0, 0, -.2f, 0, .4f, .2f, 1} }, { "sleep", {-.5f, .3f, 0, -.6f, -.4f, .2f, 0, 1} },
            { "memory", {-.3f, .4f, 0, -.3f, .2f, .2f, 0, .8f} }, { "wake", {.6f, 0, 0, .5f, 0, 0, 0, -1} }, { "lucid", {.5f, 0, -.2f, 0, -.3f, .3f, 0, .5f} },
            { "longing", {-.4f, .5f, 0, -.5f, 0, .5f, 0, .6f} }, { "home", {-.2f, .8f, 0, -.3f, -.3f, -.4f, 0, .4f} },
            { "star", {.8f, -.3f, -.5f, 0, 0, 1, 0, .5f} }, { "gate", {.3f, -.2f, .3f, .5f, .3f, .5f, 0, 0} }, { "probe", {.6f, -.3f, -.3f, .4f, 0, .6f, 0, 0} },
            { "signal", {.5f, -.2f, 0, .3f, .2f, 0, 0, 0} }, { "ghost", {-.2f, -.2f, -.5f, 0, -.3f, .7f, .2f, .9f} },
        };
        return L;
    }
    /** text -> [kHash dims | kAxes dims]; also returns the words it recognised */
    static std::array<float, kCond> embed (const std::string& text, std::vector<std::string>* words = nullptr)
    {
        std::array<float, kCond> out {}; out.fill (0.0f);
        std::vector<std::string> toks; std::string cur;
        for (char ch : text)
        {
            const unsigned char u = (unsigned char) ch;
            if (std::isalnum (u) || u >= 128 || ch == '\'') cur += (char) std::tolower (u);
            else if (! cur.empty()) { toks.push_back (cur); cur.clear(); }
        }
        if (! cur.empty()) toks.push_back (cur);
        float v[kHash]; int n = 0; bool negate = false;
        std::array<float, kAxes> axes {}; axes.fill (0); int hitsN = 0;
        const auto& L = lexicon();
        for (auto& w : toks)
        {
            if (w == "not" || w == "no" || w == "never" || w == "without" || w == "less") { negate = true; continue; }
            const std::string bw = "<" + w + ">";
            for (int len = 3; len <= 5; ++len)
                for (size_t i = 0; i + (size_t) len <= bw.size(); ++i)
                {
                    bucketVec (fnv (bw.substr (i, (size_t) len)) & 16383u, v);
                    for (int k = 0; k < kHash; ++k) out[(size_t) k] += v[k];
                    ++n;
                }
            bucketVec (fnv ("w:" + w), v);
            for (int k = 0; k < kHash; ++k) out[(size_t) k] += 2 * v[k];
            n += 2;
            // lexicon (plural / -ing / -ed stripped)
            auto find = [&] (std::string s) -> const std::array<float, kAxes>*
            {
                for (int pass = 0; pass < 4; ++pass)
                {
                    if (auto it = L.find (s); it != L.end()) return &it->second;
                    if (s.size() > 4 && s.ends_with ("ing")) s = s.substr (0, s.size() - 3);
                    else if (s.size() > 3 && s.ends_with ("ed")) s = s.substr (0, s.size() - 2);
                    else if (s.size() > 3 && s.ends_with ("s")) s = s.substr (0, s.size() - 1);
                    else if (s.size() > 3 && s.ends_with ("y")) s = s.substr (0, s.size() - 1);
                    else break;
                }
                return nullptr;
            };
            if (auto* a = find (w))
            {
                for (int k = 0; k < kAxes; ++k) axes[(size_t) k] += (negate ? -0.7f : 1.0f) * (*a)[(size_t) k];
                ++hitsN;
                if (words) words->push_back ((negate ? "not " : "") + w);
            }
            negate = false;
        }
        if (n > 0)
        {
            float s = 0; for (int k = 0; k < kHash; ++k) s += out[(size_t) k] * out[(size_t) k];
            s = std::sqrt (s);
            for (int k = 0; k < kHash; ++k) out[(size_t) k] = s > 0 ? out[(size_t) k] / s : 0;
        }
        for (int k = 0; k < kAxes; ++k)
            out[(size_t) (kHash + k)] = hitsN ? std::tanh (axes[(size_t) k] / std::sqrt ((float) hitsN)) : 0.0f;
        return out;
    }
};
} // namespace ns::ml
