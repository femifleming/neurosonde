// Finite-difference gradient checks for every trainable model, then a few learning sanity checks.
#include "../Source/ml/Models.h"
#include <cstdio>
using namespace ns::ml;
static int fails = 0;
#define CHECK(c, ...) do { if (c) { std::printf ("  ok   "); std::printf (__VA_ARGS__); std::printf ("\n"); } else { std::printf ("  FAIL "); std::printf (__VA_ARGS__); std::printf ("\n"); ++fails; } } while (0)

// relative error between analytic and numeric gradients over a few random coordinates
template <typename LossFn>
static double check (ParamSet& ps, LossFn loss, int samples = 40)
{
    ps.zeroGrad(); loss (true);
    std::mt19937 rng (3); double worst = 0;
    for (int s = 0; s < samples; ++s)
    {
        Param* p = ps.list[rng() % ps.list.size()];
        const size_t i = rng() % p->size();
        const float a = p->g[i], w0 = p->w[i], h = 1e-3f;
        p->w[i] = w0 + h; const double lp = loss (false);
        p->w[i] = w0 - h; const double lm = loss (false);
        p->w[i] = w0;
        const double num = (lp - lm) / (2 * h);
        const double rel = std::abs (num - a) / std::max (1e-4, std::abs (num) + std::abs (a));
        worst = std::max (worst, rel);
    }
    return worst;
}

int main()
{
    std::mt19937 rng (1);
    std::normal_distribution<float> nd;
    // --- VAE (deterministic: fixed eps via a reseeded rng each call)
    {
        VAE v; v.init (7); v.beta = 0.3f;
        std::vector<Frame> fr (6);
        for (auto& f : fr)
        {
            for (auto& x : f.x) x = nd (rng);
            float s = 0; for (auto& h : f.harm) { h = std::abs (nd (rng)); s += h; } for (auto& h : f.harm) h /= s;
            for (auto& x : f.noise) x = nd (rng);
            for (auto& x : f.cond) x = 0.5f * nd (rng);
            f.voiced = 0.8f;
        }
        std::vector<const Frame*> b; for (auto& f : fr) b.push_back (&f);
        auto loss = [&] (bool) { std::mt19937 r (99); v.ps.zeroGrad(); auto L = v.trainBatch (b, r); return L.total; };
        // trainBatch accumulates grads; the checker needs grads from the first call only
        v.ps.zeroGrad(); { std::mt19937 r (99); v.trainBatch (b, r); }
        Vec g; for (auto* p : v.ps.list) g.insert (g.end(), p->g.begin(), p->g.end());
        double worst = 0; std::mt19937 pick (5);
        for (int s = 0; s < 60; ++s)
        {
            const size_t pi = pick() % v.ps.list.size(); Param* p = v.ps.list[pi];
            const size_t i = pick() % p->size();
            size_t off = 0; for (size_t q = 0; q < pi; ++q) off += v.ps.list[q]->size();
            const float a = g[off + i], w0 = p->w[i];
            // central differences at three steps: a LeakyReLU kink inside the step spoils the big ones, float
            // rounding the small one; the gradient must agree with at least one of them
            double rel = 1e9, num = 0;
            for (float h : { 2e-3f, 1e-3f, 5e-4f })
            {
                p->w[i] = w0 + h; const double lp = loss (false);
                p->w[i] = w0 - h; const double lm = loss (false); p->w[i] = w0;
                const double nh = (lp - lm) / (2 * h);
                // below ~1e-4 a float central difference can't resolve the gradient (the loss is O(1), float carries 7
                // digits, the steps are 5e-4 to 2e-3): nothing to compare
                if (std::abs (nh) + std::abs (a) < 1e-4) { rel = 0; num = nh; break; }
                const double r = std::abs (nh - a) / (std::abs (nh) + std::abs (a));
                if (r < rel) { rel = r; num = nh; }
            }
            if (rel > 0.03) std::printf ("    vae param %zu idx %zu: analytic %.6g numeric %.6g (rel %.3f)\n", pi, i, (double) a, num, rel);
            worst = std::max (worst, rel);
        }
        CHECK (worst < 0.05, "VAE + FiLM + heads + KL gradients match finite differences (worst rel err %.4f), %zu params", worst, v.ps.count());
    }
    // --- latent GRU
    {
        LatentGRU m; m.init (11);
        std::vector<std::vector<std::array<float, kZ>>> seq (3, std::vector<std::array<float, kZ>> (6));
        std::vector<std::array<float, kCond>> cond (3);
        for (auto& s : seq) for (auto& z : s) for (auto& x : z) x = nd (rng);
        for (auto& c : cond) for (auto& x : c) x = 0.3f * nd (rng);
        auto loss = [&] (bool) { m.ps.zeroGrad(); return m.trainBatch (seq, cond); };
        m.ps.zeroGrad(); m.trainBatch (seq, cond);
        Vec g; for (auto* p : m.ps.list) g.insert (g.end(), p->g.begin(), p->g.end());
        double worst = 0; std::mt19937 pick (8);
        for (int s = 0; s < 60; ++s)
        {
            const size_t pi = pick() % m.ps.list.size(); Param* p = m.ps.list[pi];
            const size_t i = pick() % p->size();
            size_t off = 0; for (size_t q = 0; q < pi; ++q) off += m.ps.list[q]->size();
            const float a = g[off + i], w0 = p->w[i], h = 2e-3f;
            p->w[i] = w0 + h; const double lp = loss (false);
            p->w[i] = w0 - h; const double lm = loss (false); p->w[i] = w0;
            const double num = (lp - lm) / (2 * h);
            if (std::abs (num) + std::abs (a) < 1e-4) continue;   // below float finite-difference noise
            const double rel = std::abs (num - a) / (std::abs (num) + std::abs (a));
            if (rel > 0.05) std::printf ("    param %zu idx %zu: analytic %.6g numeric %.6g\n", pi, i, (double) a, num);
            worst = std::max (worst, rel);
        }
        CHECK (worst < 0.05, "GRU (truncated BPTT) + Gaussian head gradients match (worst rel err %.4f), %zu params", worst, m.ps.count());
    }
    // --- learning: the VAE learns two clusters, the GRU learns a sequence, ESN predicts, SOM organises
    {
        VAE v; v.init (2); Adam opt; std::mt19937 r (4);
        std::vector<Frame> data (400);
        for (size_t k = 0; k < data.size(); ++k)
        {
            auto& f = data[k]; const float sgn = k % 2 ? 1.0f : -1.0f;
            for (int i = 0; i < kFeat; ++i) f.x[(size_t) i] = sgn * std::sin (0.3f * i) + 0.1f * nd (r);
            for (auto& h : f.harm) h = 1.0f / kHarm;
            for (auto& x : f.noise) x = sgn * 0.5f; f.cond.fill (0); f.voiced = 0.5f;
        }
        double first = 0, last = 0;
        for (int step = 0; step < 600; ++step)
        {
            std::vector<const Frame*> b; for (int i = 0; i < 32; ++i) b.push_back (&data[r() % data.size()]);
            v.beta = std::min (0.2f, step / 2000.0f);
            v.ps.zeroGrad(); auto L = v.trainBatch (b, r); opt.step (v.ps);
            if (step < 20) first += L.recon / 20; if (step >= 580) last += L.recon / 20;
        }
        CHECK (last < first * 0.15, "VAE learns: reconstruction %.3f -> %.3f", first, last);
        float za[kZ], zb[kZ], c[kCond] = {}; v.encode (data[0].x.data(), c, za); v.encode (data[1].x.data(), c, zb);
        float d = 0; for (int j = 0; j < kZ; ++j) d += (za[j] - zb[j]) * (za[j] - zb[j]);
        CHECK (d > 0.5f, "the two sounds land apart in latent space (distance^2 %.2f)", d);
        CHECK (v.ps.finite(), "weights finite");
    }
    {
        LatentGRU m; m.init (5); Adam opt; opt.lr = 2e-3f;
        auto pat = [] (int t, int j) { return std::sin (0.4f * t + j) ; };
        double first = 0, last = 0;
        for (int step = 0; step < 300; ++step)
        {
            std::vector<std::vector<std::array<float, kZ>>> seq (8, std::vector<std::array<float, kZ>> (17));
            std::vector<std::array<float, kCond>> cond (8); for (auto& c : cond) c.fill (0);
            for (int b = 0; b < 8; ++b) { const int o = (int) (rng() % 50); for (int t = 0; t < 17; ++t) for (int j = 0; j < kZ; ++j) seq[(size_t) b][(size_t) t][(size_t) j] = pat (t + o, j); }
            m.ps.zeroGrad(); const double nll = m.trainBatch (seq, cond); opt.step (m.ps);
            if (step < 10) first += nll / 10; if (step >= 290) last += nll / 10;
        }
        CHECK (last < first - 0.8, "GRU learns a latent trajectory: NLL %.3f -> %.3f", first, last);
    }
    {
        EchoState e; e.init (3); std::vector<float> s (EchoState::N, 0.0f);
        double err0 = 0, err1 = 0; float prev[kZ] = {};
        for (int t = 0; t < 3000; ++t)
        {
            float in[EchoState::I] = {}; for (int j = 0; j < kZ; ++j) in[j] = prev[j];
            e.advance (s.data(), in);
            float target[kZ]; for (int j = 0; j < kZ; ++j) target[j] = std::sin (0.2f * (t + 1) + j);
            float y[kZ]; e.readout (s.data(), y);
            double err = 0; for (int j = 0; j < kZ; ++j) err += (y[j] - target[j]) * (y[j] - target[j]);
            if (t >= 100 && t < 200) err0 += err / 100; if (t >= 2900) err1 += err / 100;
            e.learn (s.data(), target);
            for (int j = 0; j < kZ; ++j) prev[j] = target[j];
        }
        CHECK (err1 < err0 * 0.05, "echo state network + RLS learns to predict: err %.3f -> %.5f", err0, err1);
    }
    {
        SOM m; m.init (1);
        for (int t = 0; t < 20000; ++t) { float z[kZ] = {}; z[0] = std::uniform_real_distribution<float> (-1, 1) (rng); z[1] = std::uniform_real_distribution<float> (-1, 1) (rng); m.learn (z); }
        // topology: neighbours on the map are close in latent space
        double nearD = 0, farD = 0;
        for (int r = 0; r < kSom; ++r) for (int c = 0; c + 1 < kSom; ++c)
        {
            auto d = [&] (int a, int b) { double s = 0; for (int j = 0; j < kZ; ++j) { double e = m.w[(size_t) a * kZ + j] - m.w[(size_t) b * kZ + j]; s += e * e; } return std::sqrt (s); };
            nearD += d (r * kSom + c, r * kSom + c + 1); farD += d (r * kSom + c, ((r + 8) % kSom) * kSom + (c + 7) % kSom);
        }
        CHECK (nearD < farD * 0.3, "SOM keeps topology (neighbour distance %.3f vs far %.3f)", nearD, farD);
    }
    {
        std::vector<std::string> w1, w2;
        auto a = TextEmbed::embed ("dusty warehouse, golden light through broken windows", &w1);
        auto b = TextEmbed::embed ("dust in the warehouse, gold light, broken window", &w2);
        auto c = TextEmbed::embed ("cold glass rain at night, fast pulses", nullptr);
        auto dot = [] (auto& x, auto& y) { double s = 0; for (int i = 0; i < TextEmbed::kHash; ++i) s += x[(size_t) i] * y[(size_t) i]; return s; };
        CHECK (dot (a, b) > dot (a, c) + 0.2, "similar notes embed close (%.2f vs %.2f)", dot (a, b), dot (a, c));
        CHECK (a[TextEmbed::kHash + TextEmbed::Rough] > 0.3f && c[TextEmbed::kHash + TextEmbed::Wet] > 0.3f, "lexicon axes: dusty→rough %.2f, rain→wet %.2f", a[TextEmbed::kHash + TextEmbed::Rough], c[TextEmbed::kHash + TextEmbed::Wet]);
        auto n = TextEmbed::embed ("not bright", nullptr);
        CHECK (n[TextEmbed::kHash + TextEmbed::Bright] < 0, "negation flips: 'not bright' → bright %.2f", n[TextEmbed::kHash + TextEmbed::Bright]);
    }
    // --- spatial net: gradients, and it learns a timbre → stereo-image mapping
    {
        SpatialNet m; m.init (21); m.beta = 0;   // plain NLL so the returned loss is the objective
        const int B = 5; Vec zs ((size_t) B * kZ), cs ((size_t) B * kCond), ts ((size_t) B * kSpatial);
        for (auto& v : zs) v = nd (rng); for (auto& v : cs) v = 0.3f * nd (rng); for (auto& v : ts) v = nd (rng);
        m.ps.zeroGrad(); m.trainBatch (zs, cs, ts, B);
        Vec g; for (auto* p : m.ps.list) g.insert (g.end(), p->g.begin(), p->g.end());
        double worst = 0; std::mt19937 pick (12);
        for (int s2 = 0; s2 < 60; ++s2)
        {
            const size_t pi = pick() % m.ps.list.size(); Param* p = m.ps.list[pi]; const size_t i = pick() % p->size();
            size_t off = 0; for (size_t q = 0; q < pi; ++q) off += m.ps.list[q]->size();
            const float a = g[off + i], w0 = p->w[i];
            double rel = 1e9, num = 0;    // three steps, as for the VAE
            for (float h : { 1e-3f, 5e-4f, 2.5e-4f })
            {
                p->w[i] = w0 + h; m.ps.zeroGrad(); const double lp = m.trainBatch (zs, cs, ts, B);
                p->w[i] = w0 - h; m.ps.zeroGrad(); const double lm = m.trainBatch (zs, cs, ts, B); p->w[i] = w0;
                const double nh = (lp - lm) / (2 * h);
                if (std::abs (nh) + std::abs (a) < 1e-4) { rel = 0; num = nh; break; }
                const double r = std::abs (nh - a) / (std::abs (nh) + std::abs (a));
                if (r < rel) { rel = r; num = nh; }
            }
            if (rel > 0.03) std::printf ("    spatial param %zu idx %zu: analytic %.6g numeric %.6g (rel %.3f)\n", pi, i, (double) a, num, rel);
            worst = std::max (worst, rel);
        }
        CHECK (worst < 0.05, "spatial net (softplus-variance NLL) gradients match (worst rel err %.4f), %zu params", worst, m.ps.count());
        // learn: width = tanh(z0), balance = z1 (+ noise)
        SpatialNet l; l.init (5); Adam opt; std::mt19937 r (2); double first = 0, last = 0;
        for (int step = 0; step < 2500; ++step)
        {
            const int b = 32; Vec z ((size_t) b * kZ), c ((size_t) b * kCond, 0.0f), t ((size_t) b * kSpatial, 0.0f);
            for (int k = 0; k < b; ++k) { for (int j = 0; j < kZ; ++j) z[(size_t) k * kZ + j] = nd (r); t[(size_t) k * kSpatial + 16] = std::tanh (z[(size_t) k * kZ]); t[(size_t) k * kSpatial + 18] = z[(size_t) k * kZ + 1] + 0.1f * nd (r); t[(size_t) k * kSpatial + 20] = z[(size_t) k * kZ + 2] + 1.0f * nd (r); }
            l.ps.zeroGrad(); const double nll = l.trainBatch (z, c, t, b); opt.step (l.ps);
            if (step < 20) first += nll / 20; if (step >= 2480) last += nll / 20;
        }
        float zz[kZ] {}, cc[kCond] {}, mean[kSpatial], lv[kSpatial]; zz[0] = 1.5f; zz[1] = -1.0f; l.infer (zz, cc, mean, lv);
        CHECK (last < first - 0.8 && std::abs (mean[16] - std::tanh (1.5f)) < 0.15f && std::abs (mean[18] + 1.0f) < 0.2f,
               "spatial net learns width/balance from the latent: NLL %.2f -> %.2f, predicts width %.2f (true %.2f), balance %.2f (true -1)", first, last, mean[16], std::tanh (1.5f), mean[18]);
        CHECK (lv[20] - lv[18] > 3.0f && lv[20] - lv[18] < 6.0f, "its variance is real: σ=1.0 vs σ=0.1 targets give a log-variance gap of %.2f (theory: ln 100 = 4.61)", lv[20] - lv[18]);
    }
    std::printf (fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
