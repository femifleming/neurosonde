// Neurosonde Space — the spatial renderer.
//   encoding   each source is a stereo pair (two points ± width/2 around its direction) in Ambisonics (ACN / SN3D,
//              up to 7th order, Ambi.h), with distance gain (1/r beyond 1 m), air absorption (a one-pole whose cut
//              falls with distance) and a near-field spread (higher orders faded as a source gets close / diffuse)
//   room       first-order early reflections of a shoebox (image-source method, Allen & Berkley 1979): six images
//              per source, delayed by their path length, absorbed, encoded from their direction (order ≤ 3);
//              a late tail from an 8-line feedback delay network whose lines are encoded from the cube's corners
//   decoding   AmbiX passthrough; speaker layouts by a sampling decoder with max-rE weights (Zotter & Frank);
//              binaural through virtual loudspeakers on a Fibonacci sphere, each filtered by the Brown & Duda (1998)
//              structural HRTF model: Woodworth ITD, the one-pole/one-zero head shadow, an elevation-dependent
//              pinna notch
#pragma once
#include "Ambi.h"
#include "Synths.h"
#include <Accelerate/Accelerate.h>

namespace ns::dsp
{
enum class OutFormat { Ambix, Binaural, Stereo, Quad, Surround51, Surround714 };

struct SourceCtl
{
    float az = 0, el = 0;      // radians: az + = left, el + = up
    float dist = 2;            // metres
    float width = 0.6f;        // radians between the pair
    float diffuse = 0;         // 0..1: spreads the direction (higher orders faded)
    float send = 0.3f;         // to the room
};

/** max-rE weights per order (Zotter & Frank 2012): g_n = P_n(cos(137.9° / (N + 1.51))) */
inline std::array<float, ambi::kMaxOrder + 1> maxRe (int N)
{
    std::array<float, ambi::kMaxOrder + 1> g {};
    const double x = std::cos (137.9 * M_PI / 180.0 / (N + 1.51));
    double a = 1, b = x; g[0] = 1; if (N >= 1) g[1] = (float) x;
    for (int n = 2; n <= N; ++n) { const double c = ((2 * n - 1) * x * b - (n - 1) * a) / n; a = b; b = c; g[(size_t) n] = (float) c; }   // Legendre recursion
    return g;
}

/** a fractional delay line (power-of-two ring, linear interpolation) */
struct Delay
{
    std::vector<float> d; int w = 0, mask = 0;
    void prepare (int n) { int sz = 1; while (sz < n + 2) sz <<= 1; d.assign ((size_t) sz, 0); w = 0; mask = sz - 1; }
    void push (float x) { d[(size_t) w] = x; w = (w + 1) & mask; }
    float read (float delay) const
    {
        const int di = (int) delay; const float f = delay - (float) di;
        const int i = (w - 1 - di) & mask, j = (i - 1) & mask;
        return d[(size_t) i] * (1 - f) + d[(size_t) j] * f;
    }
    int capacity() const { return mask + 1; }
};

/** Brown & Duda (1998) head-shadow: H(s) = (α s + β)/(s + β), β = 2c/a, α(θ) = 1.05 + 0.95 cos(θ·180/150),
    bilinear-transformed; θ = angle between the source and the ear's axis */
struct HeadShadow
{
    float b0 = 1, b1 = 0, a1 = 0, z = 0;
    void set (float thetaRad, float sr)
    {
        const float a = 0.0875f, c = 343.0f, beta = 2 * c / a;
        const float alpha = 1.05f + 0.95f * std::cos (thetaRad * 180.0f / 150.0f);
        const float k = 2 * sr;   // bilinear
        const float n0 = alpha * k + beta, n1 = -alpha * k + beta, d0 = k + beta, d1 = -k + beta;
        b0 = n0 / d0; b1 = n1 / d0; a1 = d1 / d0;
    }
    float process (float x) { const float y = b0 * x + z; z = b1 * x - a1 * y; return y; }
};

/** a peaking notch (RBJ) for the pinna cue */
struct Notch
{
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;
    void set (float f, float q, float gainDb, float sr)
    {
        const float A = std::pow (10.0f, gainDb / 40), w = 2 * (float) M_PI * f / sr, al = std::sin (w) / (2 * q), cw = std::cos (w);
        const float a0 = 1 + al / A;
        b0 = (1 + al * A) / a0; b1 = -2 * cw / a0; b2 = (1 - al * A) / a0; a1 = -2 * cw / a0; a2 = (1 - al / A) / a0;
    }
    float process (float x) { const float y = b0 * x + z1; z1 = b1 * x - a1 * y + z2; z2 = b2 * x - a2 * y; return y; }
};

class SpatialScene
{
public:
    static constexpr int kSources = Synth::kLayers + 1;   // the four layers + the dry input
    void prepare (float sampleRate, int maxBlock)
    {
        sr = sampleRate; block = maxBlock;
        for (auto& c : hoa) c.assign ((size_t) maxBlock, 0.0f);
        hoaFlat.assign ((size_t) ambi::kCh * (size_t) maxBlock, 0.0f);
        for (auto& d : srcDelay) d.prepare ((int) (sr * 0.25f));
        for (auto& l : air) l = 0; for (auto& l : airR) l = 0;
        // late tail: 8 lines, encoded from the cube's corners
        const int base[8] = { 1103, 1307, 1499, 1693, 1873, 2087, 2297, 2539 };
        for (int k = 0; k < 8; ++k) { fdn[(size_t) k].prepare ((int) (base[k] * sr / 48000.0f * 2.2f) + 8); fdnLen[(size_t) k] = (float) (base[k] * sr / 48000.0f * 2.2f); fdnLp[(size_t) k] = 0; }
        for (int k = 0; k < 8; ++k)
        {
            const float x = (k & 1) ? 1.f : -1.f, y = (k & 2) ? 1.f : -1.f, z = (k & 4) ? 1.f : -1.f;
            cornerSh[(size_t) k] = ambi::sh (std::atan2 (y, x), std::atan2 (z, std::sqrt (x * x + y * y)), ambi::kMaxOrder);
        }
        reflDirs.reserve (16);
        setOrder (order, format, true);
    }
    void setOrder (int o, OutFormat f, bool force = false)
    {
        o = std::clamp (o, 1, ambi::kMaxOrder);
        if (! force && o == order && f == format) return;
        order = o; format = f; nCh = ambi::chFor (order);
        // decoder
        speakers.clear();
        auto add = [&] (float azDeg, float elDeg) { speakers.push_back ({ azDeg * (float) M_PI / 180, elDeg * (float) M_PI / 180 }); };
        switch (format)
        {
            case OutFormat::Stereo: add (30, 0); add (-30, 0); break;
            case OutFormat::Quad: add (45, 0); add (-45, 0); add (135, 0); add (-135, 0); break;
            case OutFormat::Surround51: add (30, 0); add (-30, 0); add (0, 0); add (0, -90); add (110, 0); add (-110, 0); break;          // L R C LFE Ls Rs
            case OutFormat::Surround714: add (30, 0); add (-30, 0); add (0, 0); add (0, -90); add (90, 0); add (-90, 0); add (135, 0); add (-135, 0);
                                         add (45, 45); add (-45, 45); add (135, 45); add (-135, 45); break;                          // L R C LFE Ls Rs Lrs Rrs Ltf Rtf Ltr Rtr
            case OutFormat::Binaural:
            {
                const int L = std::clamp (2 * nCh, 16, 64);
                for (int i = 0; i < L; ++i)   // Fibonacci sphere
                {
                    const float zz = 1 - 2 * (i + 0.5f) / L, r = std::sqrt (1 - zz * zz), phi = i * 2.399963f;
                    speakers.push_back ({ std::atan2 (r * std::sin (phi), r * std::cos (phi)), std::asin (zz) });
                }
                break;
            }
            default: break;
        }
        // sampling decoder with max-rE weights, energy-normalised: D_s = (4π / L) g_n Y_nm(θ_s)
        const auto g = maxRe (order);
        dec.assign (speakers.size(), std::vector<float> ((size_t) nCh, 0));
        float en = 0; for (int n = 0; n <= order; ++n) en += (2 * n + 1) * g[(size_t) n] * g[(size_t) n];
        const float norm = 1.0f / std::sqrt (std::max (1e-6f, en));
        for (size_t s = 0; s < speakers.size(); ++s)
        {
            if (format == OutFormat::Surround51 && s == 3) continue;     // LFE stays silent
            if (format == OutFormat::Surround714 && s == 3) continue;
            const auto y = ambi::sh (speakers[s].az, speakers[s].el, order);
            for (int c = 0; c < nCh; ++c)
            {
                const int n = ambi::orderOf (c);
                // SN3D → N3D for the decoder: multiply by sqrt(2n+1)
                dec[s][(size_t) c] = y[(size_t) c] * g[(size_t) n] * std::sqrt (2.0f * n + 1) * norm;
            }
        }
        // binaural: with many virtual speakers, normalise so a frontal source reaches each ear at about unity —
        // between the coherent (low-frequency) amplitude sum and the incoherent (high-frequency) energy sum
        if (format == OutFormat::Binaural)
        {
            const auto yf = ambi::sh (0, 0, order); double amp = 0, en2 = 0;
            for (auto& D : dec) { double v = 0; for (int c = 0; c < nCh; ++c) v += D[(size_t) c] * yf[(size_t) c]; amp += v; en2 += v * v; }
            const float k = (float) (1.0 / std::sqrt (std::max (1e-9, std::abs (amp)) * std::sqrt (en2)));
            for (auto& D : dec) for (auto& v : D) v *= k;
        }
        decFlat.assign (speakers.size() * (size_t) nCh, 0.0f);
        for (size_t s = 0; s < speakers.size(); ++s) for (int c = 0; c < nCh; ++c) decFlat[s * (size_t) nCh + (size_t) c] = dec[s][(size_t) c];
        spk.assign (speakers.size() * (size_t) block, 0.0f);
        // binaural: Brown & Duda filters per virtual speaker and ear
        if (format == OutFormat::Binaural)
        {
            ears.assign (speakers.size(), {});
            for (size_t s = 0; s < speakers.size(); ++s)
            {
                const float az = speakers[s].az, el = speakers[s].el;
                // direction vector; ears on the ±y axis
                const float vy = std::cos (el) * std::sin (az);
                for (int e = 0; e < 2; ++e)
                {
                    const float earY = e == 0 ? 1.0f : -1.0f;            // left ear at +y
                    const float cosT = std::clamp (vy * earY, -1.0f, 1.0f), theta = std::acos (cosT);
                    auto& E = ears[s][(size_t) e];
                    E.shadow.set (theta, sr);
                    // Woodworth ITD relative to the head centre: a/c (−cos θ) for the near side, a/c (|θ| − π/2) for the far
                    const float a = 0.0875f, c = 343.0f;
                    const float tau = theta < (float) M_PI / 2 ? -a / c * std::cos (theta) : a / c * (theta - (float) M_PI / 2);
                    E.delay = (tau + a / c) * sr;   // ≥ 0
                    E.pinna.set (std::min (0.42f * sr, 6500.0f + 4000.0f * (el / ((float) M_PI / 2))), 4.0f, -10.0f + 6.0f * std::abs (el) / ((float) M_PI / 2), sr);
                    E.line.prepare (64);
                }
            }
        }
        for (auto& r : reflections) r.ready = false;
    }
    int channels() const { return nCh; }
    int ord() const { return order; }
    OutFormat fmt() const { return format; }
    const std::vector<std::pair<float, float>>& reflectionDirs() const { return reflDirs; }

    // room
    float roomSize = 1.0f;   // scales a 9 x 7 x 3.5 m shoebox
    float early = 0.4f, late = 0.3f, decay = 0.6f;

    /** in: per-source stereo buffers. out: nOut channels (AmbiX = nCh channels; others = speakers / 2). */
    void process (const std::array<const float*, kSources>& L, const std::array<const float*, kSources>& R, const std::array<SourceCtl, kSources>& ctl,
                  float* const* out, int nOut, int n)
    {
        if (n > block || hoa[0].size() < (size_t) n) { for (int o = 0; o < nOut; ++o) std::fill (out[o], out[o] + n, 0.0f); return; }   // not prepared for this block
        for (int c = 0; c < nCh; ++c) std::fill (hoa[(size_t) c].begin(), hoa[(size_t) c].begin() + n, 0.0f);
        reflDirs.clear();
        std::array<float, 8> lateIn {};
        const int reflOrder = std::min (order, 3), reflCh = ambi::chFor (reflOrder);
        const float lw = 1.0f / sr * 0.0f;
        (void) lw;
        for (int s = 0; s < kSources; ++s)
        {
            if (! L[(size_t) s]) continue;
            const auto& c = ctl[(size_t) s];
            // direct: the pair
            const float d = std::max (0.3f, c.dist), gain = 1.0f / std::max (1.0f, d);
            const float hw = 0.5f * c.width;
            auto yL = ambi::sh (c.az + hw, c.el, order), yR = ambi::sh (c.az - hw, c.el, order);
            // diffuse / near: fade the higher orders (a wider, vaguer image)
            for (int ch = 0; ch < nCh; ++ch)
            {
                const int nn = ambi::orderOf (ch);
                const float k = std::pow (std::max (0.0f, 1.0f - c.diffuse), (float) nn);
                yL[(size_t) ch] *= k; yR[(size_t) ch] *= k;
            }
            const float airK = 1.0f - std::exp (-2 * (float) M_PI * std::clamp (20000.0f / (1 + 0.6f * d), 1500.0f, 20000.0f) / sr);
            for (int i = 0; i < n; ++i)
            {
                air[(size_t) s] += airK * (L[(size_t) s][i] - air[(size_t) s]);
                airR[(size_t) s] += airK * (R[(size_t) s][i] - airR[(size_t) s]);
                const float l = air[(size_t) s] * gain, r = airR[(size_t) s] * gain;
                for (int ch = 0; ch < nCh; ++ch) hoa[(size_t) ch][(size_t) i] += l * yL[(size_t) ch] + r * yR[(size_t) ch];
                srcDelay[(size_t) s].push (0.5f * (L[(size_t) s][i] + R[(size_t) s][i]));
            }
            // early reflections: the six first-order images of a shoebox (listener at the centre)
            if (early > 0.001f)
            {
                const float Wd = 9.0f * roomSize, Dd = 7.0f * roomSize, Hd = 3.5f * roomSize;
                const float sx = d * std::cos (c.el) * std::cos (c.az), sy = d * std::cos (c.el) * std::sin (c.az), sz = d * std::sin (c.el);
                const float img[6][3] = { { Wd - sx, sy, sz }, { -Wd - sx, sy, sz }, { sx, Dd - sy, sz }, { sx, -Dd - sy, sz }, { sx, sy, Hd - sz }, { sx, sy, -Hd * 0.6f - sz } };
                for (int k = 0; k < 6; ++k)
                {
                    const float x = img[k][0], y = img[k][1], z = img[k][2];
                    const float dist = std::sqrt (x * x + y * y + z * z), az = std::atan2 (y, x), el = std::atan2 (z, std::sqrt (x * x + y * y));
                    if (s == 0) reflDirs.push_back ({ az, el });
                    const float delay = std::min ((float) srcDelay[(size_t) s].capacity() - n - 4, (dist - d) / 343.0f * sr);
                    const float g = early * 0.7f / std::max (1.0f, dist);
                    const auto y2 = ambi::sh (az, el, reflOrder);
                    for (int i = 0; i < n; ++i)
                    {
                        const float x0 = srcDelay[(size_t) s].read (std::max (1.0f, delay + (float) (n - 1 - i))) * g;
                        for (int ch = 0; ch < reflCh; ++ch) hoa[(size_t) ch][(size_t) i] += x0 * y2[(size_t) ch];
                    }
                }
            }
            lateIn[(size_t) s % 8] += c.send;
        }
        // late tail: an 8-line FDN fed by the sources' mono mix, lines encoded from the cube's corners
        if (late > 0.001f)
        {
            const float g = 0.6f + 0.38f * std::clamp (decay, 0.0f, 1.0f);
            for (int i = 0; i < n; ++i)
            {
                float in = 0; for (int s = 0; s < kSources; ++s) if (L[(size_t) s]) in += 0.5f * (L[(size_t) s][i] + R[(size_t) s][i]) * ctl[(size_t) s].send;
                float o[8]; float sum = 0;
                for (int k = 0; k < 8; ++k) { const float v = fdn[(size_t) k].read (fdnLen[(size_t) k]); fdnLp[(size_t) k] += 0.4f * (v - fdnLp[(size_t) k]); o[k] = fdnLp[(size_t) k]; sum += o[k]; }
                const float h = sum * 0.25f;
                // irregular injection gains (they must not sum to zero, or the lines cancel in the low orders)
                static const float inj[8] = { 1.0f, 0.8f, -0.6f, 0.9f, 0.7f, -0.5f, 0.75f, 0.85f };
                for (int k = 0; k < 8; ++k) fdn[(size_t) k].push ((o[k] - h) * g + in * 0.7f * inj[k]);
                for (int k = 0; k < 8; ++k)
                {
                    const float x = o[k] * late * 1.6f;
                    for (int ch = 0; ch < nCh; ++ch) hoa[(size_t) ch][(size_t) i] += x * cornerSh[(size_t) k][(size_t) ch];
                }
            }
        }
        // DirAC-style analysis of the output field (first order): intensity vector I = W·(X, Y, Z), diffuseness
        if (nCh >= 4)
        {
            double ix = 0, iy = 0, iz = 0, e = 0;
            for (int i = 0; i < n; ++i)
            {
                const double w = hoa[0][(size_t) i], y = hoa[1][(size_t) i], z = hoa[2][(size_t) i], x = hoa[3][(size_t) i];
                ix += w * x; iy += w * y; iz += w * z; e += 0.5 * (w * w + x * x + y * y + z * z);
            }
            const double mag = std::sqrt (ix * ix + iy * iy + iz * iz);
            fieldAz = (float) std::atan2 (iy, ix); fieldEl = (float) std::atan2 (iz, std::sqrt (ix * ix + iy * iy));
            fieldDiffuse = (float) std::clamp (1.0 - mag / (e + 1e-12), 0.0, 1.0); fieldEnergy = (float) (e / n);
        }
        decode (out, nOut, n);
    }
    float fieldAz = 0, fieldEl = 0, fieldDiffuse = 1, fieldEnergy = 0;
    /** the HOA bus of the last block (for meters) */
    const std::vector<float>& channel (int c) const { return hoa[(size_t) c]; }

private:
    void decode (float* const* out, int nOut, int n)
    {
        for (int o = 0; o < nOut; ++o) std::fill (out[o], out[o] + n, 0.0f);
        if (format == OutFormat::Ambix)
        {
            for (int c = 0; c < std::min (nOut, nCh); ++c) std::copy (hoa[(size_t) c].begin(), hoa[(size_t) c].begin() + n, out[c]);
            return;
        }
        // speaker feeds for the whole block in one matrix product: [S x nCh] · [nCh x n]
        const int S = (int) speakers.size();
        for (int c = 0; c < nCh; ++c) std::copy (hoa[(size_t) c].begin(), hoa[(size_t) c].begin() + n, hoaFlat.begin() + (long) c * n);
        cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, S, n, nCh, 1.0f, decFlat.data(), nCh, hoaFlat.data(), n, 0.0f, spk.data(), n);
        if (format == OutFormat::Binaural)
        {
            if (nOut < 2) return;
            for (int s = 0; s < S; ++s)
            {
                const float* x = spk.data() + (size_t) s * (size_t) n;
                for (int e = 0; e < 2; ++e)
                {
                    auto& E = ears[(size_t) s][(size_t) e]; float* o = out[e];
                    for (int i = 0; i < n; ++i) { E.line.push (x[i]); o[i] += E.pinna.process (E.shadow.process (E.line.read (E.delay))); }
                }
            }
            return;
        }
        for (int s = 0; s < std::min (nOut, S); ++s) std::copy (spk.begin() + (long) s * n, spk.begin() + (long) (s + 1) * n, out[s]);
    }

    struct Dir { float az, el; };
    struct Ear { HeadShadow shadow; Notch pinna; Delay line; float delay = 0; };
    struct Refl { bool ready = false; };
    float sr = 48000; int block = 512, order = 3, nCh = 16; OutFormat format = OutFormat::Binaural;
    std::array<std::vector<float>, ambi::kCh> hoa;
    std::array<Delay, kSources> srcDelay; std::array<float, kSources> air {}, airR {};
    std::array<Delay, 8> fdn; std::array<float, 8> fdnLen {}, fdnLp {}; std::array<ambi::Coefs, 8> cornerSh {};
    std::vector<Dir> speakers; std::vector<std::vector<float>> dec; std::vector<float> decFlat, spk, hoaFlat; std::vector<std::array<Ear, 2>> ears;
    std::array<Refl, 6> reflections; std::vector<std::pair<float, float>> reflDirs;
};
} // namespace ns::dsp
