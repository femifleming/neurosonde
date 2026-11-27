// Neurosonde — the lab page: the training, measured, at full depth. Seven panels (one component, seven kinds), all
// drawn from ml/Lab.h (written by the trainer during the real optimisation) and the instance's telemetry (the path
// from the latent into the synthesis). Black and white like the rest: hairlines, pixels, monospace numbers.
#pragma once
#include "Widgets.h"
#include "../Plugin.h"

namespace ns::ui
{
inline juce::String U (const char* s) { return juce::String::fromUTF8 (s); }
inline juce::String sci (double v, int sig = 2) { char b[32]; std::snprintf (b, sizeof (b), "%.*e", sig, v); return b; }
inline juce::String fx (double v, int d = 4) { return juce::String (v, d); }

class LabPanel : public juce::Component
{
public:
    enum Kind { Objective, Optimiser, Layers, Inside, Recon, Synth, Log };
    LabPanel (Processor& p, Kind k) : proc (p), kind (k) { setOpaque (false); }

    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        switch (kind)
        {
            case Objective: objective (g, r); break;
            case Optimiser: optimiser (g, r); break;
            case Layers:    layers (g, r); break;
            case Inside:    inside (g, r); break;
            case Recon:     recon (g, r); break;
            case Synth:     synth (g, r); break;
            case Log:       logView (g, r); break;
        }
    }
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails& w) override
    {
        if (kind != Log) return;
        scroll = juce::jmax (0, scroll + (int) std::round (w.deltaY * 12)); repaint();
    }
    void mouseDoubleClick (const juce::MouseEvent&) override { scroll = 0; repaint(); }

private:
    Processor& proc; Kind kind; int scroll = 0;
    ml::Lab& lab() { return proc.brain().lab; }

    // ---- drawing helpers --------------------------------------------------------------------------------------------
    static void text (juce::Graphics& g, const juce::String& s, juce::Rectangle<float> r, juce::Colour c = C::text2, float h = 9.0f, juce::Justification j = juce::Justification::centredLeft)
    { g.setFont (mono (h)); g.setColour (c); g.drawText (s, r, j, false); }
    static void label (juce::Graphics& g, const juce::String& s, float x, float y, float w = 280)
    { text (g, s, { x, y, w, 11 }, C::muted, 8.5f); }
    /** a time series, log or linear, in a frame with min / max read-outs; dash: 0 solid, 1 dashed, 2 dotted */
    static void series (juce::Graphics& g, juce::Rectangle<float> r, const std::vector<float>& v, bool logY, float lo, float hi, int dash, float alpha)
    {
        if (v.size() < 2) return;
        auto map = [&] (float x) { const float a = logY ? std::log10 (std::max (1e-12f, x)) : x, l = logY ? std::log10 (lo) : lo, h = logY ? std::log10 (hi) : hi; return r.getBottom() - juce::jlimit (0.0f, 1.0f, (a - l) / std::max (1e-9f, h - l)) * r.getHeight(); };
        juce::Path p;
        for (size_t i = 0; i < v.size(); ++i)
        {
            const float x = r.getX() + r.getWidth() * (float) i / (float) (v.size() - 1), y = map (v[i]);
            if (i == 0) p.startNewSubPath (x, y); else p.lineTo (x, y);
        }
        g.setColour (juce::Colours::white.withAlpha (alpha));
        if (dash == 0) g.strokePath (p, juce::PathStrokeType (1.0f));
        else { const float d1[] = { 4, 3 }, d2[] = { 1, 2 }; juce::Path q; juce::PathStrokeType (1.0f).createDashedStroke (q, p, dash == 1 ? d1 : d2, 2); g.fillPath (q); }
    }
    static void frame (juce::Graphics& g, juce::Rectangle<float> r, const juce::String& top, const juce::String& bottom)
    {
        g.setColour (juce::Colours::white.withAlpha (0.12f)); g.drawRect (r, 1.0f);
        for (int k = 1; k < 4; ++k) { g.setColour (juce::Colours::white.withAlpha (0.05f)); g.fillRect (r.getX(), r.getY() + r.getHeight() * k / 4, r.getWidth(), 1.0f); }
        text (g, top, { r.getRight() - 70, r.getY() + 1, 68, 9 }, C::muted, 7.5f, juce::Justification::centredRight);
        text (g, bottom, { r.getRight() - 70, r.getBottom() - 10, 68, 9 }, C::muted, 7.5f, juce::Justification::centredRight);
    }
    /** a matrix as pixels: white = positive, grey = negative, brightness = |value| / (2.5 scale) */
    static void matrix (juce::Graphics& g, juce::Rectangle<float> r, const ml::Lab::Mat& M, bool signedMap = true)
    {
        if (M.rows <= 0 || M.cols <= 0 || M.v.size() < (size_t) M.rows * M.cols) return;
        juce::Image img (juce::Image::RGB, M.cols, M.rows, false);
        juce::Image::BitmapData bd (img, juce::Image::BitmapData::writeOnly);
        const float k = 1.0f / (2.5f * std::max (1e-9f, M.scale));
        for (int y = 0; y < M.rows; ++y)
            for (int x = 0; x < M.cols; ++x)
            {
                const float v = M.v[(size_t) y * M.cols + x] * k;
                const float a = std::tanh (std::abs (v));
                const int lvl = (int) (signedMap ? (v >= 0 ? 255 * a : 120 * a) : 255 * std::tanh (std::max (0.0f, v)));
                bd.setPixelColour (x, y, juce::Colour::greyLevel (juce::jlimit (0, 255, lvl) / 255.0f));
            }
        g.setImageResamplingQuality (juce::Graphics::lowResamplingQuality);
        g.drawImage (img, r, juce::RectanglePlacement::stretchToFit);
        g.setColour (juce::Colours::white.withAlpha (0.25f)); g.drawRect (r, 1.0f);
    }
    /** bars (centred at zero when signed) */
    static void bars (juce::Graphics& g, juce::Rectangle<float> r, const float* v, int n, float lo, float hi, float alpha = 0.85f, bool hollow = false)
    {
        const float w = r.getWidth() / n, zero = r.getBottom() - juce::jlimit (0.0f, 1.0f, (0 - lo) / (hi - lo)) * r.getHeight();
        for (int i = 0; i < n; ++i)
        {
            const float y = r.getBottom() - juce::jlimit (0.0f, 1.0f, (v[i] - lo) / (hi - lo)) * r.getHeight();
            const juce::Rectangle<float> b (r.getX() + i * w, std::min (y, zero), std::max (1.0f, w - 1), std::max (1.0f, std::abs (zero - y)));
            g.setColour (juce::Colours::white.withAlpha (alpha));
            if (hollow) g.drawRect (b, 0.8f); else g.fillRect (b);
        }
        g.setColour (juce::Colours::white.withAlpha (0.15f)); g.fillRect (r.getX(), zero, r.getWidth(), 1.0f);
    }
    static void hist (juce::Graphics& g, juce::Rectangle<float> r, const std::array<float, 33>& h)
    {
        const float w = r.getWidth() / 33;
        for (int i = 0; i < 33; ++i) { const float hh = h[(size_t) i] * r.getHeight(); g.setColour (juce::Colours::white.withAlpha (0.8f)); g.fillRect (r.getX() + i * w, r.getBottom() - hh, std::max (1.0f, w - 0.5f), hh); }
        g.setColour (juce::Colours::white.withAlpha (0.15f)); g.fillRect (r.getX(), r.getBottom(), r.getWidth(), 1.0f); g.fillRect (r.getCentreX(), r.getY(), 1.0f, r.getHeight());
    }
    static float norm (const float* v, int n) { double s = 0; for (int i = 0; i < n; ++i) s += (double) v[i] * v[i]; return (float) std::sqrt (s); }

    // ---- 1. the objective ------------------------------------------------------------------------------------------
    void objective (juce::Graphics& g, juce::Rectangle<float> r)
    {
        std::vector<float> tot, rec, harm, noise, kl; ml::Lab::Step s; uint64_t n;
        {
            std::lock_guard<std::mutex> l (lab().m); n = lab().count; s = lab().last;
            const int cnt = (int) std::min<uint64_t> (n, ml::Lab::N);
            for (int i = 0; i < cnt; ++i)
            {
                const auto& q = lab().series[(size_t) ((lab().pos - cnt + i + ml::Lab::N) % ml::Lab::N)];
                tot.push_back (q.total); rec.push_back (q.recon); harm.push_back (0.25f * q.harm); noise.push_back (q.noise); kl.push_back (q.beta * q.kl);
            }
        }
        float y = r.getY();
        const char* eq[] = {
            "L(phi,theta) = rec + 0.25 harm + noise + beta KL",
            "rec   = sum_i w_i (x^_i - x_i)^2 / 84    w: mel 1, chroma .5, scalars 1",
            "harm  = -v sum_k h_k log softmax(h^)_k   48 partials, v = voicing",
            "noise = sum_b (n^_b - n_b)^2 / 32        32 bands, standardised dB",
            "KL    = -1/2 sum_j (1 + log s_j^2 - m_j^2 - s_j^2) / 16",
            "z = m + s * e,  e ~ N(0, I)              reparameterised, backprop through z",
        };
        for (auto* e : eq) { text (g, U (e), { r.getX(), y, r.getWidth(), 11 }, e == eq[0] ? C::text : C::text2, 8.6f); y += 12; }
        y += 6;
        // the terms now: value, weighted contribution, share of the total
        struct T { const char* name; float val, w; int dash; };
        const T terms[] = { { "rec", s.recon, 1.0f, 0 }, { "harm", s.harm, 0.25f, 1 }, { "noise", s.noise, 1.0f, 2 }, { "kl", s.kl, s.beta, 3 } };
        text (g, "term      value      x weight   = in L     share", { r.getX(), y, r.getWidth(), 10 }, C::muted, 8); y += 12;
        const float L = std::max (1e-9f, s.total);
        for (auto& t : terms)
        {
            const float c = t.val * t.w;
            text (g, juce::String (t.name).paddedRight (' ', 9) + fx (t.val, 5).paddedRight (' ', 11) + ("x " + fx (t.w, 3)).paddedRight (' ', 11) + fx (c, 5), { r.getX(), y, r.getWidth(), 10 }, C::text, 8.5f);
            const float bw = 60 * juce::jlimit (0.0f, 1.0f, c / L);
            g.setColour (juce::Colours::white.withAlpha (0.12f)); g.fillRect (r.getRight() - 62, y + 2, 60.0f, 6.0f);
            g.setColour (juce::Colours::white); g.fillRect (r.getRight() - 62, y + 2, bw, 6.0f);
            y += 12;
        }
        text (g, "L total  " + fx (s.total, 5) + "     step " + juce::String ((juce::int64) n), { r.getX(), y, r.getWidth(), 10 }, juce::Colours::white, 8.5f); y += 16;
        // the curves, log scale
        auto pr = juce::Rectangle<float> (r.getX(), y, r.getWidth(), r.getBottom() - y - 14);
        float lo = 1e9f, hi = 1e-9f;
        for (auto* v : { &tot, &rec, &harm, &noise, &kl }) for (float x : *v) if (x > 0) { lo = std::min (lo, x); hi = std::max (hi, x); }
        if (lo > hi) { lo = 1e-3f; hi = 1; }
        lo = std::pow (10.0f, std::floor (std::log10 (lo))); hi = std::pow (10.0f, std::ceil (std::log10 (hi)));
        frame (g, pr, sci (hi, 0), sci (lo, 0));
        // per-step values faint (each is one batch of 32: noisy by nature), their 40-step moving average bright
        auto smooth = [] (const std::vector<float>& v) { std::vector<float> o (v.size()); double e = v.empty() ? 0 : v[0]; for (size_t i = 0; i < v.size(); ++i) { e = 0.95 * e + 0.05 * v[i]; o[i] = (float) e; } return o; };
        series (g, pr, tot, true, lo, hi, 0, 0.16f);
        series (g, pr, smooth (tot), true, lo, hi, 0, 1.0f); series (g, pr, smooth (rec), true, lo, hi, 0, 0.55f);
        series (g, pr, smooth (harm), true, lo, hi, 1, 0.75f); series (g, pr, smooth (noise), true, lo, hi, 2, 0.85f); series (g, pr, smooth (kl), true, lo, hi, 1, 0.4f);
        text (g, "last " + juce::String ((int) tot.size()) + " steps, log10, EMA 0.95:  L bright (raw faint),  rec grey,  harm dashed,  noise dotted,  bKL faint dashed", { r.getX(), r.getBottom() - 11, r.getWidth(), 10 }, C::muted, 7.2f);
    }

    // ---- 2. the optimiser -------------------------------------------------------------------------------------------
    void optimiser (juce::Graphics& g, juce::Rectangle<float> r)
    {
        std::vector<float> lr, gn, us; ml::Lab::Step s; float sps, gruNll, gruG, spNll, spG; long t; int br, bp, bv; bool off; uint64_t rest, pubs, gruN, n; float clipK = 0; int clipped = 0;
        {
            std::lock_guard<std::mutex> l (lab().m); auto& L = lab();
            s = L.last; sps = L.stepsPerSec; t = L.adamT; br = L.batchRecent; bp = L.batchReplay; bv = L.batchVoiced; off = L.offlineMode;
            gruNll = L.gruNll; gruG = L.gruGnorm; spNll = L.spatNll; spG = L.spatGnorm; rest = L.restores; pubs = L.publishes; gruN = L.gruStepsSeen; n = L.count;
            const int cnt = (int) std::min<uint64_t> (n, ml::Lab::N);
            for (int i = 0; i < cnt; ++i) { const auto& q = L.series[(size_t) ((L.pos - cnt + i + ml::Lab::N) % ml::Lab::N)]; lr.push_back (q.lr); gn.push_back (q.gnorm); us.push_back (q.us); clipK += q.clipK; clipped += q.clipK < 1; }
        }
        float y = r.getY();
        auto row = [&] (const juce::String& k, const juce::String& v) { text (g, k, { r.getX(), y, 150, 10 }, C::muted, 8.5f); text (g, v, { r.getX() + 110, y, r.getWidth() - 110, 10 }, C::text, 8.5f, juce::Justification::centredRight); y += 12; };
        text (g, U ("AdamW: m = .9m + .1g   v = .999v + .001g²   w -= lr (m^/(sqrt v^ + 1e-8) + 1e-5 w)"), { r.getX(), y, r.getWidth(), 10 }, C::text2, 7.8f); y += 12;
        text (g, U ("global-norm clip: g <- g min(1, 1/|g|)    lr = max(2e-4, 1e-3 exp(-t/60000)) x plasticity"), { r.getX(), y, r.getWidth(), 10 }, C::text2, 7.8f); y += 12;
        text (g, U ("beta = beta_max min(1, t/3000)  (KL warm-up)"), { r.getX(), y, r.getWidth(), 10 }, C::text2, 7.8f); y += 16;
        row ("adam t", juce::String ((juce::int64) t));
        row ("learning rate", sci (s.lr) + "   beta " + fx (s.beta, 3));
        row ("|g| pre-clip", fx (s.gnorm, 4) + (s.clipK < 1 ? "  clipped x" + fx (s.clipK, 3) : "  (under 1.0)"));
        row ("clipped steps", juce::String (clipped) + " / " + juce::String ((int) gn.size()));
        row ("steps / s", fx (sps, 1) + "   " + juce::String ((int) s.us) + U (" us / step"));
        row ("batch", "32 = " + juce::String (br) + " recent + " + juce::String (bp) + " replay,  " + juce::String (bv) + " voiced");
        row ("mode", off ? "offline (epochs, early stop)" : "online (live + replay, sleep)");
        row ("gru  (bptt 25)", "nll " + fx (gruNll, 4) + "  |g| " + fx (gruG, 3) + "  #" + juce::String ((juce::int64) gruN));
        row ("spatial net", spNll != 0 ? "b-nll " + fx (spNll, 4) + "  |g| " + fx (spG, 3) : juce::String ("(no stereo yet)"));
        row ("snapshots -> audio", juce::String ((juce::int64) pubs) + "   restores " + juce::String ((juce::int64) rest));
        y += 4;
        const float ph = (r.getBottom() - y - 30) / 2;
        auto p1 = juce::Rectangle<float> (r.getX(), y, r.getWidth(), ph);
        float glo = 1e9f, ghi = 1e-9f; for (float x : gn) if (x > 0) { glo = std::min (glo, x); ghi = std::max (ghi, x); }
        if (glo > ghi) { glo = 0.01f; ghi = 10; }
        glo = std::min (0.1f, std::pow (10.0f, std::floor (std::log10 (glo)))); ghi = std::max (10.0f, std::pow (10.0f, std::ceil (std::log10 (ghi))));
        frame (g, p1, sci (ghi, 0), sci (glo, 0));
        { const float yc = p1.getBottom() - (0 - std::log10 (glo)) / (std::log10 (ghi) - std::log10 (glo)) * p1.getHeight(); const float d[] = { 2, 3 }; juce::Path l; l.startNewSubPath (p1.getX(), yc); l.lineTo (p1.getRight(), yc); juce::Path q; juce::PathStrokeType (1).createDashedStroke (q, l, d, 2); g.setColour (juce::Colours::white.withAlpha (0.5f)); g.fillPath (q); text (g, "clip 1.0", { p1.getX() + 3, yc - 10, 60, 9 }, C::text2, 7.5f); }
        series (g, p1, gn, true, glo, ghi, 0, 0.25f);
        { std::vector<float> o (gn.size()); double e = gn.empty() ? 0 : gn[0]; for (size_t i = 0; i < gn.size(); ++i) { e = 0.95 * e + 0.05 * gn[i]; o[i] = (float) e; } series (g, p1, o, true, glo, ghi, 0, 1.0f); }
        text (g, "|g| before clipping (log)", { p1.getX() + 3, p1.getY() + 1, 200, 9 }, C::muted, 7.5f);
        auto p2 = p1.translated (0, ph + 14);
        float llo = 1e9f, lhi = 1e-9f; for (float x : lr) if (x > 0) { llo = std::min (llo, x); lhi = std::max (lhi, x); }
        if (llo > lhi) { llo = 1e-4f; lhi = 1e-3f; }
        llo = std::pow (10.0f, std::floor (std::log10 (llo / 1.5f))); lhi = std::pow (10.0f, std::ceil (std::log10 (lhi * 1.5f)));
        frame (g, p2, sci (lhi, 0), sci (llo, 0));
        series (g, p2, lr, true, llo, lhi, 0, 0.9f);
        text (g, "learning rate (log)", { p2.getX() + 3, p2.getY() + 1, 200, 9 }, C::muted, 7.5f);
    }

    // ---- 3. the layers ----------------------------------------------------------------------------------------------
    void layers (juce::Graphics& g, juce::Rectangle<float> r)
    {
        std::vector<ml::LayerStat> L; uint64_t at;
        { std::lock_guard<std::mutex> l (lab().m); L = lab().layers; at = lab().layersAt; }
        const float cols[] = { 0, 62, 128, 196, 240, 288, 338, 420, 488, 522, 580 };
        const char* head[] = { "layer", "in x out", "activation", "params", "|W| rms", "|dL/dW|", "|dW|/|W| (log)", "act m +- s", "neg", "W hist", "log|g| hist" };
        for (int c = 0; c < 11; ++c) text (g, head[c], { r.getX() + cols[c], r.getY(), 90, 10 }, C::muted, 8);
        float y = r.getY() + 14;
        if (L.empty()) { text (g, "waiting for the first ten optimiser steps...", { r.getX(), y, 400, 12 }, C::text2, 9); return; }
        for (size_t i = 0; i < L.size(); ++i)
        {
            const auto& s = L[i]; if (s.name.empty()) continue;
            if (i == 4 || i == 8) { g.setColour (juce::Colours::white.withAlpha (0.12f)); g.fillRect (r.getX(), y - 2, r.getWidth(), 1.0f); }
            const float x = r.getX();
            text (g, s.name, { x + cols[0], y, 62, 12 }, C::text, 8.5f);
            text (g, juce::String (s.in) + " x " + juce::String (s.out), { x + cols[1], y, 66, 12 }, C::text2, 8.5f);
            text (g, s.act, { x + cols[2], y, 68, 12 }, C::muted, 8.2f);
            text (g, juce::String ((int) s.params), { x + cols[3], y, 44, 12 }, C::text2, 8.5f);
            text (g, fx (s.wRms, 4), { x + cols[4], y, 48, 12 }, C::text, 8.5f);
            text (g, sci (s.gRms, 1), { x + cols[5], y, 50, 12 }, C::text, 8.5f);
            // update ratio on a log bar 1e-6 .. 1e-1 with the 1e-3 rule of thumb marked
            {
                const juce::Rectangle<float> b (x + cols[6], y + 3, 46, 6);
                g.setColour (juce::Colours::white.withAlpha (0.12f)); g.fillRect (b);
                const float f = juce::jlimit (0.0f, 1.0f, (std::log10 (std::max (1e-9f, s.upd)) + 6) / 5);
                g.setColour (juce::Colours::white); g.fillRect (b.withWidth (b.getWidth() * f));
                g.setColour (juce::Colours::white.withAlpha (0.6f)); g.fillRect (b.getX() + b.getWidth() * 0.6f, b.getY() - 2, 1.0f, b.getHeight() + 4);
                text (g, s.upd > 0 ? sci (s.upd, 0) : juce::String ("-"), { b.getRight() + 3, y, 34, 12 }, C::text2, 7.8f);
            }
            text (g, s.hasAct ? fx (s.actMean, 2) + " " + fx (s.actStd, 2) : juce::String ("-"), { x + cols[7], y, 66, 12 }, C::text2, 8.2f);
            text (g, s.hasAct ? juce::String ((int) (s.negFrac * 100)) + "%" : juce::String ("-"), { x + cols[8], y, 34, 12 }, C::text2, 8.2f);
            hist (g, { x + cols[9], y + 1, 52, 10 }, s.wHist);
            hist (g, { x + cols[10], y + 1, 52, 10 }, s.gHist);
            y += 17;
        }
        text (g, U ("|dW|/|W| per optimiser step (tick: 1e-3, the usual healthy rate)  ·  neg: share of LeakyReLU outputs below 0 in the batch  ·  W hist: -max..+max  ·  log|g|: 1e-9..1e-1  ·  measured at step ")
                 + juce::String ((juce::int64) at), { r.getX(), r.getBottom() - 11, r.getWidth(), 10 }, C::muted, 7.6f);
    }

    // ---- 4. inside: weight matrices and a batch's activations ---------------------------------------------------------
    void inside (juce::Graphics& g, juce::Rectangle<float> r)
    {
        std::array<ml::Lab::Mat, 3> W; std::array<ml::Lab::Mat, 6> A;
        { std::lock_guard<std::mutex> l (lab().m); W = lab().W; A = lab().A; }
        const float h = r.getHeight() - 26;
        float x = r.getX();
        for (int k = 0; k < 3; ++k)
        {
            const float w = k == 0 ? h * 84.0f / 128.0f : h * 128.0f / (k == 2 ? 164.0f : 128.0f);
            matrix (g, { x, r.getY() + 12, w, h }, W[(size_t) k]);
            text (g, W[(size_t) k].name.empty() ? juce::String ("-") : juce::String (W[(size_t) k].name), { x, r.getY(), w + 40, 10 }, C::text2, 8);
            text (g, "rms " + fx (W[(size_t) k].scale, 4), { x, r.getY() + 14 + h, w, 10 }, C::muted, 7.6f);
            x += w + 12;
        }
        // activations: rows = the 32 frames of the batch, columns = units
        const float ax = x + 6, aw = r.getRight() - ax;
        const float rh = (h - 5 * 4) / 6;
        for (int k = 0; k < 6; ++k)
        {
            const float yy = r.getY() + 12 + k * (rh + 4);
            const float lw = 44;
            text (g, A[(size_t) k].name.empty() ? juce::String ("-") : juce::String (A[(size_t) k].name), { ax, yy, lw, rh }, C::text2, 8);
            matrix (g, { ax + lw, yy, aw - lw, rh }, A[(size_t) k], true);
        }
        text (g, "activations: 32 frames of the last batch x units", { ax, r.getY(), aw, 10 }, C::text2, 8);
        text (g, "white +  grey -  brightness |v| / 2.5 rms", { ax, r.getY() + 14 + h, aw, 10 }, C::muted, 7.6f);
    }

    // ---- 5. a held-out frame, reconstructed ---------------------------------------------------------------------------
    void recon (juce::Graphics& g, juce::Rectangle<float> r)
    {
        bool ok; std::array<float, ml::kFeat> X, Xh, fe; std::array<float, ml::kHarm> Ht, Hp, he; std::array<float, ml::kNoise> Nt, Np, ne;
        std::array<float, ml::kZ> mu, sg, klD; float err, voiced; uint64_t at;
        {
            std::lock_guard<std::mutex> l (lab().m); auto& L = lab();
            ok = L.exValid; X = L.exX; Xh = L.exXh; Ht = L.exHt; Hp = L.exHp; Nt = L.exNt; Np = L.exNp; mu = L.exMu; sg = L.exSig; err = L.exErr; voiced = L.exVoiced; at = L.exAt;
            fe = L.featErr; he = L.harmErr; ne = L.noiseErr; klD = L.klDim;
        }
        float y = r.getY();
        if (! ok) { text (g, "waiting for a held-out frame (every 25 steps)...", { r.getX(), y, r.getWidth(), 12 }); return; }
        text (g, "held-out frame @" + juce::String ((juce::int64) at) + "   mse " + fx (err, 5) + "   voiced " + fx (voiced, 2), { r.getX(), y, r.getWidth(), 10 }, C::text, 8.5f); y += 13;
        // mel: input solid, reconstruction dashed
        auto mr = juce::Rectangle<float> (r.getX(), y + 10, r.getWidth(), 52);
        label (g, U ("64 mel bands (standardised)   x solid   x^ dashed"), r.getX(), y);
        frame (g, mr, "+3", "-3");
        std::vector<float> a (X.begin(), X.begin() + ml::kMel), b (Xh.begin(), Xh.begin() + ml::kMel);
        series (g, mr, a, false, -3, 3, 0, 0.95f); series (g, mr, b, false, -3, 3, 1, 0.8f);
        y = mr.getBottom() + 6;
        // harmonics: target hollow, prediction solid
        label (g, U ("48 harmonics: target hollow, softmax(h^) solid"), r.getX(), y);
        auto hr = juce::Rectangle<float> (r.getX(), y + 10, r.getWidth(), 34);
        float hm = 0.05f; for (int i = 0; i < ml::kHarm; ++i) hm = std::max ({ hm, Ht[(size_t) i], Hp[(size_t) i] });
        bars (g, hr, Hp.data(), ml::kHarm, 0, hm, 0.9f); bars (g, hr, Ht.data(), ml::kHarm, 0, hm, 0.6f, true);
        y = hr.getBottom() + 6;
        label (g, U ("32 noise bands: n solid, n^ dashed"), r.getX(), y);
        auto nr = juce::Rectangle<float> (r.getX(), y + 10, r.getWidth(), 30);
        frame (g, nr, "+3", "-3");
        series (g, nr, std::vector<float> (Nt.begin(), Nt.end()), false, -3, 3, 0, 0.9f); series (g, nr, std::vector<float> (Np.begin(), Np.end()), false, -3, 3, 1, 0.8f);
        y = nr.getBottom() + 6;
        // scalars
        const char* sn[] = { "f0", "voicing", "loud", "centroid", "spread", "flat", "flux", "rolloff" };
        label (g, U ("8 scalars   x / x^"), r.getX(), y); y += 11;
        for (int i = 0; i < 8; ++i)
        {
            const int k = ml::kMel + ml::kChroma + i;
            text (g, juce::String (sn[i]).paddedRight (' ', 9) + fx (X[(size_t) k], 2).paddedLeft (' ', 6) + fx (Xh[(size_t) k], 2).paddedLeft (' ', 7),
                  { r.getX() + (i % 2) * r.getWidth() * 0.5f, y + (i / 2) * 11.0f, r.getWidth() * 0.5f, 10 }, C::text2, 8);
        }
        y += 4 * 11 + 6;
        // the latent of this frame: mean with ±sigma
        label (g, U ("its latent: m with +-s,   and KL per dimension (batch EMA)"), r.getX(), y);
        auto zr = juce::Rectangle<float> (r.getX(), y + 10, r.getWidth(), 34);
        frame (g, zr, "+3", "-3");
        const float cw = zr.getWidth() / ml::kZ;
        for (int j = 0; j < ml::kZ; ++j)
        {
            auto Y = [&] (float v) { return zr.getBottom() - juce::jlimit (0.0f, 1.0f, (v + 3) / 6) * zr.getHeight(); };
            const float cx = zr.getX() + (j + 0.5f) * cw;
            g.setColour (juce::Colours::white.withAlpha (0.5f)); g.fillRect (cx - 0.5f, Y (mu[(size_t) j] + sg[(size_t) j]), 1.0f, Y (mu[(size_t) j] - sg[(size_t) j]) - Y (mu[(size_t) j] + sg[(size_t) j]));
            g.setColour (juce::Colours::white); g.fillRect (cx - 2, Y (mu[(size_t) j]) - 1, 4.0f, 2.0f);
        }
        y = zr.getBottom() + 2;
        auto kr = juce::Rectangle<float> (r.getX(), y, r.getWidth(), 14);
        float km = 0.05f; for (float v : klD) km = std::max (km, v);
        bars (g, kr, klD.data(), ml::kZ, 0, km, 0.7f);
        y = kr.getBottom() + 6;
        // the error per output (batch EMA), log: which parts of the sound it has learned
        label (g, U ("error per output (batch EMA, log): 84 features | 48 harmonics | 32 noise"), r.getX(), y);
        auto er = juce::Rectangle<float> (r.getX(), y + 10, r.getWidth(), std::max (14.0f, r.getBottom() - y - 10));
        std::vector<float> all; for (float v : fe) all.push_back (v); for (float v : he) all.push_back (v * 10); for (float v : ne) all.push_back (v);
        const float w = er.getWidth() / (float) all.size();
        for (size_t i = 0; i < all.size(); ++i)
        {
            const float f = juce::jlimit (0.0f, 1.0f, (std::log10 (std::max (1e-5f, all[i])) + 4) / 5);
            g.setColour (juce::Colours::white.withAlpha (0.8f)); g.fillRect (er.getX() + i * w, er.getBottom() - f * er.getHeight(), std::max (0.8f, w - 0.3f), f * er.getHeight());
        }
        for (float k : { (float) ml::kFeat, (float) (ml::kFeat + ml::kHarm) }) { g.setColour (juce::Colours::white.withAlpha (0.4f)); g.fillRect (er.getX() + k * w, er.getY(), 1.0f, er.getHeight()); }
    }

    // ---- 6. into the synthesis ------------------------------------------------------------------------------------------
    void synth (juce::Graphics& g, juce::Rectangle<float> r)
    {
        const auto t = proc.instance().telemetry();
        float y = r.getY();
        text (g, U ("z = l z_in + (1-l) z_dream,  z += g (z_probe - z),  z += drift"), { r.getX(), y, r.getWidth(), 10 }, C::text2, 8); y += 12;
        text (g, U ("out = dec(z, notes) -> harmonics | noise | f0, loudness, timbre"), { r.getX(), y, r.getWidth(), 10 }, C::text2, 8); y += 16;
        auto row = [&] (const juce::String& k, const juce::String& v) { text (g, k, { r.getX(), y, 140, 10 }, C::muted, 8.5f); text (g, v, { r.getX() + 100, y, r.getWidth() - 100, 10 }, C::text, 8.5f, juce::Justification::centredRight); y += 12; };
        row ("lucidity l", fx (t.lucidMix, 3) + "   |z_in| " + fx (norm (t.zIn, ml::kZ), 2));
        row ("dream", fx (1 - t.lucidMix, 3) + "   |z_dream| " + fx (norm (t.zDream, ml::kZ), 2));
        row ("probe / route g", t.havePull ? fx (t.gravMix, 3) + "   |pull| " + fx (norm (t.zPull, ml::kZ), 3) : juce::String ("off"));
        row ("drift (O-U)", fx (t.driftMix, 3) + "   |step| " + fx (norm (t.zDrift, ml::kZ), 3));
        // z: 16 dims, the input (hollow) and the dream (tick) under the result (solid)
        auto zr = juce::Rectangle<float> (r.getX(), y + 2, r.getWidth(), 40);
        frame (g, zr, "+4", "-4");
        const float cw = zr.getWidth() / ml::kZ;
        auto Y = [&] (float v) { return zr.getBottom() - juce::jlimit (0.0f, 1.0f, (v + 4) / 8) * zr.getHeight(); };
        for (int j = 0; j < ml::kZ; ++j)
        {
            const float cx = zr.getX() + (j + 0.5f) * cw;
            g.setColour (juce::Colours::white); g.fillRect (cx - 3, std::min (Y (0), Y (t.z[j])), 6.0f, std::abs (Y (t.z[j]) - Y (0)) + 1);
            g.setColour (juce::Colours::white.withAlpha (0.8f)); g.drawRect (juce::Rectangle<float> (7, 3).withCentre ({ cx, Y (t.zIn[j]) }), 0.8f);
            g.fillRect (cx - 5, Y (t.zDream[j]), 10.0f, 1.0f);
        }
        y = zr.getBottom() + 4;
        text (g, "z solid   z_in hollow   z_dream line", { r.getX(), y, r.getWidth(), 9 }, C::muted, 7.6f); y += 14;
        text (g, t.netDrives ? "the network drives the engines  (step " + juce::String ((juce::int64) t.netSteps) + " > 200)"
                             : "warming up: the engines follow the analysis until step 200 (" + juce::String ((juce::int64) t.netSteps) + ")",
              { r.getX(), y, r.getWidth(), 10 }, t.netDrives ? juce::Colours::white : C::text2, 8.5f); y += 14;
        const char* src[] = { "network", "input", "keys" };
        row ("f0", fx (t.synF0, 1) + " hz  (" + src[juce::jlimit (0, 2, t.pitchFrom)] + ")   net " + fx (t.netF0, 1));
        row ("voicing / loud", fx (t.synVoicing, 2) + "   " + fx (t.synLoud, 1) + " db");
        row ("centroid / flat", fx (t.synCentroid, 0) + " hz   " + fx (t.synFlat, 3));
        y += 2;
        label (g, "harmonic distribution -> additive engine (48)", r.getX(), y);
        auto hr = juce::Rectangle<float> (r.getX(), y + 10, r.getWidth(), 30);
        float hm = 0.05f; for (float v : t.synHarm) hm = std::max (hm, v);
        bars (g, hr, t.synHarm, ml::kHarm, 0, hm, 0.9f);
        y = hr.getBottom() + 6;
        label (g, "noise bands -> filtered noise (32, db)", r.getX(), y);
        auto nr = juce::Rectangle<float> (r.getX(), y + 10, r.getWidth(), std::max (16.0f, r.getBottom() - y - 10));
        bars (g, nr, t.synNoise, ml::kNoise, -90, 0, 0.8f);
    }

    // ---- 7. the log ---------------------------------------------------------------------------------------------------
    void logView (juce::Graphics& g, juce::Rectangle<float> r)
    {
        std::vector<std::string> lines; uint64_t seq;
        {
            std::lock_guard<std::mutex> l (lab().m); seq = lab().logSeq;
            const size_t n = lab().log.size(), take = std::min<size_t> (n, 160 + (size_t) scroll);
            for (size_t i = n - take; i < n; ++i) lines.push_back (lab().log[i]);
        }
        const auto f = mono (8.2f); g.setFont (f);
        const float cw = textWidth (f, "0"); const int maxC = std::max (20, (int) ((r.getWidth() - 4) / cw));
        // wrap, newest at the bottom; scroll moves back in time
        std::vector<std::pair<juce::String, bool>> rows;
        for (auto& s : lines)
        {
            const auto js = juce::String::fromUTF8 (s.c_str());
            for (int o = 0; o < js.length(); o += o == 0 ? maxC : maxC - 2)
                rows.push_back ({ (o == 0 ? juce::String() : juce::String ("  ")) + js.substring (o, o + (o == 0 ? maxC : maxC - 2)), o == 0 });
        }
        const float lh = 10.5f; const int fit = (int) (r.getHeight() / lh);
        const int end = juce::jmax (0, (int) rows.size() - scroll), start = juce::jmax (0, end - fit);
        float y = r.getY();
        for (int i = start; i < end; ++i)
        {
            const auto& s = rows[(size_t) i].first;
            const bool hot = s.startsWith ("!!") || s.startsWith ("==") || s.startsWith ("eval") || s.startsWith ("held");
            g.setColour (hot ? juce::Colours::white : s.startsWith ("deep") ? C::text : C::text2.withAlpha (rows[(size_t) i].second ? 1.0f : 0.8f));
            g.drawText (s, juce::Rectangle<float> (r.getX(), y, r.getWidth(), lh), juce::Justification::centredLeft, false);
            y += lh;
        }
        g.setFont (mono (7.6f)); g.setColour (C::muted);
        g.drawText (juce::String ((juce::int64) seq) + " events" + (scroll > 0 ? "   (scrolled back " + juce::String (scroll) + ", double-click: live)" : juce::String()), r.withTop (r.getBottom() + 1).withHeight (10), juce::Justification::centredRight, false);
    }
};
} // namespace ns::ui
