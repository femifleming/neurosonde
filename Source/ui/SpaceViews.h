// Neurosonde Space — the space page: the sound field as a probe-scanner sphere (sources, their stereo pairs,
// image-source reflections, the measured intensity vector), the input's goniometer, and the spatial network's
// prediction against what it measures.
#pragma once
#include "../Plugin.h"
#include "Theme.h"

namespace ns::ui
{
static const char* kSpatNames[24] = { "ild 60", "ild 120", "ild 250", "ild 500", "ild 1k", "ild 2k", "ild 4k", "ild 8k",
                                      "coh 60", "coh 120", "coh 250", "coh 500", "coh 1k", "coh 2k", "coh 4k", "coh 8k",
                                      "width", "correlation", "balance", "azimuth", "diffuseness", "motion", "width tilt", "phase spread" };

/** the field seen from outside, as a probe sphere you can turn (drag) — listener at the centre, front = −z on screen */
class SphereView : public juce::Component
{
public:
    explicit SphereView (Processor& p) : proc (p) {}
    void mouseDrag (const juce::MouseEvent& e) override
    {
        const auto d = e.getPosition() - last; last = e.getPosition();
        yaw += d.x * 0.01f; pitch = juce::jlimit (-1.4f, 1.4f, pitch + d.y * 0.01f); repaint();
    }
    void mouseDown (const juce::MouseEvent& e) override { last = e.getPosition(); }
    void mouseDoubleClick (const juce::MouseEvent&) override { yaw = 0.6f; pitch = 0.35f; repaint(); }
    void paint (juce::Graphics& g) override
    {
        const auto t = proc.instance().telemetry();
        auto r = getLocalBounds().toFloat();
        const auto c = r.getCentre(); const float R = std::min (r.getWidth(), r.getHeight()) * 0.4f;
        spin += 0.01f;
        // world: x front, y left, z up → screen
        auto proj = [&] (float x, float y, float z, float* depth = nullptr)
        {
            // camera yaw about z, then pitch about the screen x axis
            const float cy = std::cos (yaw), sy = std::sin (yaw);
            const float X = -y * cy + x * sy, Y = x * cy + y * sy;      // X: screen right, Y: depth (away)
            const float cp = std::cos (pitch), sp = std::sin (pitch);
            const float Z = z * cp - Y * sp, D = z * sp + Y * cp;
            if (depth) *depth = D;
            const float persp = 1.0f / (1.0f + 0.25f * D);
            return juce::Point<float> (c.x + X * R * persp, c.y - Z * R * persp);
        };
        auto dir = [] (float az, float el, float rad) { return std::array<float, 3> { rad * std::cos (el) * std::cos (az), rad * std::cos (el) * std::sin (az), rad * std::sin (el) }; };
        // diffuse halo: how much of the field has no direction
        g.setColour (juce::Colours::white.withAlpha (0.06f * t.fieldDiffuse));
        g.fillEllipse (c.x - R * 1.3f, c.y - R * 1.3f, R * 2.6f, R * 2.6f);
        // lattice: latitudes every 30°, meridians every 30°, far half fainter
        auto curve = [&] (std::function<std::array<float, 3> (float)> fn, Colour col, float w)
        {
            juce::Point<float> prev; float pd = 0;
            for (int i = 0; i <= 96; ++i)
            {
                const auto v = fn ((float) i / 96 * juce::MathConstants<float>::twoPi); float d = 0; const auto p = proj (v[0], v[1], v[2], &d);
                if (i > 0) { g.setColour (col.withMultipliedAlpha ((d + pd) * 0.5f > 0 ? 0.35f : 1.0f)); g.drawLine ({ prev, p }, w); }
                prev = p; pd = d;
            }
        };
        for (int k = -2; k <= 2; ++k)
        {
            const float el = k * juce::MathConstants<float>::pi / 6;
            curve ([&] (float a) { return dir (a, el, 1); }, juce::Colours::white.withAlpha (k == 0 ? 0.35f : 0.12f), k == 0 ? 1.0f : 0.7f);
        }
        for (int k = 0; k < 6; ++k)
        {
            const float az = k * juce::MathConstants<float>::pi / 6;
            curve ([&] (float a) { return std::array<float, 3> { std::cos (a) * std::cos (az), std::cos (a) * std::sin (az), std::sin (a) }; }, juce::Colours::white.withAlpha (0.08f), 0.7f);
        }
        // azimuth ticks + labels on the horizon
        g.setFont (mono (9)); 
        for (int d = 0; d < 360; d += 10)
        {
            const float a = d * juce::MathConstants<float>::pi / 180;
            const auto p0 = proj (std::cos (a), std::sin (a), 0), p1 = proj (std::cos (a) * (d % 90 == 0 ? 1.1f : 1.04f), std::sin (a) * (d % 90 == 0 ? 1.1f : 1.04f), 0);
            g.setColour (juce::Colours::white.withAlpha (d % 90 == 0 ? 0.6f : 0.25f)); g.drawLine ({ p0, p1 }, 1.0f);
            if (d % 90 == 0)
            {
                const auto pl = proj (std::cos (a) * 1.22f, std::sin (a) * 1.22f, 0);
                const char* nm[] = { "front", "left", "back", "right" };
                g.setColour (C::text2); g.drawText (nm[d / 90], juce::Rectangle<float> (pl.x - 30, pl.y - 7, 60, 14), juce::Justification::centred);
            }
        }
        // listener: a small head with a nose
        {
            const auto h = proj (0, 0, 0); g.setColour (juce::Colours::white.withAlpha (0.8f)); g.drawEllipse (h.x - 6, h.y - 6, 12, 12, 1.0f);
            const auto nose = proj (0.09f, 0, 0); g.drawLine ({ h, nose }, 1.0f);
            const auto el = proj (0, 0.07f, 0), er = proj (0, -0.07f, 0); g.fillEllipse (el.x - 1.5f, el.y - 1.5f, 3, 3); g.fillEllipse (er.x - 1.5f, er.y - 1.5f, 3, 3);
        }
        // reflections (first source's six images), small hollow squares on the sphere
        for (int k = 0; k < t.nRefl; ++k)
        {
            const auto v = dir (t.reflections[(size_t) k].first, t.reflections[(size_t) k].second, 1); float d = 0; const auto p = proj (v[0], v[1], v[2], &d);
            g.setColour (C::text2.withAlpha (d > 0 ? 0.3f : 0.7f)); g.drawRect (p.x - 3, p.y - 3, 6.0f, 6.0f, 1.0f);
        }
        // sources: each layer is a pair (two points, width apart) on an arc; distance pulls it toward the centre
        const Colour cols[5] = { juce::Colour (0xffffffff), juce::Colour (0xffd0d0d0), juce::Colour (0xffa8a8a8), juce::Colour (0xff808080), juce::Colour (0xff5a5a5a) };
        const char* names[5] = { "1", "2", "3", "4", "in" };
        for (int s = 0; s < 5; ++s)
        {
            const auto& S = t.sources[(size_t) s];
            if (S.weight <= 0.01f) continue;
            const float rad = juce::jlimit (0.35f, 1.0f, 1.6f / (1.0f + 0.3f * S.dist));
            const auto cl = dir (S.az + S.width / 2, S.el, rad), cr = dir (S.az - S.width / 2, S.el, rad), cm = dir (S.az, S.el, rad);
            juce::Path arc; for (int i = 0; i <= 16; ++i) { const float a = S.az + S.width / 2 - S.width * i / 16.0f; const auto v = dir (a, S.el, rad); const auto p = proj (v[0], v[1], v[2]); if (i == 0) arc.startNewSubPath (p); else arc.lineTo (p); }
            g.setColour (cols[s].withAlpha (0.25f + 0.5f * S.weight)); g.strokePath (arc, juce::PathStrokeType (2.0f));
            float d = 0; const auto pm = proj (cm[0], cm[1], cm[2], &d);
            const auto h = proj (0, 0, 0);
            g.setColour (cols[s].withAlpha (0.18f)); g.drawLine ({ h, pm }, 0.7f);
            for (auto* v : { &cl, &cr }) { const auto p = proj ((*v)[0], (*v)[1], (*v)[2]); g.setColour (cols[s]); g.fillEllipse (p.x - 2.5f, p.y - 2.5f, 5, 5); }
            const float rr = 7 + 5 * S.weight + 1.5f * std::sin (spin * 3 + s);
            g.setColour (cols[s].withAlpha (0.8f)); g.drawEllipse (pm.x - rr, pm.y - rr, rr * 2, rr * 2, 1.0f);
            g.setFont (mono (9.5f)); g.setColour (C::text); g.drawText (names[s], juce::Rectangle<float> (pm.x + rr + 3, pm.y - 7, 30, 14), juce::Justification::centredLeft);
        }
        // the intensity vector of the rendered field (first order, DirAC): where the energy comes from
        {
            const float len = 1.0f - t.fieldDiffuse;
            const auto v = dir (t.fieldAz, t.fieldEl, 0.2f + 0.85f * len);
            const auto p0 = proj (0, 0, 0), p1 = proj (v[0], v[1], v[2]);
            juce::Path a; a.addArrow ({ p0, p1 }, 1.2f, 8, 8);
            g.setColour (C::red.withAlpha (0.85f)); g.fillPath (a);
        }
        // readout
        g.setFont (mono (9.5f)); g.setColour (C::text2);
        auto ro = r.reduced (12, 8);
        auto line = [&] (const juce::String& s) { g.drawText (s, ro.removeFromTop (14), juce::Justification::topLeft); };
        const auto fmt = proc.apvts.getParameter ("sformat")->getCurrentValueAsText();
        line ("output " + fmt + "   order " + proc.apvts.getParameter ("sorder")->getCurrentValueAsText());
        line ("field az " + juce::String (juce::roundToInt (t.fieldAz * 180 / juce::MathConstants<float>::pi)) + juce::String::fromUTF8 ("°  el ") + juce::String (juce::roundToInt (t.fieldEl * 180 / juce::MathConstants<float>::pi))
              + juce::String::fromUTF8 ("°  diffuse ") + juce::String ((int) (t.fieldDiffuse * 100)) + "%");
        g.setColour (C::muted); g.drawText ("drag to turn  /  double-click to reset", r.reduced (12, 8), juce::Justification::bottomLeft);
    }
    float yaw = 0.6f, pitch = 0.35f, spin = 0;
private:
    Processor& proc; juce::Point<int> last;
};

/** the input's stereo image: a goniometer (mid up, side across) and a correlation meter */
class Goniometer : public juce::Component
{
public:
    explicit Goniometer (Processor& p) : proc (p) {}
    void paint (juce::Graphics& g) override
    {
        const auto t = proc.instance().telemetry();
        auto r = getLocalBounds().toFloat();
        auto meter = r.removeFromBottom (24);
        const float s = std::min (r.getWidth(), r.getHeight()) * 0.46f; const auto c = r.getCentre();
        g.setColour (juce::Colours::white.withAlpha (0.12f));
        g.drawLine (c.x - s, c.y, c.x + s, c.y, 0.7f); g.drawLine (c.x, c.y - s, c.x, c.y + s, 0.7f);
        g.drawLine (c.x - s * 0.7f, c.y - s * 0.7f, c.x + s * 0.7f, c.y + s * 0.7f, 0.5f); g.drawLine (c.x + s * 0.7f, c.y - s * 0.7f, c.x - s * 0.7f, c.y + s * 0.7f, 0.5f);
        g.setFont (mono (9)); g.setColour (C::muted);
        g.drawText ("m", juce::Rectangle<float> (c.x - 8, c.y - s - 12, 16, 12), juce::Justification::centred);
        g.drawText ("l", juce::Rectangle<float> (c.x - s * 0.75f - 12, c.y - s * 0.75f - 6, 12, 12), juce::Justification::centred);
        g.drawText ("r", juce::Rectangle<float> (c.x + s * 0.75f, c.y - s * 0.75f - 6, 12, 12), juce::Justification::centred);
        float peak = 1e-4f; for (int k = 0; k < 256; ++k) peak = std::max ({ peak, std::abs (t.scopeL[(size_t) k]), std::abs (t.scopeR[(size_t) k]) });
        const float k0 = 0.9f / peak;
        for (int k = 0; k < 256; ++k)
        {
            const float l = t.scopeL[(size_t) k] * k0, rr = t.scopeR[(size_t) k] * k0;
            const float x = (rr - l) * 0.7071f, y = (l + rr) * 0.7071f;
            g.setColour (C::accent.withAlpha (0.15f + 0.6f * k / 256.0f)); g.fillRect (c.x + x * s, c.y - y * s, 1.5f, 1.5f);
        }
        // correlation: −1 .. +1
        const float corr = t.spatIn[17];
        meter = meter.reduced (10, 6);
        g.setColour (juce::Colours::white.withAlpha (0.15f)); g.fillRect (meter.withHeight (1).withY (meter.getCentreY()));
        for (int i = 0; i <= 8; ++i) { const float x = meter.getX() + meter.getWidth() * i / 8; g.fillRect (x, meter.getCentreY() - (i % 4 == 0 ? 4 : 2), 1.0f, i % 4 == 0 ? 8.0f : 4.0f); }
        const float x = meter.getX() + meter.getWidth() * (corr + 1) / 2;
        g.setColour (corr < 0 ? C::red : juce::Colours::white); g.fillRect (x - 1.5f, meter.getY(), 3.0f, meter.getHeight());
        g.setFont (mono (9)); g.setColour (C::text2);
        g.drawText ("-1", meter.withWidth (20).translated (-22, 0), juce::Justification::centredRight);
        g.drawText ("+1", meter.withX (meter.getRight() + 2).withWidth (20), juce::Justification::centredLeft);
        g.drawText (t.stereoInput ? "stereo input" : "mono / no input", r.reduced (6, 2), juce::Justification::topLeft);
    }
private:
    Processor& proc;
};

/** what the spatial network expects (cyan bar ± σ) against what it measures (white tick), per feature */
class SpatialFeatures : public juce::Component
{
public:
    explicit SpatialFeatures (Processor& p) : proc (p) {}
    void paint (juce::Graphics& g) override
    {
        const auto t = proc.instance().telemetry();
        auto r = getLocalBounds().toFloat();
        auto& st = proc.brain().stats;
        g.setFont (mono (9.5f)); g.setColour (C::text2);
        g.drawText ("steps " + juce::String ((juce::int64) st.spatSteps.load()) + "   nll " + juce::String (st.spatNll.load(), 3) + "   frames " + juce::String ((juce::int64) st.spatFrames.load()),
                    r.removeFromTop (16), juce::Justification::centredLeft);
        {
            const float cov = st.spatCoverage.load();
            g.setColour (std::abs (cov - 0.683f) < 0.12f ? C::text2 : C::red);
            g.drawText ("held out: nll " + juce::String (st.spatValNll.load(), 2) + juce::String::fromUTF8 ("   inside ±1σ ") + juce::String (juce::roundToInt (cov * 100)) + "% (68% if calibrated)",
                        r.removeFromTop (16), juce::Justification::centredLeft);
        }
        g.setColour (C::muted); g.drawText (juce::String::fromUTF8 ("thin: measured   thick: predicted from the latent ± σ"), r.removeFromTop (14), juce::Justification::centredLeft);
        r.removeFromTop (4);
        const float rowH = r.getHeight() / 24;
        for (int j = 0; j < 24; ++j)
        {
            auto row = r.removeFromTop (rowH);
            g.setColour (C::muted); g.drawText (kSpatNames[j], row.removeFromLeft (84), juce::Justification::centredLeft);
            // range per kind of feature
            float lo = 0, hi = 1;
            if (j < 8 || j == 18) { lo = -24; hi = 24; } else if (j == 17 || j == 19 || j == 22) { lo = -1; hi = 1; }
            auto X = [&] (float v) { return row.getX() + row.getWidth() * juce::jlimit (0.0f, 1.0f, (v - lo) / (hi - lo)); };
            const float cy = row.getCentreY();
            g.setColour (juce::Colours::white.withAlpha (0.10f)); g.fillRect (row.getX(), cy, row.getWidth(), 1.0f);
            if (lo < 0) g.fillRect (X (0), cy - 3, 1.0f, 6.0f);
            const float m = t.spatPred[j], sd = t.spatSd[j];
            g.setColour (C::accent.withAlpha (0.25f)); g.fillRect (juce::Rectangle<float>::leftTopRightBottom (X (m - sd), cy - 3, std::max (X (m - sd) + 1, X (m + sd)), cy + 3));
            g.setColour (C::accent); g.fillRect (X (m) - 1, cy - 4, 2.0f, 8.0f);
            g.setColour (juce::Colours::white); g.fillRect (X (t.spatIn[j]) - 0.5f, cy - 5, 1.0f, 10.0f);
        }
    }
private:
    Processor& proc;
};
} // namespace ns::ui
