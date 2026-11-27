#include "Editor.h"

namespace ns
{
using namespace ns::ui;
static const char* kPageNames[] = { "constellation", "network", "synthesis", "patch notes", "corpus", "space", "lab" };

static juce::String noteName (float hz)
{
    if (hz <= 0) return "--";
    static const char* n[] = { "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b" };
    const int m = (int) std::lround (69 + 12 * std::log2 (hz / 440.0f));
    return juce::String (n[(m % 12 + 12) % 12]) + juce::String (m / 12 - 1);
}

// ---- layers: engine selector + level / character -----------------------------------------------------------------
struct EngineSelect : juce::Component, private juce::AudioProcessorParameter::Listener, private juce::AsyncUpdater
{
    EngineSelect (juce::RangedAudioParameter& p, int layer) : param (p), k (layer) { param.addListener (this); setRepaintsOnMouseActivity (true); }
    ~EngineSelect() override { param.removeListener (this); }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        g.setColour (juce::Colours::white.withAlpha (isMouseOver() ? 0.08f : 0.0f)); g.fillRect (r);
        g.setColour (juce::Colours::white.withAlpha (0.18f)); g.drawRect (r, 1.0f);
        const int m = (int) std::lround (param.getValue() * ((int) Mode::Count - 1));
        g.setColour (C::text); g.setFont (mono (11)); g.drawText (juce::String (modeName (m)), r.reduced (6, 1).removeFromTop (r.getHeight() * 0.55f), juce::Justification::bottomLeft);
        g.setColour (C::muted); g.setFont (font (9.5f)); g.drawText (juce::String::fromUTF8 (modeLineage (m)).toLowerCase(), r.reduced (6, 1).removeFromBottom (r.getHeight() * 0.45f), juce::Justification::topLeft, true);
        // morph weight
        g.setColour (juce::Colours::white.withAlpha (0.15f)); g.fillRect (r.getRight() - 5, r.getY() + 3, 2.0f, r.getHeight() - 6);
        g.setColour (C::accent); const float h = (r.getHeight() - 6) * weight; g.fillRect (r.getRight() - 5, r.getBottom() - 3 - h, 2.0f, h);
    }
    void mouseDown (const juce::MouseEvent&) override
    {
        juce::PopupMenu menu;
        for (int m = 0; m < (int) Mode::Count; ++m) menu.addItem (m + 1, juce::String (modeName (m)) + juce::String::fromUTF8 ("   ·   ") + juce::String::fromUTF8 (modeLineage (m)));
        menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (this), [this] (int r)
        {
            if (r <= 0) return;
            param.beginChangeGesture(); param.setValueNotifyingHost ((float) (r - 1) / ((int) Mode::Count - 1)); param.endChangeGesture();
        });
    }
    void parameterValueChanged (int, float) override { triggerAsyncUpdate(); }
    void parameterGestureChanged (int, bool) override {}
    void handleAsyncUpdate() override { repaint(); }
    juce::RangedAudioParameter& param; int k; float weight = 0;
};

struct Editor::LayerRow : juce::Component
{
    LayerRow (Processor& p, int layer)
        : on (*p.apvts.getParameter (Processor::lid (layer, "on")), juce::String (layer + 1)),
          sel (*p.apvts.getParameter (Processor::lid (layer, "mode")), layer),
          level (*p.apvts.getParameter (Processor::lid (layer, "level")), "level"),
          character (*p.apvts.getParameter (Processor::lid (layer, "char")), "character", C::star)
    { addAndMakeVisible (on); addAndMakeVisible (sel); addAndMakeVisible (level); addAndMakeVisible (character); }
    void resized() override
    {
        auto r = getLocalBounds(); auto top = r.removeFromTop (28);
        on.setBounds (top.removeFromLeft (30).reduced (0, 3)); top.removeFromLeft (5); sel.setBounds (top);
        r.removeFromTop (2);
        level.setBounds (r.removeFromLeft (r.getWidth() / 2).reduced (0, 0)); character.setBounds (r);
    }
    ParamToggle on; EngineSelect sel; ParamBar level, character;
};

// ---- probe scan results: the nearest remembered moments ------------------------------------------------------------
struct Editor::ScanList : juce::Component
{
    explicit ScanList (Processor& p) : proc (p) {}
    void paint (juce::Graphics& g) override
    {
        const auto t = proc.instance().telemetry();
        auto r = getLocalBounds();
        const int c1 = 44, c2 = 128, c3 = 168;   // signal | moment | note | db
        auto head = r.removeFromTop (16);
        g.setFont (mono (9.5f)); g.setColour (C::muted);
        g.drawText ("signal", head.withWidth (c1), juce::Justification::centredLeft);
        g.drawText ("moment", head.withTrimmedLeft (c1).withWidth (c2 - c1), juce::Justification::centredLeft);
        g.drawText ("note", head.withTrimmedLeft (c2).withWidth (c3 - c2), juce::Justification::centredLeft);
        g.drawText ("db", head.withTrimmedLeft (c3), juce::Justification::centredLeft);
        g.setColour (juce::Colours::white.withAlpha (0.12f));
        for (int x : { c1 - 4, c2 - 4, c3 - 4 }) g.fillRect (x, head.getY(), 1, getHeight() - 16);
        g.fillRect (0, head.getBottom(), getWidth(), 1);
        for (int i = 0; i < t.nHits; ++i)
        {
            auto row = r.removeFromTop (15);
            const auto& h = t.hits[(size_t) i];
            const float s = juce::jlimit (0.0f, 1.0f, h.strength);
            if (i == 0) { g.setColour (juce::Colours::white.withAlpha (0.10f)); g.fillRect (row); g.setColour (juce::Colours::white); g.fillRect (row.getX(), row.getY(), 2, row.getHeight()); }
            g.setFont (mono (10));
            g.setColour (signalColour (s)); g.drawText (juce::String ((int) (s * 100)).paddedLeft (' ', 3) + "%", row.withWidth (c1).withTrimmedLeft (4), juce::Justification::centredLeft);
            g.setColour (i == 0 ? C::text : C::text2);
            g.drawText (juce::String::toHexString (h.unit).paddedLeft ('0', 4), row.withTrimmedLeft (c1).withWidth (c2 - c1), juce::Justification::centredLeft);
            g.drawText (noteName (h.f0), row.withTrimmedLeft (c2).withWidth (c3 - c2), juce::Justification::centredLeft);
            g.drawText (juce::String ((int) h.loud), row.withTrimmedLeft (c3), juce::Justification::centredLeft);
        }
        if (t.nHits == 0) { g.setColour (C::muted); g.setFont (font (10.5f)); g.drawText ("no signatures. play, or feed files", r.reduced (2), juce::Justification::topLeft, true); }
        g.setColour (C::muted); g.setFont (mono (9.5f));
        g.drawText (juce::String (t.corpusUnits) + " moments in memory", getLocalBounds().removeFromBottom (12), juce::Justification::centredLeft);
    }
    Processor& proc;
};

// ---- the latent space, live: where the input is (amber) and where the sound is (cyan) ------------------------------
struct LatentView : juce::Component
{
    explicit LatentView (Processor& p) : proc (p) {}
    void paint (juce::Graphics& g) override
    {
        const auto t = proc.instance().telemetry();
        auto r = getLocalBounds().toFloat();
        auto meters = r.removeFromBottom (30);
        const float rh = juce::jmin (13.0f, r.getHeight() / kZ);
        for (int j = 0; j < kZ; ++j)
        {
            auto row = r.removeFromTop (rh);
            g.setFont (mono (9.5f)); g.setColour (C::muted);
            g.drawText ("z" + juce::String (j + 1).paddedLeft ('0', 2), row.withWidth (26), juce::Justification::centredLeft);
            const float v = t.z[j];
            g.setColour (C::text); g.drawText ((v >= 0 ? "+" : "") + juce::String (v, 3), row.withTrimmedLeft (28).withWidth (50), juce::Justification::centredLeft);
            // 8-bit pattern of the value
            const int q = juce::jlimit (0, 255, (int) ((v + 4.0f) / 8.0f * 255.0f));
            for (int b = 0; b < 8; ++b)
            {
                const bool on = (q >> (7 - b)) & 1;
                juce::Rectangle<float> cell (row.getX() + 82 + b * 8.0f, row.getY() + 2, 6, rh - 4);
                g.setColour (on ? juce::Colours::white : juce::Colours::white.withAlpha (0.10f)); g.fillRect (cell);
            }
            // a mini ruler: where the input is (red) vs the sound (white)
            auto ru = row.withTrimmedLeft (154).reduced (0, rh * 0.3f);
            g.setColour (juce::Colours::white.withAlpha (0.14f)); g.fillRect (ru.getX(), ru.getCentreY(), ru.getWidth(), 1.0f);
            auto xAt = [&] (float x) { return ru.getX() + ru.getWidth() * juce::jlimit (0.0f, 1.0f, (x + 3) / 6); };
            g.setColour (C::red); g.fillRect (xAt (t.zIn[j]) - 0.5f, ru.getY(), 1.5f, ru.getHeight());
            g.setColour (juce::Colours::white); g.fillRect (xAt (v) - 0.5f, ru.getY() - 1, 1.5f, ru.getHeight() + 2);
        }
        auto line = [&] (juce::Rectangle<float> m, const juce::String& k, float v, Colour c)
        {
            g.setFont (mono (9.5f)); g.setColour (C::muted); g.drawText (k, m.withWidth (60), juce::Justification::centredLeft);
            g.setColour (C::text); g.drawText (juce::String ((int) (v * 100)) + "%", m.removeFromRight (36), juce::Justification::centredRight);
            ruler (g, m.withTrimmedLeft (60).reduced (0, 3), 3, juce::Colours::white.withAlpha (0.15f));
            g.setColour (c); g.fillRect (m.getX() + 60, m.getCentreY() - 1, (m.getWidth() - 60) * juce::jlimit (0.0f, 1.0f, v), 2.0f);
        };
        line (meters.removeFromTop (15), "presence", t.presence, juce::Colours::white);
        line (meters, "surprise", t.surprise, C::red);
    }
    Processor& proc;
};

// ---- the network, drawn from its real weights ---------------------------------------------------------------------
struct Editor::NetView : juce::Component
{
    explicit NetView (Processor& p) : proc (p) {}
    void refresh()
    {
        auto& b = proc.brain();
        const Snapshot& S = b.acquireUi();
        auto take = [&] (const Dense& d, Mat& m) { m.in = d.in; m.out = d.out; m.w = d.W.w; };
        take (S.vae.e1, mats[0]); take (S.vae.e2, mats[1]); take (S.vae.eh, mats[2]); take (S.vae.d1, mats[3]); take (S.vae.d2, mats[4]); take (S.vae.dout, mats[5]);
        b.releaseUi();
    }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        auto curves = r.removeFromBottom (150);
        // layers as columns of neurons (sampled), connections coloured by the weights' sign and size
        const char* names[] = { "features 84", "128 · film", "64", "μ,σ 16", "z 16", "128 · film", "128", "out 164" };
        const int sizes[] = { kFeat, 128, 64, 2 * kZ, kZ, 128, 128, kOut };
        const int cols = 8; const float cw = r.getWidth() / cols;
        auto ny = [&] (int col, int i, int shown) { return r.getY() + 26 + (r.getHeight() - 50) * (i + 0.5f) / shown; };
        auto shownN = [] (int n) { return std::min (n, 22); };
        const int mapCol[6][2] = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 4, 5 }, { 5, 6 }, { 6, 7 } };
        for (int m = 0; m < 6; ++m)
        {
            const auto& M = mats[m]; if (M.w.empty()) continue;
            const int c0 = mapCol[m][0], c1 = mapCol[m][1];
            const int s0 = shownN (sizes[c0]), s1 = shownN (sizes[c1]);
            float mx = 1e-6f; for (float w : M.w) mx = std::max (mx, std::abs (w));
            for (int o = 0; o < s1; ++o)
                for (int i = 0; i < s0; ++i)
                {
                    const int oi = o * M.out / s1, ii = i * M.in / s0;
                    if (oi >= M.out || ii >= M.in) continue;
                    const float w = M.w[(size_t) oi * M.in + ii] / mx;
                    if (std::abs (w) < 0.25f) continue;
                    g.setColour (juce::Colours::white.withAlpha (std::abs (w) * (w > 0 ? 0.42f : 0.16f)));   // positive bright, negative faint
                    g.drawLine (r.getX() + cw * (c0 + 0.5f), ny (c0, i, s0), r.getX() + cw * (c1 + 0.5f), ny (c1, o, s1), 0.7f);
                }
        }
        // reparameterisation: μ,σ → z
        for (int i = 0; i < shownN (kZ); ++i) { g.setColour (C::star.withAlpha (0.5f)); g.drawLine (r.getX() + cw * 3.5f, ny (3, i, shownN (2 * kZ)), r.getX() + cw * 4.5f, ny (4, i, shownN (kZ)), 0.8f); }
        const auto t = proc.instance().telemetry();
        for (int c = 0; c < cols; ++c)
        {
            const int s = shownN (sizes[c]);
            for (int i = 0; i < s; ++i)
            {
                float act = 0.3f;
                if (c == 4) act = juce::jlimit (0.0f, 1.0f, 0.5f + 0.25f * t.z[std::min (kZ - 1, i)]);
                if (c == 0) act = juce::jlimit (0.0f, 1.0f, (t.melIn[std::min (kMel - 1, i * kMel / s)] + 90) / 80);
                if (c == 7) act = juce::jlimit (0.0f, 1.0f, (t.mel[std::min (kMel - 1, i * kMel / s)] + 90) / 80);
                const float x = r.getX() + cw * (c + 0.5f), y = ny (c, i, s);
                g.setColour (C::bg); g.fillRect (x - 3.5f, y - 3.5f, 7.0f, 7.0f);
                if (c == 4) { g.setColour (juce::Colours::white); g.drawRect (x - 3.5f, y - 3.5f, 7.0f, 7.0f, 1.0f); }   // the latent code: hollow squares
                g.setColour (juce::Colours::white.withAlpha (0.25f + 0.75f * act)); g.fillRect (x - 2.0f, y - 2.0f, 4.0f, 4.0f);
            }
            g.setColour (C::muted); g.setFont (caps (9.5f));
            g.drawText (juce::String::fromUTF8 (names[c]).toUpperCase(), juce::Rectangle<float> (cw, 16).withPosition (r.getX() + cw * c, r.getY() + 2), juce::Justification::centred);
        }
        g.setFont (font (11)); g.setColour (C::muted);
        g.drawText ("encoder  q(z | x, notes)", juce::Rectangle<float> (cw * 3, 14).withPosition (r.getX(), r.getBottom() - 16), juce::Justification::centred);
        g.drawText (juce::String::fromUTF8 ("decoder  p(x | z, notes)  →  features · harmonics · noise"), juce::Rectangle<float> (cw * 4, 14).withPosition (r.getX() + cw * 4, r.getBottom() - 16), juce::Justification::centred);
        // loss curves
        auto& st = proc.brain().stats;
        g.setColour (C::header); g.fillRect (curves);
        g.setColour (C::edge); g.drawRect (curves, 1);
        auto plot = [&] (const float* h, Colour c, float scale, const juce::String& label, int row)
        {
            juce::Path p; const int n = std::min (Stats::kHist, st.histPos.load()); if (n < 2) return;
            const int start = st.histPos.load() - n;
            for (int i = 0; i < n; ++i)
            {
                const float v = h[(start + i) % Stats::kHist];
                const float x = curves.getX() + 6 + (curves.getWidth() - 12) * i / (Stats::kHist - 1);
                const float y = curves.getBottom() - 8 - juce::jlimit (0.0f, 1.0f, v / scale) * (curves.getHeight() - 30);
                if (i == 0) p.startNewSubPath (x, y); else p.lineTo (x, y);
            }
            g.setColour (c); g.strokePath (p, juce::PathStrokeType (1.3f));
            g.setFont (mono (10.5f)); g.drawText (label, curves.reduced (8, 4).withHeight (14).translated (0, (float) row * 14), juce::Justification::topRight);
        };
        plot (st.reconHist, C::accent, 0.6f, "reconstruction " + juce::String (st.recon.load(), 4), 0);
        float gh[Stats::kHist]; for (int i = 0; i < Stats::kHist; ++i) gh[i] = st.gruHist[i] + 3.0f;
        plot (gh, C::text2, 4.0f, "dream nll " + juce::String (st.gruNll.load(), 3), 1);
        // generalisation: the same eval-mode loss on frames it trains on and on frames it never sees
        {
            const int en = std::min (128, st.evalPos.load()), e0 = st.evalPos.load() - en;
            auto evplot = [&] (const float* h, Colour c, bool dashed)
            {
                if (en < 2) return;
                juce::Path p;
                for (int i = 0; i < en; ++i)
                {
                    const float x = curves.getX() + 6 + (curves.getWidth() - 12) * (float) i / 127;
                    const float y = curves.getBottom() - 8 - juce::jlimit (0.0f, 1.0f, h[(e0 + i) % 128] / 0.6f) * (curves.getHeight() - 30);
                    if (i == 0) p.startNewSubPath (x, y); else p.lineTo (x, y);
                }
                if (dashed) { const float d[] = { 3, 3 }; juce::Path q; juce::PathStrokeType (1.0f).createDashedStroke (q, p, d, 2); g.setColour (c); g.fillPath (q); }
                else { g.setColour (c.withAlpha (0.6f)); g.strokePath (p, juce::PathStrokeType (1.0f)); }
            };
            evplot (st.evTrainHist, juce::Colours::white, false);
            evplot (st.evValHist, C::red, true);
            g.setFont (mono (10.5f)); g.setColour (C::red);
            g.drawText ("held out " + juce::String (st.valRecon.load(), 4) + "  /  train " + juce::String (st.trainRecon.load(), 4) + "  (" + juce::String (st.holdout.load()) + " frames)",
                        curves.reduced (8, 4).withHeight (14).translated (0, 28.0f), juce::Justification::topRight);
            // active units: one bar per latent dimension, height = log variance of its posterior mean over the data
            auto au = curves.reduced (8, 4).withHeight (40).translated (0, 46.0f).withWidth (16 * 7);
            au.setX (curves.getRight() - 8 - au.getWidth());
            for (int j = 0; j < kZ; ++j)
            {
                const float v = st.unitVar[(size_t) j].load(), h = juce::jlimit (0.0f, 1.0f, (std::log10 (v + 1e-6f) + 4) / 5) * au.getHeight();
                g.setColour (v > 0.01f ? C::accent : C::muted); g.fillRect (au.getX() + j * 7.0f, au.getBottom() - h, 5.0f, h);
            }
            g.setColour (juce::Colours::white.withAlpha (0.3f)); g.fillRect (au.getX(), au.getBottom() - (2.0f / 5) * au.getHeight(), au.getWidth(), 1.0f);   // the 0.01 line
            g.setFont (mono (9.5f)); g.setColour (C::text2);
            g.drawText (juce::String (st.activeUnits.load()) + " / 16 active units", au.translated (-au.getWidth() - 8, 0).withWidth (au.getWidth()), juce::Justification::bottomRight);
        }
        g.setColour (C::muted); g.setFont (caps (9.5f)); g.drawText ("TRAINING", curves.reduced (8, 4), juce::Justification::topLeft);
    }
    struct Mat { int in = 0, out = 0; std::vector<float> w; };
    Mat mats[6]; Processor& proc;
};

// ---- memories (brain snapshots) -----------------------------------------------------------------------------------
struct Editor::MemoryList : juce::Component
{
    explicit MemoryList (Processor& p) : proc (p) { refresh(); }
    void refresh() { items = proc.brain().memories(); std::reverse (items.begin(), items.end()); repaint(); }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds();
        if (items.empty()) { g.setColour (C::muted); g.setFont (font (12)); g.drawText ("no memories yet: 'remember' keeps this brain as it is now", r, juce::Justification::topLeft, true); return; }
        for (int i = scroll; i < (int) items.size() && r.getHeight() >= 22; ++i)
        {
            auto row = r.removeFromTop (22);
            if (i == sel) { g.setColour (C::selected); g.fillRect (row); g.setColour (C::accent); g.fillRect (row.removeFromLeft (2)); }
            const auto& m = items[(size_t) i];
            g.setColour (i == sel ? juce::Colours::white : C::text2); g.setFont (font (12.5f));
            g.drawText (m.label.empty() ? juce::String ("(untitled)") : juce::String (m.label), row.withTrimmedLeft (8).withTrimmedRight (150), juce::Justification::centredLeft, true);
            g.setFont (mono (10.5f)); g.setColour (C::muted);
            g.drawText (juce::Time ((juce::int64) m.time * 1000).formatted ("%d %b %H:%M") + "  " + juce::String ((juce::int64) m.steps) + " st", row.removeFromRight (150), juce::Justification::centredRight);
        }
    }
    void mouseDown (const juce::MouseEvent& e) override { sel = scroll + e.y / 22; repaint(); }
    void mouseDoubleClick (const juce::MouseEvent&) override { recallSelected(); }
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails& w) override { scroll = juce::jlimit (0, std::max (0, (int) items.size() - 3), scroll - (w.deltaY > 0 ? 1 : -1)); repaint(); }
    void recallSelected() { if (sel >= 0 && sel < (int) items.size()) { proc.brain().recall (items[(size_t) sel].file); jumped = 1.0f; } }
    Processor& proc; std::vector<Brain::Memory> items; int sel = -1, scroll = 0; float jumped = 0;
};

// ---- notes: the lexicon axes and recognised words -------------------------------------------------------------------
struct Editor::AxesView : juce::Component
{
    explicit AxesView (Processor& p) : proc (p) {}
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds();
        const auto v = TextEmbed::embed (proc.notes.toStdString());
        for (int a = 0; a < TextEmbed::kAxes; ++a)
        {
            auto row = r.removeFromTop (24);
            g.setFont (font (12.5f)); g.setColour (C::text2); g.drawText (TextEmbed::axisName (a), row.removeFromLeft (70), juce::Justification::centredLeft);
            auto bar = row.reduced (4, 9).toFloat();
            g.setColour (juce::Colours::white.withAlpha (0.15f)); g.fillRect (bar.getX(), bar.getCentreY(), bar.getWidth(), 1.0f);
            g.fillRect (bar.getCentreX(), bar.getY() - 2, 1.0f, bar.getHeight() + 4);
            const float x = v[(size_t) (TextEmbed::kHash + a)];
            g.setColour (x >= 0 ? C::accent : C::amber);
            if (x >= 0) g.fillRect (bar.getCentreX(), bar.getY(), bar.getWidth() / 2 * x, bar.getHeight());
            else g.fillRect (bar.getCentreX() + bar.getWidth() / 2 * x, bar.getY(), -bar.getWidth() / 2 * x, bar.getHeight());
        }
        r.removeFromTop (8);
        g.setColour (C::muted); g.setFont (caps (9.5f)); g.drawText ("RECOGNISED", r.removeFromTop (16), juce::Justification::centredLeft);
        auto words = proc.instance().recognisedWords();
        int x = r.getX(), y = r.getY();
        g.setFont (font (12.5f));
        for (auto& w : words)
        {
            const int tw = (int) textWidth (font (12.5f), w) + 16;
            if (x + tw > r.getRight()) { x = r.getX(); y += 24; }
            if (y + 20 > r.getBottom()) break;
            juce::Rectangle<float> chip ((float) x, (float) y, (float) tw, 20);
            g.setColour (C::accent.withAlpha (0.15f)); g.fillPath (cutRect (chip, 4));
            g.setColour (C::accent.withAlpha (0.7f)); g.strokePath (cutRect (chip.reduced (0.5f), 4), juce::PathStrokeType (1));
            g.setColour (C::text); g.drawText (w, chip, juce::Justification::centred);
            x += tw + 6;
        }
    }
    Processor& proc;
};

struct Editor::JournalList : juce::Component
{
    explicit JournalList (Processor& p) : proc (p) { refresh(); }
    void refresh() { lines = proc.journal(); repaint(); }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds();
        if (lines.isEmpty()) { g.setColour (C::muted); g.setFont (font (12)); g.drawText ("the journal is empty: 'log entry' writes the notes here (shared by every instance and the app)", r, juce::Justification::topLeft, true); return; }
        for (int i = scroll; i < lines.size() && r.getHeight() >= 36; ++i)
        {
            auto row = r.removeFromTop (36);
            auto parts = juce::StringArray::fromTokens (lines[i], "\t", "");
            if (i == hover) { g.setColour (C::hover); g.fillRect (row); }
            g.setFont (mono (10.5f)); g.setColour (C::muted);
            g.drawText (parts[0] + "   " + parts[1] + " steps", row.removeFromTop (14).withTrimmedLeft (6), juce::Justification::centredLeft);
            g.setFont (font (12.5f)); g.setColour (C::text2);
            g.drawText (parts[2], row.withTrimmedLeft (6), juce::Justification::centredLeft, true);
        }
    }
    void mouseMove (const juce::MouseEvent& e) override { hover = scroll + e.y / 36; repaint(); }
    void mouseExit (const juce::MouseEvent&) override { hover = -1; repaint(); }
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails& w) override { scroll = juce::jlimit (0, std::max (0, lines.size() - 3), scroll - (w.deltaY > 0 ? 1 : -1)); repaint(); }
    void mouseDown (const juce::MouseEvent& e) override
    {
        const int i = scroll + e.y / 36;
        if (i < lines.size() && onPick) onPick (juce::StringArray::fromTokens (lines[i], "\t", "")[2]);
    }
    std::function<void (const juce::String&)> onPick;
    Processor& proc; juce::StringArray lines; int scroll = 0, hover = -1;
};

/** offline training: train / validation loss per epoch, the best epoch, early stopping */
struct OfflineView : juce::Component
{
    explicit OfflineView (Processor& p) : proc (p) {}
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        auto& o = proc.brain().offline;
        const int st = o.state.load(), ne = std::min (256, o.epoch.load());
        g.setFont (mono (10)); g.setColour (C::text2);
        const char* states[] = { "no offline training yet: 'train offline...' trains on a set of files in epochs, with a validation split and early stopping",
                                 "analysing files", "training", "done", "cancelled", "too little audio" };
        auto top = r.removeFromTop (16);
        juce::String head = states[juce::jlimit (0, 5, st)];
        if (st >= 2 && st != 5)
            head << "   epoch " << o.epoch.load() << " / " << o.epochs.load() << "   train " << juce::String (o.trainLoss.load(), 4) << "   validation " << juce::String (o.valLoss.load(), 4)
                 << "   best " << juce::String (o.bestVal.load(), 4) << " @ " << o.bestEpoch.load() << "   " << o.trainFrames.load() << " / " << o.valFrames.load() << " frames"
                 << (o.spatFrames.load() > 0 ? juce::String ("   stereo ") + juce::String (o.spatFrames.load()) : juce::String());
        g.drawText (head, top, juce::Justification::centredLeft, true);
        r.removeFromTop (4);
        g.setColour (juce::Colours::white.withAlpha (0.08f)); g.drawRect (r, 1.0f);
        if (ne < 1) return;
        float hi = 1e-6f; for (int i = 0; i < ne; ++i) hi = std::max ({ hi, o.trainCurve[i], o.valCurve[i] });
        auto X = [&] (int i) { return r.getX() + 8 + (r.getWidth() - 16) * (float) i / (float) std::max (1, std::max (ne, o.epochs.load()) - 1); };
        auto Y = [&] (float v) { return r.getBottom() - 6 - (r.getHeight() - 12) * v / hi; };
        for (int pass = 0; pass < 2; ++pass)
        {
            juce::Path p; const float* c = pass == 0 ? o.trainCurve : o.valCurve;
            for (int i = 0; i < ne; ++i) { if (i == 0) p.startNewSubPath (X (i), Y (c[i])); else p.lineTo (X (i), Y (c[i])); }
            if (pass == 0) { g.setColour (juce::Colours::white.withAlpha (0.7f)); g.strokePath (p, juce::PathStrokeType (1.0f)); }
            else { const float d[] = { 3, 3 }; juce::Path q; juce::PathStrokeType (1.2f).createDashedStroke (q, p, d, 2); g.setColour (C::red); g.fillPath (q); }
            for (int i = 0; i < ne; ++i) { g.setColour (pass == 0 ? juce::Colours::white : C::red); g.fillRect (X (i) - 1, Y (c[i]) - 1, 2.0f, 2.0f); }
        }
        const int be = o.bestEpoch.load() - 1;
        if (be >= 0) { g.setColour (C::accent); g.drawLine (X (be), r.getY() + 4, X (be), r.getBottom() - 4, 1.0f); g.setFont (mono (9)); g.drawText ("best", juce::Rectangle<float> (X (be) + 3, r.getY() + 3, 40, 12), juce::Justification::centredLeft); }
    }
    Processor& proc;
};

struct Editor::JobList : juce::Component
{
    explicit JobList (Processor& p) : proc (p) {}
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds();
        const auto jobs = proc.jobs();
        if (jobs.empty()) { g.setColour (C::muted); g.setFont (font (12.5f)); g.drawText ("no files yet", r, juce::Justification::topLeft); return; }
        for (int i = (int) jobs.size(); --i >= 0 && r.getHeight() >= 24;)
        {
            const auto& j = *jobs[(size_t) i];
            auto row = r.removeFromTop (24);
            g.setColour (j.train ? C::red : j.process ? C::star : C::accent); g.setFont (caps (9.5f));
            g.drawText (j.train ? "TRAIN" : j.process ? "PROCESS" : "FEED", row.removeFromLeft (64).withTrimmedLeft (4), juce::Justification::centredLeft);
            g.setColour (C::text2); g.setFont (font (12.5f));
            const juce::String name = j.train ? j.file.getFileName() + "  (" + juce::String (j.files.size()) + " files)" : j.file.getFileName();
            g.drawText (name, row.removeFromLeft (row.getWidth() - 330), juce::Justification::centredLeft, true);
            const float p = j.state == 2 ? 1.0f : j.progress.load();
            auto bar = row.removeFromLeft (100).reduced (4, 9);
            g.setColour (juce::Colours::white.withAlpha (0.12f)); g.fillRect (bar);
            g.setColour (j.state == 3 ? C::danger : signalColour (p)); g.fillRect (bar.withWidth ((int) (bar.getWidth() * p)));
            g.setFont (mono (10.5f)); g.setColour (C::muted);
            g.drawText (j.state == 0 ? "queued" : j.state == 1 ? juce::String ((int) (p * 100)) + "%" : j.result, row, juce::Justification::centredLeft, true);
        }
    }
    Processor& proc;
};

// ---------------------------------------------------------------------------------------------------------------
Editor::Editor (Processor& p) : juce::AudioProcessorEditor (p), proc (p)
{
    space = std::make_unique<MapView> (proc);
    addAndMakeVisible (*space);
    space->overlay = [this] (juce::Graphics& g) { paintPanels (g); };
    map = space.get();
    // ---- constellation
    {
        auto& pg = pages[0];
        for (int k = 0; k < Synth::kLayers; ++k) layerRows.push_back (make<LayerRow> (pg, proc, k));
        make<ChoiceRow> (pg, P ("combine"), "combine");
        make<ParamBar> (pg, P ("morph"), "morph");
        make<ChoiceRow> (pg, P ("morphsrc"), "morph by");
        make<ChoiceRow> (pg, P ("dream"), "dream with");
        make<ParamToggle> (pg, P ("route"), "autopilot", C::good);
        make<ParamBar> (pg, P ("routespeed"), "speed", C::good);
        latent = make<LatentView> (pg, proc);
        make<ParamBar> (pg, P ("lucid"), "lucidity");
        make<ParamBar> (pg, P ("temp"), "temperature");
        make<ParamBar> (pg, P ("drift"), "drift");
        make<ParamBar> (pg, P ("gravity"), "gravity");
        scan = make<ScanList> (pg, proc);
    }
    // ---- network
    {
        auto& pg = pages[1];
        net = make<NetView> (pg, proc);
        make<ChoiceRow> (pg, P ("learnmode"), "learning");
        make<ParamToggle> (pg, P ("listen"), "listen");
        make<ParamBar> (pg, P ("plastic"), "plasticity");
        make<ParamBar> (pg, P ("sleep"), "sleep (replay when quiet)");
        make<ParamBar> (pg, P ("betamax"), juce::String::fromUTF8 ("kl weight β"));
        make<ParamBar> (pg, P ("gate"), "listen above");
        memories = make<MemoryList> (pg, proc);
        auto* rem = make<CutButton> (pg, "remember"); rem->onClick = [this] { proc.brain().remember (proc.notes.upToFirstOccurrenceOf ("\n", false, false).substring (0, 60).toStdString()); memories->refresh(); };
        auto* rec = make<CutButton> (pg, "recall"); rec->onClick = [this] { memories->recallSelected(); };
        auto* fgt = make<CutButton> (pg, "forget all", C::danger);
        fgt->onClick = [this]
        {
            auto opts = juce::MessageBoxOptions().withIconType (juce::MessageBoxIconType::WarningIcon).withTitle ("Forget everything?")
                          .withMessage ("The shared brain goes back to random weights. Memories you saved stay.").withButton ("Forget", 1).withButton ("Cancel", 0);
            juce::AlertWindow::showAsync (opts, [this] (int r) { if (r == 1) proc.brain().reset(); });
        };
    }
    // ---- synthesis
    {
        auto& pg = pages[2];
        for (int k = 0; k < Synth::kLayers; ++k)
        {
            make<ParamToggle> (pg, P (Processor::lid (k, "on")), "layer " + juce::String (k + 1));
            make<ParamBar> (pg, P (Processor::lid (k, "level")), "level");
            make<ParamBar> (pg, P (Processor::lid (k, "pan")), "pan");
            make<ParamBar> (pg, P (Processor::lid (k, "char")), "character", C::star);
        }
        for (auto* id : { "lucid", "temp", "drift", "gravity", "morphrate" }) make<ParamBar> (pg, P (id));
        make<ChoiceRow> (pg, P ("pitchsrc"), "pitch from");
        for (auto* id : { "transpose", "pitchfollow" }) make<ParamBar> (pg, P (id));
        for (auto* id : { "grainsize", "spray", "focus", "ring" }) make<ParamBar> (pg, P (id), juce::String(), C::star);
        make<ChoiceRow> (pg, P ("space"), "similarity");
        make<ChoiceRow> (pg, P ("playback"), "mosaic");
        for (auto* id : { "imprint", "tilt", "width", "spin", "nebula", "decay" }) make<ParamBar> (pg, P (id));
        for (auto* id : { "ingain", "gate", "mix", "out", "notesinf" }) make<ParamBar> (pg, P (id));
    }
    // ---- notes
    {
        auto& pg = pages[3];
        notesEd = make<juce::TextEditor> (pg);
        notesEd->setMultiLine (true, true); notesEd->setReturnKeyStartsNewLine (true);
        notesEd->setFont (font (17)); notesEd->setColour (juce::TextEditor::backgroundColourId, C::bg.brighter (0.03f));
        notesEd->setColour (juce::TextEditor::textColourId, C::text); notesEd->setColour (juce::TextEditor::outlineColourId, C::edge);
        notesEd->setColour (juce::TextEditor::focusedOutlineColourId, C::topLine); notesEd->setColour (juce::CaretComponent::caretColourId, C::accent);
        notesEd->setColour (juce::TextEditor::highlightColourId, C::accent.withAlpha (0.3f));
        notesEd->setBorder ({ 14, 16, 14, 16 });
        notesEd->setTextToShowWhenEmpty ("patch notes: write what you hear, what you want, a dream, a room. the words steer the network (and it learns what they sound like).", C::muted);
        notesEd->setText (proc.notes, false);
        notesEd->onTextChange = [this] { proc.setNotes (notesEd->getText()); axes->repaint(); };
        axes = make<AxesView> (pg, proc);
        auto* log = make<CutButton> (pg, "log entry"); log->onClick = [this] { proc.logJournal(); journalList->refresh(); memories->refresh(); };
        journalList = make<JournalList> (pg, proc);
        journalList->onPick = [this] (const juce::String& t) { notesEd->setText (t, true); };
    }
    // ---- corpus
    {
        auto& pg = pages[4];
        auto* feed = make<CutButton> (pg, "feed files..."); auto* proc_ = make<CutButton> (pg, "process files...", C::star);
        auto pick = [this] (bool process)
        {
            auto fc = std::make_shared<juce::FileChooser> (process ? "Process through the network" : "Feed to the network", juce::File(), "*.wav;*.aif;*.aiff;*.flac;*.mp3;*.ogg");
            fc->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::canSelectDirectories | juce::FileBrowserComponent::canSelectMultipleItems,
                             [this, fc, process] (const juce::FileChooser& c) { juce::StringArray s; for (auto& f : c.getResults()) s.add (f.getFullPathName()); proc.addFiles (s, process); });
        };
        feed->onClick = [pick] { pick (false); }; proc_->onClick = [pick] { pick (true); };
        auto* train = make<CutButton> (pg, "train offline...", C::red);
        train->onClick = [this]
        {
            auto& off = proc.brain().offline;
            if (off.state == 1 || off.state == 2) { off.cancel = true; return; }   // a second press cancels
            auto fc = std::make_shared<juce::FileChooser> ("Train offline on these files (folders are searched)", juce::File(), "*.wav;*.aif;*.aiff;*.flac;*.mp3;*.ogg");
            fc->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::canSelectDirectories | juce::FileBrowserComponent::canSelectMultipleItems,
                             [this, fc] (const juce::FileChooser& c) { juce::StringArray s; for (auto& f : c.getResults()) s.add (f.getFullPathName()); if (! s.isEmpty()) proc.trainOnFiles (s); });
        };
        auto* reveal = make<CutButton> (pg, "open output folder"); reveal->onClick = [this] { proc.outputFolder().createDirectory(); proc.outputFolder().revealToUser(); };
        jobList = make<JobList> (pg, proc);
        offlineView = make<OfflineView> (pg, proc);
        make<ParamBar> (pg, P ("epochs"), "offline epochs (max)", C::red);
        make<ChoiceRow> (pg, P ("space"), "similarity");
        make<ChoiceRow> (pg, P ("playback"), "playback");
        for (auto* id : { "ring", "grainsize", "spray", "focus", "pitchfollow" }) make<ParamBar> (pg, P (id), juce::String(), C::star);
    }
#if NS_SPATIAL
    // ---- space
    {
        auto& pg = pages[5];
        make<ChoiceRow> (pg, P ("sformat"), "output");
        make<ChoiceRow> (pg, P ("sorder"), "order");
        make<ParamBar> (pg, P ("follow"), "follow the network", C::accent);
        for (auto* id : { "span", "elev", "lspread", "orbit", "distance", "yaw", "stilt" }) make<ParamBar> (pg, P (id));
        for (auto* id : { "room", "early", "late", "rdecay" }) make<ParamBar> (pg, P (id), juce::String(), C::star);
        sphere = make<ui::SphereView> (pg, proc);
        gonio = make<ui::Goniometer> (pg, proc);
        spatFeat = make<ui::SpatialFeatures> (pg, proc);
    }
#endif
    // ---- lab: the training, measured
    {
        auto& pg = pages[kLab];
        using K = ui::LabPanel::Kind;
        for (auto k : { K::Objective, K::Optimiser, K::Layers, K::Inside, K::Log, K::Synth, K::Recon }) make<ui::LabPanel> (pg, proc, k);
    }
    // header buttons: presets, capture
    {
        auto pb = std::make_unique<CutButton> ("presets");
        pb->onClick = [this]
        {
            juce::PopupMenu m;
            for (int i = 0; i < proc.getNumPrograms(); ++i) m.addItem (i + 1, proc.getProgramName (i), true, i == proc.getCurrentProgram());
            m.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (presetButton.get()), [this] (int r) { if (r > 0) proc.setCurrentProgram (r - 1); });
        };
        presetButton = std::move (pb); addAndMakeVisible (*presetButton);
        auto cb = std::make_unique<CutButton> ("rec", C::danger);
        cb->setClickingTogglesState (false);
        auto* raw = cb.get();
        cb->onClick = [this, raw] { proc.toggleCapture(); raw->setToggleState (proc.capturing(), juce::dontSendNotification); if (! proc.capturing() && proc.lastCapture.existsAsFile()) proc.lastCapture.revealToUser(); };
        captureButton = std::move (cb); addAndMakeVisible (*captureButton);
    }
    built = true;
    setSize (1280, 800);
    setPage (0);
    startTimerHz (20);
}

Editor::~Editor() = default;

void Editor::setPage (int p)
{
    { const auto t = tabOrder(); page = std::find (t.begin(), t.end(), p) != t.end() ? p : 0; }
    for (int k = 0; k < kPages; ++k) for (auto* c : pages[k]) c->setVisible (k == page);
    space->showWorld = page == 0; space->dim = page == 0 ? 0.0f : 0.5f;
    if (page == 1) { net->refresh(); memories->refresh(); }
    if (page == 3) journalList->refresh();
    resized(); repaint();
}

void Editor::resized()
{
    if (! built) return;
    auto body = getLocalBounds().withTrimmedTop (40).withTrimmedBottom (26).reduced (10, 0);
    for (auto& pp : panels) for (auto& r : pp) r = {};
    auto place = [] (std::vector<juce::Component*>& v, size_t& i, juce::Rectangle<int>& area, int h, int gap = 4) { if (i < v.size()) v[i++]->setBounds (area.removeFromTop (h)); area.removeFromTop (gap); };
    space->setBounds (getLocalBounds());
    // everything is laid out mirror-symmetrically about the vertical centre line
    presetButton->setBounds (210, 5, 76, 20);
    captureButton->setBounds (getWidth() - 210 - 76, 5, 76, 20);
    // constellation: two equal columns of overlays on the space, the hud in the middle
    {
        auto& v = pages[0]; size_t i = 0;
        auto full = getLocalBounds().withTrimmedTop (36).withTrimmedBottom (28);
        auto left = full.withWidth (262).translated (14, 0);
        auto right = full.withWidth (262).withX (getWidth() - 276);
        panels[0][0] = left.removeFromTop (346); panelTitle[0][0] = "layers";
        auto a = panels[0][0].reduced (8, 0).withTrimmedTop (26);
        for (int k = 0; k < 4; ++k) place (v, i, a, 56, 4);
        place (v, i, a, 22, 4); place (v, i, a, 26, 2); place (v, i, a, 22);
        left.removeFromTop (8);
        panels[0][1] = left; panelTitle[0][1] = "dream";
        a = panels[0][1].reduced (8, 0).withTrimmedTop (26);
        place (v, i, a, 22, 6);
        auto rowA = a.removeFromTop (26); v[i++]->setBounds (rowA.removeFromLeft (92)); rowA.removeFromLeft (8); v[i++]->setBounds (rowA);
        a.removeFromTop (8); v[i++]->setBounds (a.withTrimmedBottom (6));
        auto hud = juce::Rectangle<int> (0, 0, 440, 124).withCentre ({ getWidth() / 2, getHeight() - 28 - 70 });
        panels[0][3] = hud; panelTitle[0][3] = "hud";
        // four sliders, two by two (lucidity | temperature / drift | gravity)
        auto hr = hud.reduced (10, 0).withTrimmedTop (30);
        const int cw = (hr.getWidth() - 16) / 2;
        for (int k = 0; k < 4; ++k)
            v[i++]->setBounds (hr.getX() + (k % 2) * (cw + 16), hr.getY() + (k / 2) * 44, cw, 38);
        panels[0][4] = right.removeFromTop (346); panelTitle[0][4] = "overview";
        v[i++]->setBounds (panels[0][4].reduced (8, 0).withTrimmedTop (26).withTrimmedBottom (6));
        right.removeFromTop (8);
        panels[0][5] = right; panelTitle[0][5] = "learning";
    }
    const int side = 300;   // the side columns on every full page
    // network: learning | the network | memories
    {
        auto& v = pages[1]; size_t i = 0; auto b = body;
        auto left = b.removeFromLeft (side); b.removeFromLeft (10);
        auto right = b.removeFromRight (side); b.removeFromRight (10);
        panels[1][0] = b; panelTitle[1][0] = "network";
        v[i++]->setBounds (b.reduced (10, 0).withTrimmedTop (30).withTrimmedBottom (10));
        panels[1][1] = left; panelTitle[1][1] = "learning";
        auto a = left.reduced (8, 0).withTrimmedTop (30);
        place (v, i, a, 24, 6);
        auto tg = a.removeFromTop (28); v[i++]->setBounds (tg.removeFromLeft (110)); a.removeFromTop (8);
        for (int k = 0; k < 4; ++k) place (v, i, a, 40, 2);
        panels[1][2] = right; panelTitle[1][2] = "memories";
        a = right.reduced (8, 0).withTrimmedTop (30);
        auto btns = a.removeFromBottom (34).withTrimmedBottom (6);
        v[i++]->setBounds (a.withTrimmedBottom (6));
        const int bw = (btns.getWidth() - 16) / 3;
        v[i++]->setBounds (btns.removeFromLeft (bw)); btns.removeFromLeft (8); v[i++]->setBounds (btns.removeFromLeft (bw)); btns.removeFromLeft (8); v[i++]->setBounds (btns);
    }
    // synthesis: four columns, each split in two at the same heights as its mirror column
    {
        auto& v = pages[2]; size_t i = 0; auto b = body;
        const int cw = (b.getWidth() - 30) / 4;
        juce::Rectangle<int> col[4]; for (int k = 0; k < 4; ++k) { col[k] = b.removeFromLeft (cw); b.removeFromLeft (10); }
        const int outerSplit = (body.getHeight() - 10) / 2, innerSplit = 240;
        // column 0: layers 1 2 / layers 3 4
        panels[2][0] = col[0].removeFromTop (outerSplit); panelTitle[2][0] = "layers 1 2";
        col[0].removeFromTop (10);
        panels[2][6] = col[0]; panelTitle[2][6] = "layers 3 4";
        juce::Rectangle<int> la;
        for (int k = 0; k < 4; ++k)
        {
            if (k % 2 == 0) la = panels[2][k == 0 ? 0 : 6].reduced (8, 0).withTrimmedTop (30);
            auto& a = la;
            auto t = a.removeFromTop (26); v[i++]->setBounds (t.removeFromLeft (100)); a.removeFromTop (4);
            for (int q = 0; q < 3; ++q) place (v, i, a, 38, 2);
            a.removeFromTop (10);
        }
        // column 1: dream / pitch
        panels[2][1] = col[1].removeFromTop (innerSplit); panelTitle[2][1] = "dream";
        auto a = panels[2][1].reduced (8, 0).withTrimmedTop (30);
        for (int k = 0; k < 5; ++k) place (v, i, a, 38, 2);
        col[1].removeFromTop (10);
        panels[2][2] = col[1]; panelTitle[2][2] = "pitch";
        a = col[1].reduced (8, 0).withTrimmedTop (30);
        place (v, i, a, 24, 8);
        for (int k = 0; k < 2; ++k) place (v, i, a, 38, 2);
        // column 2: grains / mosaic
        panels[2][3] = col[2].removeFromTop (innerSplit); panelTitle[2][3] = juce::String::fromUTF8 ("memory · grains");
        a = panels[2][3].reduced (8, 0).withTrimmedTop (30);
        for (int k = 0; k < 4; ++k) place (v, i, a, 38, 2);
        col[2].removeFromTop (10);
        panels[2][7] = col[2]; panelTitle[2][7] = juce::String::fromUTF8 ("mosaic · tape");
        a = col[2].reduced (8, 0).withTrimmedTop (30);
        place (v, i, a, 24, 8); place (v, i, a, 24, 8);
        // column 3: spectral / input output
        panels[2][4] = col[3].removeFromTop (outerSplit); panelTitle[2][4] = juce::String::fromUTF8 ("spectral · stargate · nebula");
        a = panels[2][4].reduced (8, 0).withTrimmedTop (30);
        for (int k = 0; k < 6; ++k) place (v, i, a, 38, 2);
        col[3].removeFromTop (10);
        panels[2][5] = col[3]; panelTitle[2][5] = juce::String::fromUTF8 ("input · output");
        a = col[3].reduced (8, 0).withTrimmedTop (30);
        for (int k = 0; k < 5; ++k) place (v, i, a, 38, 2);
    }
    // notes: what the notes say | the notes | journal
    {
        auto& v = pages[3]; size_t i = 0; auto b = body;
        auto left = b.removeFromLeft (side); b.removeFromLeft (10);
        auto right = b.removeFromRight (side); b.removeFromRight (10);
        panels[3][0] = b; panelTitle[3][0] = "patch notes";
        v[i++]->setBounds (b.reduced (1).withTrimmedTop (22).withTrimmedBottom (24));
        panels[3][1] = left; panelTitle[3][1] = "what the notes say";
        auto a = left.reduced (10, 0).withTrimmedTop (30);
        v[i++]->setBounds (a.removeFromTop (300));
        v[i++]->setBounds (a.removeFromTop (36).withTrimmedTop (6).withSizeKeepingCentre (130, 30));
        panels[3][2] = right; panelTitle[3][2] = "journal";
        v[i++]->setBounds (right.reduced (8, 0).withTrimmedTop (30).withTrimmedBottom (6));
    }
    // space: scene / room | the field | spatial network / input image
    if (NS_SPATIAL)
    {
        auto& v = pages[5]; size_t i = 0; auto b = body;
        auto left = b.removeFromLeft (side + 40); b.removeFromLeft (10);
        auto right = b.removeFromRight (side + 40); b.removeFromRight (10);
        const int split = 410;
        panels[5][0] = left.removeFromTop (split); panelTitle[5][0] = "scene";
        left.removeFromTop (10); panels[5][4] = left; panelTitle[5][4] = "room";
        auto a = panels[5][0].reduced (8, 0).withTrimmedTop (30);
        place (v, i, a, 24, 6); place (v, i, a, 24, 10);
        for (int k = 0; k < 8; ++k) place (v, i, a, 36, 2);
        a = panels[5][4].reduced (8, 0).withTrimmedTop (30);
        for (int k = 0; k < 4; ++k) place (v, i, a, 36, 2);
        panels[5][1] = b; panelTitle[5][1] = "field";
        v[i++]->setBounds (b.reduced (2).withTrimmedTop (24));
        panels[5][3] = right.removeFromTop (split); panelTitle[5][3] = "spatial network";
        right.removeFromTop (10); panels[5][2] = right; panelTitle[5][2] = "input image";
        v[i++]->setBounds (panels[5][2].reduced (8, 0).withTrimmedTop (28).withTrimmedBottom (6));
        v[i++]->setBounds (panels[5][3].reduced (10, 0).withTrimmedTop (30).withTrimmedBottom (8));
    }
    // lab: objective / optimiser | layers / inside / log | into the synthesis / a held-out frame
    {
        auto& v = pages[kLab]; auto b = body;
        auto left = b.removeFromLeft (side); b.removeFromLeft (10);
        auto right = b.removeFromRight (side); b.removeFromRight (10);
        const int split = 362;
        panels[kLab][0] = left.removeFromTop (split); panelTitle[kLab][0] = "objective"; left.removeFromTop (10);
        panels[kLab][1] = left; panelTitle[kLab][1] = "optimiser";
        panels[kLab][5] = right.removeFromTop (split); panelTitle[kLab][5] = juce::String::fromUTF8 ("latent → synthesis"); right.removeFromTop (10);
        panels[kLab][6] = right; panelTitle[kLab][6] = "held-out frame";
        panels[kLab][2] = b.removeFromTop (258); panelTitle[kLab][2] = "layers"; b.removeFromTop (10);
        panels[kLab][3] = b.removeFromTop (206); panelTitle[kLab][3] = "inside the network"; b.removeFromTop (10);
        panels[kLab][4] = b; panelTitle[kLab][4] = "training log";
        const int order[] = { 0, 1, 2, 3, 4, 5, 6 };
        for (size_t k = 0; k < v.size(); ++k) v[k]->setBounds (panels[kLab][order[k]].reduced (10, 0).withTrimmedTop (30).withTrimmedBottom (k == 4 ? 16 : 8));
    }
    // corpus: memory | files | offline training
    {
        auto& v = pages[4]; size_t i = 0; auto b = body;
        auto left = b.removeFromLeft (side); b.removeFromLeft (10);
        auto right = b.removeFromRight (side); b.removeFromRight (10);
        panels[4][0] = b; panelTitle[4][0] = "files";
        auto a = b.reduced (10, 0).withTrimmedTop (34);
        auto btn = a.removeFromTop (30);
        const int bw = (btn.getWidth() - 24) / 4;
        for (int k = 0; k < 4; ++k) { v[i++]->setBounds (btn.removeFromLeft (bw)); btn.removeFromLeft (8); }
        a.removeFromTop (150);   // drop zone drawn in paint
        v[i++]->setBounds (a.withTrimmedBottom (8));                        // job list
        panels[4][2] = right; panelTitle[4][2] = "offline training";
        auto r = right.reduced (8, 0).withTrimmedTop (30);
        auto off = v[i++]; auto ep = v[i++];
        ep->setBounds (r.removeFromTop (38)); r.removeFromTop (10);
        off->setBounds (r.withTrimmedBottom (8));
        panels[4][1] = left; panelTitle[4][1] = "memory";
        a = left.reduced (8, 0).withTrimmedTop (30);
        place (v, i, a, 24, 10); place (v, i, a, 24, 12);
        for (int k = 0; k < 5; ++k) place (v, i, a, 38, 2);
    }
}

void Editor::paint (juce::Graphics& g) { g.fillAll (C::bg); }

void Editor::paintPanels (juce::Graphics& g)
{
    // header and footer strips (under the header buttons)
    g.setColour (juce::Colours::black.withAlpha (0.7f)); g.fillRect (getLocalBounds().removeFromTop (30)); g.fillRect (getLocalBounds().removeFromBottom (22));
    // panels: translucent black under the hairline frames
    for (int k = 0; k < 8; ++k)
        if (! panels[page][k].isEmpty())
        {
            g.setColour (juce::Colours::black.withAlpha (0.88f)); g.fillRect (panels[page][k]);
            drawWindow (g, panels[page][k], panelTitle[page][k], true, {}, page * 16 + k + 1);
        }
    if (page == 0)
    {
        auto& st2 = proc.brain().stats;
        const auto star = space->currentStarName();
        auto tl = juce::Rectangle<int> (0, 0, 560, 34).withCentre ({ getWidth() / 2, 63 });
        g.setFont (font (14, 1)); g.setColour (C::text);
        g.drawText (star.isNotEmpty() ? star : "node " + juce::String (space->currentStar()).paddedLeft ('0', 3), tl.removeFromTop (17), juce::Justification::centred);
        g.setFont (mono (9.5f)); g.setColour (C::text2);
        g.drawText ("route " + juce::String ((int) proc.route.size()) + " waypoints  /  autopilot " + (proc.apvts.getRawParameterValue ("route")->load() > 0.5f ? "engaged" : "off")
                    + "  /  " + juce::String ((juce::int64) st2.framesLearned.load()) + " frames heard", tl, juce::Justification::centred);
        auto lr = panels[0][5].reduced (10, 0).withTrimmedTop (28);
        auto row = [&] (const juce::String& k, const juce::String& v2)
        { auto rr = lr.removeFromTop (15); g.setFont (mono (9.5f)); g.setColour (C::muted); g.drawText (k, rr, juce::Justification::centredLeft); g.setColour (C::text); g.drawText (v2, rr, juce::Justification::centredRight); };
        row ("steps", juce::String ((juce::int64) st2.vaeSteps.load()));
        row ("reconstruction", juce::String (st2.recon.load(), 4));
        row ("kl / beta", juce::String (st2.kl.load(), 3) + " / " + juce::String (st2.beta.load(), 3));
        row ("dream nll", juce::String (st2.gruNll.load(), 3));
        row ("replay", juce::String (st2.replaySize.load()));
        row ("state", st2.sleeping ? "sleeping / replay" : "learning");
        row ("merges", juce::String (st2.merges.load()));
    }
    // corpus drop zone
    if (page == 4)
    {
        auto z = panels[4][0].reduced (10, 0).withTrimmedTop (72).withHeight (140).toFloat();
        juce::Path p; p.addRectangle (z); const float dash[] = { 6, 4 }; juce::Path d; juce::PathStrokeType (1).createDashedStroke (d, p, dash, 2);
        g.setColour (dragOver ? C::accent : C::edge.brighter (0.4f)); g.fillPath (d);
        if (dragOver) { g.setColour (C::accent.withAlpha (0.08f)); g.fillRect (z); }
        g.setColour (C::text2); g.setFont (font (15, 1)); g.drawText ("drop audio files or folders here", z.withTrimmedBottom (40), juce::Justification::centred);
        g.setColour (C::muted); g.setFont (font (12.5f));
        auto zl = z.withTrimmedTop (64);
        for (auto* line : { "drop: feed (learn + remember)", "alt-drop: process (render through the network)", "shift-drop: train offline" })
            g.drawText (line, zl.removeFromTop (16), juce::Justification::centred);
        g.drawFittedText ("renders go to " + proc.outputFolder().getFullPathName(), zl.toNearestInt().reduced (10, 0), juce::Justification::centred, 1);
    }
    if (page == 3)
    {
        g.setColour (C::muted); g.setFont (font (11.5f));
        g.drawFittedText ("the text becomes a 32-d vector (hashed character n-grams + a lexicon of 8 axes) that conditions the network through FiLM layers, and steers the sound directly",
                          panels[3][0].withTrimmedTop (panels[3][0].getHeight() - 24).reduced (10, 0), juce::Justification::centred, 2);
    }
}

void Editor::paintOverChildren (juce::Graphics& g)
{

    // header: translucent, lowercase, minimal
    auto h = getLocalBounds().removeFromTop (30);
    g.setColour (juce::Colours::white.withAlpha (0.14f)); g.fillRect (h.getX(), h.getBottom() - 1, h.getWidth(), 1);
    {
        const juce::Point<float> c (20, 15);
        // the mark: a 4 × 4 bit pattern (n, as in the icon)
        static const int bits[4] = { 0b1001, 0b1101, 0b1011, 0b1001 };
        for (int yy = 0; yy < 4; ++yy) for (int xx = 0; xx < 4; ++xx)
        { g.setColour (juce::Colours::white.withAlpha ((bits[yy] >> (3 - xx)) & 1 ? 1.0f : 0.12f)); g.fillRect (c.x - 7 + xx * 3.5f, c.y - 7 + yy * 3.5f, 2.5f, 2.5f); }
        g.setColour (C::text); g.setFont (font (13, 1)); g.drawText (NS_SPATIAL ? "neurosonde space" : "neurosonde", 34, 0, 150, 30, juce::Justification::centredLeft);
    }
    const auto order = tabOrder();
    int tabsW = 0; for (size_t i = 0; i < order.size(); ++i) tabsW += (int) textWidth (mono (10.5f), kPageNames[order[i]]) + 20 + (i ? 4 : 0);
    int x = (getWidth() - tabsW) / 2;   // the tabs centred: the header mirrors about the middle
    for (int k : order)
    {
        const juce::String t (kPageNames[k]);
        const int w = (int) textWidth (mono (10.5f), t) + 20;
        tabArea[k] = { x, 0, w, 30 };
        if (k == page) { g.setColour (juce::Colours::white); g.fillRect (tabArea[k].reduced (2, 8)); }
        g.setFont (mono (10.5f)); g.setColour (k == page ? juce::Colours::black : C::text2);
        g.drawText (t, tabArea[k], juce::Justification::centred);
        x += w + 4;
    }
    auto& st = proc.brain().stats;
    g.setFont (mono (10)); g.setColour (C::text2);
    const juce::String status = juce::String (st.sleeping ? "sleeping" : "learning") + "  /  " + juce::String ((juce::int64) st.vaeSteps.load()) + " steps  /  "
                              + juce::String ((juce::int64) st.framesLearned.load()) + " frames";
    g.drawText (status, h.reduced (14, 0), juce::Justification::centredRight);
    g.setColour (st.sleeping ? C::muted : juce::Colours::white); g.fillRect ((float) getWidth() - 14 - textWidth (mono (10), status) - 12, 12.0f, 5.0f, 5.0f);

    // footer: data line
    auto f = getLocalBounds().removeFromBottom (22);
    g.setColour (juce::Colours::white.withAlpha (0.14f)); g.fillRect (f.getX(), f.getY(), f.getWidth(), 1);
    const auto t = proc.instance().telemetry();
    g.setFont (mono (9.5f)); g.setColour (C::muted);
    g.drawText ("recon " + juce::String (st.recon.load(), 4) + "   kl " + juce::String (st.kl.load(), 3) + "   lr " + juce::String (st.lr.load(), 5)
                + "   nll " + juce::String (st.gruNll.load(), 3) + "   replay " + juce::String (st.replaySize.load()) + "   surprise " + juce::String ((int) (t.surprise * 100))
                + "%   in " + juce::String ((int) t.inLoud) + " db   f0 " + juce::String ((int) t.f0) + " hz   " + noteName (t.f0), f.reduced (12, 0), juce::Justification::centredLeft);
    g.drawText (juce::String (proc.brain().folder()).replace (juce::File::getSpecialLocation (juce::File::userHomeDirectory).getFullPathName(), "~"),
                f.reduced (12, 0), juce::Justification::centredRight);

    // "stargate jump" after recalling a memory
    if (memories && memories->jumped > 0.01f)
    {
        // a strobe of horizontal scan lines, thinning out
        const float a = memories->jumped;
        juce::Random rnd ((juce::int64) (a * 997));
        for (int y = 0; y < getHeight(); y += 3) if (rnd.nextFloat() < a * 0.6f) { g.setColour (juce::Colours::white.withAlpha (a * 0.7f)); g.fillRect (0, y, getWidth(), 1 + rnd.nextInt (2)); }
    }
}

void Editor::mouseDown (const juce::MouseEvent& e)
{
    for (int k : tabOrder()) if (tabArea[k].contains (e.getPosition())) { setPage (k); return; }
}

void Editor::filesDropped (const juce::StringArray& files, int, int)
{
    dragOver = false;
    const auto mods = juce::ModifierKeys::getCurrentModifiers();
    if (mods.isShiftDown()) proc.trainOnFiles (files);
    else proc.addFiles (files, mods.isAltDown());
    if (page != 4) setPage (4);
    repaint();
}

void Editor::timerCallback()
{
    ++tick;
    const auto t = proc.instance().telemetry();
    for (int k = 0; k < (int) layerRows.size(); ++k) { layerRows[(size_t) k]->sel.weight = t.weights[k]; layerRows[(size_t) k]->sel.repaint(); }
    if (page == 0) { scan->repaint(); latent->repaint(); }
    if (page == 1) { if (tick % 10 == 0) net->refresh(); net->repaint(); }
    if (page == 4) { jobList->repaint(); offlineView->repaint(); }
    if (page == 5 && sphere) { sphere->repaint(); gonio->repaint(); spatFeat->repaint(); }
    if (page == kLab) for (auto* c : pages[kLab]) c->repaint();
    if (memories->jumped > 0) { memories->jumped = std::max (0.0f, memories->jumped - 0.05f); repaint(); }
    if (page != 0) space->repaint();
}

bool Editor::snapshot (Processor& p, int pg, const juce::File& png)
{
    Editor ed (p);
    ed.setPage (pg);
    ed.map->refreshNow();
    auto img = ed.createComponentSnapshot (ed.getLocalBounds(), true, 1.0f);
    png.deleteFile();
    juce::FileOutputStream os (png);
    return os.openedOk() && juce::PNGImageFormat().writeImageToStream (img, os);
}

juce::AudioProcessorEditor* Processor::createEditor() { return new Editor (*this); }
} // namespace ns
