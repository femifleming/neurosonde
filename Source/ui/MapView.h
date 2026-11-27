// Neurosonde — the main view: what the network has learned, drawn as it is. The self-organising map (Kohonen) that
// tiles the VAE's 16-dimensional latent space with 16 × 16 nodes, each at its true place on the map grid:
//   · cell shade = the U-matrix: how far a node's codebook vector is from its neighbours' in latent space (light
//     walls between cells are boundaries between kinds of sound the network tells apart)
//   · dot size / brightness = how often the sound has landed on that node (decaying hit counts)
//   · words = the patch-note words most often active when that node was hit
//   · the output latent z (solid square, with its recent path) and the input's latent z_in (hollow square), each at
//     its soft best-matching position; the probe (circle) pulls z towards its spot with the gravity you set
// A few measured numbers underneath. Black and white, the background a faint measuring grid.
#pragma once
#include "Widgets.h"
#include "../Plugin.h"
#include <deque>
#include <set>

namespace ns::ui
{
class MapView : public juce::Component, private juce::Timer
{
public:
    explicit MapView (Processor& p) : proc (p) { startTimerHz (30); }
    /** build the map now (and settle the path) instead of waiting for the timer */
    void refreshNow() { rebuild(); for (int i = 0; i < 40; ++i) timerCallback(); }
    std::function<void (juce::Graphics&)> overlay;     // panels painted over the space, under the controls
    float dim = 0;                                      // darkens the space behind full pages
    bool showWorld = true;                              // the map (constellation page)

    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        g.fillAll (juce::Colours::black);
        if (grid.getWidth() != getWidth() || grid.getHeight() != getHeight()) buildGrid();
        g.drawImageAt (grid, 0, 0);
        if (showWorld && ! nodes.empty()) drawMap (g);
        if (dim > 0) { g.setColour (juce::Colours::black.withAlpha (dim)); g.fillRect (r); }
        if (overlay) overlay (g);
    }

    /** the header (tabs, presets, rec) and the footer belong to the editor underneath: clicks there pass through */
    bool hitTest (int, int y) override { return y >= 30 && y < getHeight() - 22; }
    void mouseMove (const juce::MouseEvent& e) override { hover = showWorld ? nearestNode (e.position, 10) : -1; }
    void mouseExit (const juce::MouseEvent&) override { hover = -1; }
    void mouseDown (const juce::MouseEvent& e) override
    {
        if (! showWorld) return;
        if (e.mods.isCommandDown()) { proc.setRoute ({}); return; }
        if (e.mods.isShiftDown()) { const int s = nearestNode (e.position, 16); if (s >= 0) { auto rt = proc.route; rt.push_back (s); proc.setRoute (rt); } return; }
        if (! inner().expanded (12).contains (e.position)) return;
        dragging = true; altStart = proc.apvts.getRawParameterValue ("gravity")->load();
        for (auto* id : { "probex", "probey", "gravity" }) proc.apvts.getParameter (id)->beginChangeGesture();
        if (! e.mods.isAltDown()) moveProbe (e.position);
    }
    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (! dragging) return;
        if (e.mods.isAltDown()) proc.apvts.getParameter ("gravity")->setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, altStart - e.getDistanceFromDragStartY() / 200.0f));
        else moveProbe (e.position);
    }
    void mouseUp (const juce::MouseEvent&) override
    {
        if (! dragging) return;
        dragging = false;
        for (auto* id : { "probex", "probey", "gravity" }) proc.apvts.getParameter (id)->endChangeGesture();
    }
    /** the space between the overlay columns */
    juce::Rectangle<float> area() const { return getLocalBounds().toFloat().withTrimmedLeft (290).withTrimmedRight (290).withTrimmedTop (80).withTrimmedBottom (160); }
    juce::String currentStarName() const { return here >= 0 && here < (int) nodes.size() ? nodes[(size_t) here].name : juce::String(); }
    int currentStar() const { return here; }

private:
    struct Node { float visits = 0, rawHits = 0, centroid = 1000, umat = 0; juce::String name; juce::Point<float> at; };

    /** the map square, and the square through the node centres (the SOM grid: node c at c / 15) */
    juce::Rectangle<float> mapRect() const
    {
        const auto A = area();
        const float side = juce::jmin (A.getWidth() - 40, A.getHeight() - 96);
        return juce::Rectangle<float> (side, side).withCentre ({ A.getCentreX(), A.getY() + 26 + side * 0.5f });
    }
    juce::Rectangle<float> inner() const { const auto m = mapRect(); return m.reduced (m.getWidth() / (2.0f * kSom)); }
    juce::Point<float> toScreen (juce::Point<float> u) const { const auto i = inner(); return { i.getX() + u.x * i.getWidth(), i.getY() + u.y * i.getHeight() }; }
    juce::Point<float> pos (int n) const { return toScreen (nodes[(size_t) n].at); }

    /** the faint measuring grid of the background */
    void buildGrid()
    {
        grid = juce::Image (juce::Image::ARGB, juce::jmax (1, getWidth()), juce::jmax (1, getHeight()), true);
        juce::Graphics g (grid);
        for (int y = 0; y < getHeight(); y += 20)
            for (int x = 0; x < getWidth(); x += 20)
            {
                if (x % 100 == 0 && y % 100 == 0) { g.setColour (juce::Colours::white.withAlpha (0.14f)); g.fillRect (x - 3, y, 7, 1); g.fillRect (x, y - 3, 1, 7); }
                else { g.setColour (juce::Colours::white.withAlpha (0.05f)); g.fillRect (x, y, 1, 1); }
            }
    }

    void drawMap (juce::Graphics& g)
    {
        const auto M = mapRect();
        const float cell = M.getWidth() / kSom;
        const auto t = proc.instance().telemetry();
        // caption
        g.setFont (mono (9.5f)); g.setColour (C::text2);
        g.drawText (juce::String::fromUTF8 ("self-organising map  ·  16 x 16 nodes over the 16-d latent space"), M.withY (M.getY() - 20).withHeight (12), juce::Justification::centred);
        // the U-matrix: one cell per node, lighter where its neighbours are far away in latent space
        g.setColour (juce::Colours::black); g.fillRect (M);
        for (int n = 0; n < kSom * kSom; ++n)
        {
            const int r = n / kSom, c = n % kSom;
            g.setColour (juce::Colours::white.withAlpha (0.02f + 0.30f * nodes[(size_t) n].umat));
            g.fillRect (M.getX() + c * cell, M.getY() + r * cell, cell - 1, cell - 1);
        }
        g.setColour (juce::Colours::white.withAlpha (0.35f)); g.drawRect (M, 1.0f);
        // the nodes: size and brightness by visits
        for (int n = 0; n < kSom * kSom; ++n)
        {
            const auto& s = nodes[(size_t) n]; const auto p = pos (n);
            const float d = 2.0f + 6.0f * s.visits;
            g.setColour (juce::Colours::white.withAlpha (0.25f + 0.75f * s.visits)); g.fillEllipse (juce::Rectangle<float> (d, d).withCentre (p));
        }
        // the route (autopilot), numbered
        if (! proc.route.empty())
        {
            const size_t segs = proc.route.size() > 2 ? proc.route.size() : proc.route.size() - 1;
            for (size_t i = 0; i < segs; ++i)
            {
                juce::Path p; p.startNewSubPath (pos (proc.route[i])); p.lineTo (pos (proc.route[(i + 1) % proc.route.size()]));
                const float dash[] = { 4, 4 }; juce::Path q; juce::PathStrokeType (1.0f).createDashedStroke (q, p, dash, 2);
                g.setColour (juce::Colours::white.withAlpha (0.7f)); g.fillPath (q);
            }
            for (size_t i = 0; i < proc.route.size(); ++i)
            { const auto a = pos (proc.route[i]); g.setColour (C::text); g.setFont (mono (10)); g.drawText (juce::String ((int) i + 1), juce::Rectangle<float> (20, 12).withCentre (a + juce::Point<float> (10, -10)), juce::Justification::centred); }
        }
        // words from the patch notes on the most visited nodes (no two labels on top of each other)
        std::vector<juce::Rectangle<float>> placed;
        g.setFont (mono (10));
        for (auto& lb : labels)
        {
            const auto p = pos (lb.node);
            const auto box = juce::Rectangle<float> (textWidth (mono (10), lb.word) + 4, 12).withPosition (p.x + 6, p.y - 6);
            bool clash = false; for (auto& q : placed) if (q.intersects (box)) clash = true;
            if (clash || ! M.contains (box.getTopLeft())) continue;
            placed.push_back (box);
            g.setColour (juce::Colours::black.withAlpha (0.7f)); g.fillRect (box);
            g.setColour (C::text); g.drawText (lb.word, box, juce::Justification::centredLeft);
        }
        // the probe: where gravity pulls the latent
        {
            const float px = proc.apvts.getRawParameterValue ("probex")->load(), py = proc.apvts.getRawParameterValue ("probey")->load();
            const float grav = proc.apvts.getRawParameterValue ("gravity")->load();
            const auto pp = toScreen ({ px, py });
            const float rr = 10 + 14 * grav;
            g.setColour (juce::Colours::white.withAlpha (0.35f + 0.6f * grav)); g.drawEllipse (pp.x - rr, pp.y - rr, 2 * rr, 2 * rr, 1.0f);
            g.fillRect (pp.x - rr - 5, pp.y - 0.5f, 5.0f, 1.0f); g.fillRect (pp.x + rr, pp.y - 0.5f, 5.0f, 1.0f);
            g.fillRect (pp.x - 0.5f, pp.y - rr - 5, 1.0f, 5.0f); g.fillRect (pp.x - 0.5f, pp.y + rr, 1.0f, 5.0f);
            g.setFont (mono (9.5f)); g.setColour (C::text2);
            g.drawText ("probe  gravity " + juce::String ((int) std::round (grav * 100)) + "%", juce::Rectangle<float> (140, 12).withPosition (pp.x + rr + 8, pp.y + 4), juce::Justification::centredLeft);
        }
        // the output's recent path through the map, fading
        for (size_t i = 1; i < trail.size(); ++i)
        {
            const float age = (float) i / trail.size();
            g.setColour (juce::Colours::white.withAlpha (0.6f * age)); g.drawLine ({ toScreen (trail[i - 1]), toScreen (trail[i]) }, 1.0f);
        }
        // the input (hollow) and the output (solid), joined: how far the network moved the sound
        if (! trail.empty())
        {
            const auto out = toScreen (trail.back());
            if (inputPresent)
            {
                const auto in = toScreen (inputPos);
                juce::Path l; l.startNewSubPath (in); l.lineTo (out);
                const float d[] = { 2, 3 }; juce::Path q; juce::PathStrokeType (1.0f).createDashedStroke (q, l, d, 2);
                g.setColour (juce::Colours::white.withAlpha (0.5f)); g.fillPath (q);
                g.setColour (juce::Colours::black); g.fillRect (juce::Rectangle<float> (11, 11).withCentre (in));
                g.setColour (juce::Colours::white); g.drawRect (juce::Rectangle<float> (11, 11).withCentre (in), 1.2f);
                g.setFont (mono (9.5f)); g.setColour (C::text2); g.drawText ("input", juce::Rectangle<float> (60, 12).withPosition (in.x + 9, in.y + 4), juce::Justification::centredLeft);
            }
            g.setColour (juce::Colours::white); g.fillRect (juce::Rectangle<float> (10, 10).withCentre (out));
            g.setColour (juce::Colours::black); g.drawRect (juce::Rectangle<float> (12, 12).withCentre (out), 1.0f);
            g.setFont (mono (9.5f)); g.setColour (C::text); g.drawText ("output", juce::Rectangle<float> (60, 12).withPosition (out.x + 9, out.y - 16), juce::Justification::centredLeft);
        }
        // hover: what that node is
        if (hover >= 0)
        {
            const auto p = pos (hover); const auto& s = nodes[(size_t) hover];
            g.setColour (juce::Colours::white); g.drawRect (juce::Rectangle<float> (cell, cell).withCentre (p), 1.0f);
            const juce::String info = "node " + juce::String::toHexString (hover).paddedLeft ('0', 3) + "  (row " + juce::String (hover / kSom) + ", col " + juce::String (hover % kSom) + ")   visits " + juce::String ((int) s.rawHits)
                                    + "   decodes to centroid " + juce::String ((int) s.centroid) + " hz" + (s.name.isNotEmpty() ? "   word: " + s.name : juce::String());
            const auto box = juce::Rectangle<float> (textWidth (mono (10), info) + 12, 16).withCentre ({ M.getCentreX(), M.getBottom() - 12 });
            g.setColour (juce::Colours::black.withAlpha (0.85f)); g.fillRect (box);
            g.setColour (juce::Colours::white.withAlpha (0.4f)); g.drawRect (box, 1.0f);
            g.setFont (mono (10)); g.setColour (C::text); g.drawText (info, box, juce::Justification::centred);
        }
        // legend and the measured numbers
        g.setFont (mono (9)); g.setColour (C::muted);
        g.drawText (juce::String::fromUTF8 ("shade: U-matrix (distance to neighbours)  ·  dot: visits  ·  drag: probe  ·  alt-drag: gravity  ·  shift-click: route"),
                    M.withY (M.getBottom() + 5).withHeight (11), juce::Justification::centred);
        auto rows = juce::Rectangle<float> (M.getX(), M.getBottom() + 22, M.getWidth(), 30);
        auto left = rows.removeFromLeft (rows.getWidth() * 0.5f), right = rows;
        auto kv = [&] (juce::Rectangle<float> r, const juce::String& k, const juce::String& v, bool alignLeft)
        {
            g.setFont (mono (10));
            const float kw = textWidth (mono (10), k) + 8, vw = textWidth (mono (10), v);
            const float x0 = alignLeft ? r.getX() : r.getRight() - kw - vw;
            g.setColour (C::muted); g.drawText (k, juce::Rectangle<float> (kw, r.getHeight()).withPosition (x0, r.getY()), juce::Justification::centredLeft);
            g.setColour (C::text); g.drawText (v, juce::Rectangle<float> (vw + 2, r.getHeight()).withPosition (x0 + kw, r.getY()), juce::Justification::centredLeft);
        };
        auto nodeName = [] (int n) { return n >= 0 ? juce::String::toHexString (n).paddedLeft ('0', 3) : juce::String ("-"); };
        kv (left.removeFromTop (14), "output node", nodeName (here) + "   quantisation error " + juce::String (qErr, 3), true);
        kv (left, "surprise", juce::String ((int) std::round (t.surprise * 100)) + "%  (echo-state prediction error)", true);
        kv (right.removeFromTop (14), "input node", nodeName (inputNode) + "   presence " + juce::String ((int) std::round (t.presence * 100)) + "%", false);
        kv (right, "latent distance in -> out", juce::String (moved, 3), false);
    }

    int nearestNode (juce::Point<float> p, float maxD) const
    {
        int best = -1; float bd = maxD * maxD;
        for (size_t n = 0; n < nodes.size(); ++n) { const float d = pos ((int) n).getDistanceSquaredFrom (p); if (d < bd) { bd = d; best = (int) n; } }
        return best;
    }
    void moveProbe (juce::Point<float> p)
    {
        const auto i = inner();
        proc.apvts.getParameter ("probex")->setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, (p.x - i.getX()) / i.getWidth()));
        proc.apvts.getParameter ("probey")->setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, (p.y - i.getY()) / i.getHeight()));
    }
    /** soft best-matching position of a latent on the map (the 4 nearest nodes, weighted), the BMU and its distance */
    juce::Point<float> locate (const float* z, int* bmu = nullptr, float* err = nullptr) const
    {
        float best[4] = { 1e30f, 1e30f, 1e30f, 1e30f }; int bi[4] = { 0, 0, 0, 0 };
        for (int n = 0; n < kSom * kSom; ++n)
        {
            float d = 0; for (int j = 0; j < kZ; ++j) { const float e = somW[(size_t) n * kZ + j] - z[j]; d += e * e; }
            for (int k = 0; k < 4; ++k) if (d < best[k]) { for (int m = 3; m > k; --m) { best[m] = best[m - 1]; bi[m] = bi[m - 1]; } best[k] = d; bi[k] = n; break; }
        }
        if (bmu) *bmu = bi[0];
        if (err) *err = std::sqrt (best[0]);
        juce::Point<float> p; float ws = 0;
        for (int k = 0; k < 4; ++k) { const float wt = std::exp (-(best[k] - best[0]) * 2.0f); p += nodes[(size_t) bi[k]].at * wt; ws += wt; }
        return p / ws;
    }
    void timerCallback() override
    {
        if (++tick % 12 == 0 || nodes.empty()) rebuild();
        const auto t = proc.instance().telemetry();
        if (! nodes.empty() && ! somW.empty())
        {
            const auto np = locate (t.z, &here, &qErr);
            trail.push_back (np); if (trail.size() > 90) trail.pop_front();
            inputPos = locate (t.zIn, &inputNode); inputPresent = t.presence > 0.3f;
            double d = 0; for (int j = 0; j < kZ; ++j) d += (double) (t.z[j] - t.zIn[j]) * (t.z[j] - t.zIn[j]);
            moved = (float) std::sqrt (d);
        }
        repaint();
    }
    void rebuild()
    {
        auto& brain = proc.brain();
        const Snapshot& S = brain.acquireUi();
        somW = S.som.w; const auto hits = S.som.hits;
        std::vector<float> centroid ((size_t) kSom * kSom);
        const bool trained = S.steps > 200; std::array<float, kCond> zc {};
        for (int n = 0; n < kSom * kSom; ++n)
        {
            float out[kOut], feat[kFeat];
            S.vae.decode (somW.data() + (size_t) n * kZ, zc.data(), out); S.feat.undo (out, feat);
            centroid[(size_t) n] = trained ? 1000.0f * std::pow (2.0f, feat[kCentroid]) : 0.0f;
        }
        brain.releaseUi();
        float hmax = 1e-6f; for (float h : hits) hmax = std::max (hmax, h);
        nodes.resize ((size_t) kSom * kSom);
        auto dist = [&] (int a, int b) { float d = 0; for (int j = 0; j < kZ; ++j) { const float e = somW[(size_t) a * kZ + j] - somW[(size_t) b * kZ + j]; d += e * e; } return std::sqrt (d); };
        float umax = 1e-6f;
        for (int n = 0; n < kSom * kSom; ++n)
        {
            auto& s = nodes[(size_t) n]; const int row = n / kSom, col = n % kSom;
            s.at = { col / (kSom - 1.0f), row / (kSom - 1.0f) };   // exactly where SOM::at() puts it
            s.rawHits = hits[(size_t) n]; s.visits = std::sqrt (hits[(size_t) n] / hmax); s.centroid = centroid[(size_t) n];
            float u = 0; int k = 0;
            if (col > 0) { u += dist (n, n - 1); ++k; }
            if (col < kSom - 1) { u += dist (n, n + 1); ++k; }
            if (row > 0) { u += dist (n, n - kSom); ++k; }
            if (row < kSom - 1) { u += dist (n, n + kSom); ++k; }
            s.umat = u / (float) k; umax = std::max (umax, s.umat);
            auto w = brain.starWords (n, 1); s.name = w.empty() ? juce::String() : juce::String (w[0].first);
        }
        for (auto& s : nodes) s.umat /= umax;
        labels.clear();
        std::vector<int> order ((size_t) kSom * kSom); std::iota (order.begin(), order.end(), 0);
        std::sort (order.begin(), order.end(), [&] (int a, int b) { return nodes[(size_t) a].rawHits > nodes[(size_t) b].rawHits; });
        std::set<juce::String> used;
        for (int n : order)
        {
            if (nodes[(size_t) n].name.isEmpty() || (int) labels.size() >= 8 || used.count (nodes[(size_t) n].name)) continue;
            used.insert (nodes[(size_t) n].name);
            labels.push_back ({ n, nodes[(size_t) n].name });
        }
    }

    struct Label { int node; juce::String word; };
    Processor& proc;
    std::vector<Node> nodes; std::vector<Label> labels;
    std::vector<float> somW; std::deque<juce::Point<float>> trail; juce::Point<float> inputPos; bool inputPresent = false;
    int hover = -1, tick = 0, here = -1, inputNode = -1; bool dragging = false; float altStart = 0, qErr = 0, moved = 0;
    juce::Image grid;
};
} // namespace ns::ui
