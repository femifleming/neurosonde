// Neurosonde — the look: after Ryoji Ikeda (data.matrix, test pattern, datamatics). Black and white only:
//   a black ground, white hairlines and pixels, tiny lowercase type in greys, numbers in a monospace (IBM Plex Mono),
//   ruler ticks, binary cells and barcodes as the only ornament. No colour, no glow, no gradients: things are told
//   apart by brightness, dashes and solid vs hollow marks.
#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include "BinaryData.h"

namespace ns::ui
{
using juce::Colour;
namespace C
{
    const Colour bg       { 0xff000000 };
    const Colour header   { 0xff000000 };
    const Colour body     { 0xff000000 };
    const Colour raised   { 0xff0a0a0a };
    const Colour hover    { 0xff111111 };
    const Colour selected { 0xff1a1a1a };
    const Colour edge     { 0xff262626 };
    const Colour accent   { 0xffffffff };   // the network / predictions: full white
    const Colour topLine  { 0xffcccccc };
    const Colour bar      { 0xffdcdcdc };
    const Colour probe    { 0xffffffff };
    const Colour teal     { 0xff8c8c8c };
    const Colour text     { 0xffeeeeee };
    const Colour text2    { 0xff999999 };
    const Colour muted    { 0xff595959 };
    const Colour amber    { 0xff808080 };   // the input / negatives: mid grey (drawn hollow or dashed where it matters)
    const Colour yellow   { 0xffeeeeee };
    const Colour good     { 0xffbdbdbd };
    const Colour danger   { 0xffffffff };
    const Colour star     { 0xffffffff };
    const Colour starDim  { 0xff333333 };
    const Colour red      { 0xff8a8a8a };   // was the scanner red: now a grey, paired with dashes / hollow marks
}

struct Fonts
{
    juce::Typeface::Ptr mono, monoMed, sans, sansMed;
    Fonts()
    {
        sans = juce::Typeface::createSystemTypefaceFor (BinaryData::InterRegular_ttf, BinaryData::InterRegular_ttfSize);
        sansMed = juce::Typeface::createSystemTypefaceFor (BinaryData::InterMedium_ttf, BinaryData::InterMedium_ttfSize);
        mono = juce::Typeface::createSystemTypefaceFor (BinaryData::IBMPlexMonoLight_ttf, BinaryData::IBMPlexMonoLight_ttfSize);
        monoMed = juce::Typeface::createSystemTypefaceFor (BinaryData::IBMPlexMonoMedium_ttf, BinaryData::IBMPlexMonoMedium_ttfSize);
    }
    static Fonts& get() { static Fonts f; return f; }
};
/** Inter: a clean neo-grotesque for words (weight 1/2 = medium) */
inline juce::Font font (float h, int weight = 0)
{
    return juce::Font (juce::FontOptions (weight >= 1 ? Fonts::get().sansMed : Fonts::get().sans).withHeight (h * 0.86f));
}
inline juce::Font mono (float h) { return juce::Font (juce::FontOptions (Fonts::get().mono).withHeight (h * 0.92f)); }
inline juce::Font monoMed (float h) { return juce::Font (juce::FontOptions (Fonts::get().monoMed).withHeight (h * 0.92f)); }
inline float textWidth (const juce::Font& f, const juce::String& s) { return juce::GlyphArrangement::getStringWidth (f, s); }
/** section labels: small lowercase mono */
inline juce::Font caps (float h) { return mono (h); }

/** buttons and chips: plain rectangles (the cut corner was EVE's; Ikeda is strictly rectilinear) */
inline juce::Path cutRect (juce::Rectangle<float> r, float)
{
    juce::Path p; p.addRectangle (r); return p;
}

/** a ruler: fine ticks, longer every 5 / 10 (Ikeda's gauge strip) */
inline void ruler (juce::Graphics& g, juce::Rectangle<float> r, float step = 4, Colour c = juce::Colours::white.withAlpha (0.22f))
{
    g.setColour (c);
    int k = 0;
    for (float x = r.getX(); x <= r.getRight(); x += step, ++k)
    {
        const float h = k % 10 == 0 ? r.getHeight() : k % 5 == 0 ? r.getHeight() * 0.6f : r.getHeight() * 0.3f;
        g.fillRect (x, r.getBottom() - h, 1.0f, h);
    }
}

/** a window: hairline frame, corner crosses, index + lowercase title, a ruler under the title, no fill */
inline juce::Rectangle<int> drawWindow (juce::Graphics& g, juce::Rectangle<int> r, const juce::String& title, bool active = true, const juce::String& right = {}, int index = -1)
{
    auto rf = r.toFloat();
    g.setColour (C::edge); g.drawRect (rf, 1.0f);
    g.setColour (juce::Colours::white.withAlpha (0.7f));
    for (auto c : { rf.getTopLeft(), rf.getTopRight(), rf.getBottomLeft(), rf.getBottomRight() })
    { g.fillRect (c.x - 4, c.y - 0.5f, 9.0f, 1.0f); g.fillRect (c.x - 0.5f, c.y - 4, 1.0f, 9.0f); }
    auto h = r.removeFromTop (22).reduced (8, 0);
    g.setFont (mono (10)); g.setColour (C::text2);
    const juce::String idx = index >= 0 ? juce::String::toHexString (index).paddedLeft ('0', 2) + "  " : juce::String();
    g.drawText (idx + title.toLowerCase(), h, juce::Justification::centredLeft);
    if (right.isNotEmpty()) { g.setColour (C::muted); g.drawText (right, h, juce::Justification::centredRight); }
    ruler (g, juce::Rectangle<float> ((float) r.getX() + 8, (float) r.getY(), (float) r.getWidth() - 16, 4), 4, juce::Colours::white.withAlpha (active ? 0.16f : 0.08f));
    return r.reduced (8, 6);
}

inline void brackets (juce::Graphics& g, juce::Rectangle<float> r, float len, Colour c, float t = 1.0f)
{
    g.setColour (c);
    const float x0 = r.getX(), y0 = r.getY(), x1 = r.getRight(), y1 = r.getBottom();
    g.drawLine (x0, y0, x0 + len, y0, t); g.drawLine (x0, y0, x0, y0 + len, t);
    g.drawLine (x1, y0, x1 - len, y0, t); g.drawLine (x1, y0, x1, y0 + len, t);
    g.drawLine (x0, y1, x0 + len, y1, t); g.drawLine (x0, y1, x0, y1 - len, t);
    g.drawLine (x1, y1, x1 - len, y1, t); g.drawLine (x1, y1, x1, y1 - len, t);
}

/** signal strength: grey levels, white when strong */
inline Colour signalColour (float s)
{
    return juce::Colours::white.withAlpha (0.35f + 0.65f * juce::jlimit (0.0f, 1.0f, s));
}

/** the probe: a hairline circle with degree ticks and a crosshair (a measuring reticle, not a sphere) */
inline void probeSphere (juce::Graphics& g, juce::Point<float> c, float r, float spin, Colour col)
{
    g.setColour (col.withAlpha (0.75f)); g.drawEllipse (c.x - r, c.y - r, 2 * r, 2 * r, 1.0f);
    g.setColour (col.withAlpha (0.25f)); g.drawEllipse (c.x - r * 0.5f, c.y - r * 0.5f, r, r, 1.0f);
    for (int k = 0; k < 72; ++k)
    {
        const float a = spin * 0.1f + k * juce::MathConstants<float>::twoPi / 72, len = k % 18 == 0 ? 9.0f : k % 6 == 0 ? 5.0f : 2.5f;
        const juce::Point<float> d (std::cos (a), std::sin (a));
        g.setColour (col.withAlpha (k % 6 == 0 ? 0.8f : 0.35f)); g.drawLine ({ c + d * r, c + d * (r + len) }, 1.0f);
    }
    g.setColour (col.withAlpha (0.3f));
    g.fillRect (c.x - r * 1.35f, c.y - 0.5f, r * 2.7f, 1.0f); g.fillRect (c.x - 0.5f, c.y - r * 1.35f, 1.0f, r * 2.7f);
}

/** the probe handle: a small solid square */
inline void probeHandle (juce::Graphics& g, juce::Point<float> c, float s, Colour col)
{
    g.setColour (col); g.fillRect (juce::Rectangle<float> (s * 0.7f, s * 0.7f).withCentre (c));
    g.setColour (col.withAlpha (0.6f)); g.drawRect (juce::Rectangle<float> (s * 1.6f, s * 1.6f).withCentre (c), 1.0f);
}
} // namespace ns::ui
