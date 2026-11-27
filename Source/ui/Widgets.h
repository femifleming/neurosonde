// Neurosonde — controls in the Photon idiom, bound straight to parameters (with host gestures).
#pragma once
#include "Theme.h"
#include <juce_audio_processors/juce_audio_processors.h>

namespace ns::ui
{
/** label · thin ticked track · flat fill · cyan thumb · value. Drag, double-click resets, wheel nudges. */
class ParamBar : public juce::Component, private juce::AudioProcessorParameter::Listener, private juce::AsyncUpdater
{
public:
    ParamBar (juce::RangedAudioParameter& p, juce::String label = {}, Colour fill = C::bar) : param (p), name (std::move (label)), col (fill)
    {
        if (name.isEmpty()) name = p.getName (32);
        param.addListener (this);
        setRepaintsOnMouseActivity (true);
    }
    ~ParamBar() override { param.removeListener (this); }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        auto top = r.removeFromTop (r.getHeight() * 0.5f);
        g.setFont (font (11.5f)); g.setColour (isMouseOverOrDragging() ? C::text : C::text2);
        g.drawText (name.toLowerCase(), top.reduced (2, 0), juce::Justification::bottomLeft);
        g.setFont (mono (11)); g.setColour (C::text); g.drawText (param.getCurrentValueAsText(), top.reduced (2, 0), juce::Justification::bottomRight);
        // ruler: ticks up to the value are lit
        auto tr = r.reduced (2, 0).withTrimmedTop (4).withHeight (8);
        const float v = param.getValue(); const int n = juce::jmax (10, (int) (tr.getWidth() / 3));
        for (int k = 0; k <= n; ++k)
        {
            const float x = tr.getX() + tr.getWidth() * k / n, f = (float) k / n;
            const bool lit = f <= v + 1e-4f;
            const float h = k % 10 == 0 ? 8.0f : k % 5 == 0 ? 5.0f : 3.0f;
            g.setColour (lit ? (col == C::bar ? juce::Colours::white : col).withAlpha (0.92f) : juce::Colours::white.withAlpha (0.16f));
            g.fillRect (x, tr.getBottom() - (lit ? juce::jmax (h, 5.0f) : h), 1.0f, lit ? juce::jmax (h, 5.0f) : h);
        }
        const float x = tr.getX() + tr.getWidth() * v;
        g.setColour (juce::Colours::white); g.fillRect (x - 0.5f, tr.getY() - 3, 1.5f, tr.getHeight() + 3);
    }
    void mouseDown (const juce::MouseEvent& e) override { param.beginChangeGesture(); start = param.getValue(); if (! e.mods.isShiftDown()) set (e); }
    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (e.mods.isShiftDown()) param.setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, start + e.getDistanceFromDragStartX() / (getWidth() * 8.0f)));
        else set (e);
    }
    void mouseUp (const juce::MouseEvent&) override { param.endChangeGesture(); }
    void mouseDoubleClick (const juce::MouseEvent&) override { param.beginChangeGesture(); param.setValueNotifyingHost (param.getDefaultValue()); param.endChangeGesture(); }
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails& w) override
    {
        param.beginChangeGesture(); param.setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, param.getValue() + w.deltaY * 0.05f)); param.endChangeGesture();
    }
private:
    void set (const juce::MouseEvent& e) { param.setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, (e.position.x - 4) / (float) juce::jmax (1, getWidth() - 8))); }
    void parameterValueChanged (int, float) override { triggerAsyncUpdate(); }
    void parameterGestureChanged (int, bool) override {}
    void handleAsyncUpdate() override { repaint(); }
    juce::RangedAudioParameter& param; juce::String name; Colour col; float start = 0;
};

/** text tabs / segmented choice with a 2 px underline on the selected item (choice or bool parameter) */
class ChoiceRow : public juce::Component, private juce::AudioProcessorParameter::Listener, private juce::AsyncUpdater
{
public:
    ChoiceRow (juce::RangedAudioParameter& p, juce::String label = {}) : param (p), name (std::move (label)) { param.addListener (this); setRepaintsOnMouseActivity (true); }
    ~ChoiceRow() override { param.removeListener (this); }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        if (name.isNotEmpty()) { g.setFont (font (11.5f)); g.setColour (C::text2); g.drawText (name.toLowerCase(), r.removeFromLeft (labelW).reduced (2, 0), juce::Justification::centredLeft); }
        const int n = count(); const int cur = index();
        const float w = r.getWidth() / n;
        for (int i = 0; i < n; ++i)
        {
            auto cell = r.withWidth (w).withX (r.getX() + i * w).reduced (1, 3);
            const bool on = i == cur;
            if (on) { g.setColour (juce::Colours::white); g.fillRect (cell); }
            else { g.setColour (juce::Colours::white.withAlpha (0.12f)); g.drawRect (cell, 1.0f); }
            g.setFont (mono (10.5f)); g.setColour (on ? juce::Colours::black : C::text2);
            g.drawText (itemText (i).toLowerCase(), cell, juce::Justification::centred);
        }
    }
    void mouseDown (const juce::MouseEvent& e) override
    {
        auto r = getLocalBounds().toFloat(); if (name.isNotEmpty()) r.removeFromLeft (labelW);
        const int n = count(); const int i = juce::jlimit (0, n - 1, (int) ((e.position.x - r.getX()) / (r.getWidth() / n)));
        param.beginChangeGesture(); param.setValueNotifyingHost (n > 1 ? (float) i / (n - 1) : 0); param.endChangeGesture();
    }
    float labelW = 96;
private:
    int count() const { if (auto* c = dynamic_cast<juce::AudioParameterChoice*> (&param)) return c->choices.size(); return 2; }
    int index() const { if (auto* c = dynamic_cast<juce::AudioParameterChoice*> (&param)) return c->getIndex(); return param.getValue() > 0.5f ? 1 : 0; }
    juce::String itemText (int i) const { if (auto* c = dynamic_cast<juce::AudioParameterChoice*> (&param)) return c->choices[i]; return i ? "on" : "off"; }
    void parameterValueChanged (int, float) override { triggerAsyncUpdate(); }
    void parameterGestureChanged (int, bool) override {}
    void handleAsyncUpdate() override { repaint(); }
    juce::RangedAudioParameter& param; juce::String name;
};

/** a cut-corner button (accent border at 70 %, 15 % fill; hover 30 %) */
class CutButton : public juce::Button
{
public:
    CutButton (const juce::String& t, Colour c = C::accent) : juce::Button (t), col (c) {}
    void paintButton (juce::Graphics& g, bool over, bool down) override
    {
        auto r = getLocalBounds().toFloat().reduced (0.5f);
        auto p = cutRect (r, 5);
        const bool on = getToggleState() || down;
        const Colour c = juce::Colours::white;
        if (on) { g.setColour (c); g.fillPath (p); }
        g.setColour (c.withAlpha (over || on ? 1.0f : 0.4f)); g.strokePath (p, juce::PathStrokeType (1));
        if (isToggleable()) { g.setColour (on ? juce::Colours::black : C::muted); g.fillRect (r.getX() + 7, r.getCentreY() - 2, 4.0f, 4.0f); }
        g.setColour (on ? juce::Colours::black : C::text); g.setFont (mono (11));
        g.drawText (getButtonText().toLowerCase(), getLocalBounds().withTrimmedLeft (isToggleable() ? 10 : 0), juce::Justification::centred);
    }
    Colour col;
};

/** a toggle bound to a bool parameter, drawn as a cut-corner button */
class ParamToggle : public CutButton, private juce::AudioProcessorParameter::Listener, private juce::AsyncUpdater
{
public:
    ParamToggle (juce::RangedAudioParameter& p, const juce::String& t, Colour c = C::accent) : CutButton (t, c), param (p)
    {
        param.addListener (this); setToggleable (true); setToggleState (param.getValue() > 0.5f, juce::dontSendNotification);
        onClick = [this] { param.beginChangeGesture(); param.setValueNotifyingHost (param.getValue() > 0.5f ? 0.0f : 1.0f); param.endChangeGesture(); };
    }
    ~ParamToggle() override { param.removeListener (this); }
private:
    void parameterValueChanged (int, float) override { triggerAsyncUpdate(); }
    void parameterGestureChanged (int, bool) override {}
    void handleAsyncUpdate() override { setToggleState (param.getValue() > 0.5f, juce::dontSendNotification); repaint(); }
    juce::RangedAudioParameter& param;
};

/** HUD ring gauge (after the capacitor): segmented cells around a ring, value inside; drag vertically */
class RingGauge : public juce::Component, private juce::AudioProcessorParameter::Listener, private juce::AsyncUpdater
{
public:
    RingGauge (juce::RangedAudioParameter& p, juce::String label, Colour c = C::star) : param (p), name (std::move (label)), col (c) { param.addListener (this); }
    ~RingGauge() override { param.removeListener (this); }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        auto lab = r.removeFromBottom (16);
        const float d = juce::jmin (r.getWidth(), r.getHeight()) - 4; auto c = r.getCentre();
        const float R = d / 2; const int ticks = 72; const float v = param.getValue();
        const float a0 = -juce::MathConstants<float>::pi * 0.75f, a1 = juce::MathConstants<float>::pi * 0.75f;
        for (int i = 0; i <= ticks; ++i)
        {
            const float a = a0 + (a1 - a0) * i / ticks; const bool lit = (float) i / ticks <= v;
            const float r0 = R - (i % 9 == 0 ? 9.0f : lit ? 7.0f : 4.0f);
            g.setColour (lit ? col.withAlpha (0.95f) : juce::Colours::white.withAlpha (0.18f));
            g.drawLine (c.x + std::sin (a) * r0, c.y - std::cos (a) * r0, c.x + std::sin (a) * R, c.y - std::cos (a) * R, 1.0f);
        }
        g.setColour (juce::Colours::white.withAlpha (0.12f)); g.drawEllipse (c.x - R + 14, c.y - R + 14, 2 * R - 28, 2 * R - 28, 1);
        g.setColour (C::text); g.setFont (mono (d > 70 ? 16 : 12));
        g.drawText (param.getCurrentValueAsText(), r, juce::Justification::centred);
        g.setColour (C::muted); g.setFont (mono (10)); g.drawText (name.toLowerCase(), lab, juce::Justification::centred);
    }
    void mouseDown (const juce::MouseEvent&) override { param.beginChangeGesture(); start = param.getValue(); }
    void mouseDrag (const juce::MouseEvent& e) override { param.setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, start - e.getDistanceFromDragStartY() / 150.0f)); }
    void mouseUp (const juce::MouseEvent&) override { param.endChangeGesture(); }
    void mouseDoubleClick (const juce::MouseEvent&) override { param.beginChangeGesture(); param.setValueNotifyingHost (param.getDefaultValue()); param.endChangeGesture(); }
private:
    void parameterValueChanged (int, float) override { triggerAsyncUpdate(); }
    void parameterGestureChanged (int, bool) override {}
    void handleAsyncUpdate() override { repaint(); }
    juce::RangedAudioParameter& param; juce::String name; Colour col; float start = 0;
};
} // namespace ns::ui
