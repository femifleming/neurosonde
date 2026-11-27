#pragma once
#include "Plugin.h"
#include "ui/MapView.h"
#include "ui/SpaceViews.h"
#include "ui/LabView.h"

namespace ns
{
class Editor : public juce::AudioProcessorEditor, private juce::Timer, public juce::FileDragAndDropTarget
{
public:
    explicit Editor (Processor&);
    ~Editor() override;
    void paint (juce::Graphics&) override;
    void paintOverChildren (juce::Graphics&) override;
    void paintPanels (juce::Graphics&);
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;
    bool isInterestedInFileDrag (const juce::StringArray&) override { return true; }
    void fileDragEnter (const juce::StringArray&, int, int) override { dragOver = true; repaint(); }
    void fileDragExit (const juce::StringArray&) override { dragOver = false; repaint(); }
    void filesDropped (const juce::StringArray& files, int x, int y) override;
    void setPage (int p);
    void notesChangedExternally() { if (notesEd) notesEd->setText (proc.notes, false); }
    /** render the window into a PNG (tests / docs) */
    static bool snapshot (Processor&, int page, const juce::File& png);

private:
    void timerCallback() override;
    juce::RangedAudioParameter& P (const juce::String& id) { return *proc.apvts.getParameter (id); }
    template <typename T, typename... A> T* make (std::vector<juce::Component*>& page, A&&... a)
    {
        auto c = std::make_unique<T> (std::forward<A> (a)...); auto* raw = c.get();
        addChildComponent (*raw); page.push_back (raw); owned.push_back (std::move (c)); return raw;
    }

    Processor& proc;
    int page = 0;
    static constexpr int kPages = 8;
    static constexpr int kLab = 6;   // the lab page (the training, measured)
    static constexpr int kGain = 7;  // gain staging: meters at every stage, the limiter
    /** the tabs, in order: gain after synthesis; the space page only in Neurosonde Space */
    static std::vector<int> tabOrder() { std::vector<int> t { 0, 1, 2, kGain, 3, 4 }; if (NS_SPATIAL) t.push_back (5); t.push_back (kLab); return t; }
    std::vector<juce::Component*> pages[kPages];
    std::vector<std::unique_ptr<juce::Component>> owned;
    juce::Rectangle<int> tabArea[kPages];
    juce::Rectangle<int> panels[kPages][8];
    juce::String panelTitle[kPages][8];
    // constellation page
    std::unique_ptr<ui::MapView> space; ui::MapView* map = nullptr;
    struct LayerRow; std::vector<LayerRow*> layerRows;
    struct ScanList; ScanList* scan = nullptr;
    juce::Component* latent = nullptr;
    // network page
    struct NetView; NetView* net = nullptr;
    struct MemoryList; MemoryList* memories = nullptr;
    // notes page
    juce::TextEditor* notesEd = nullptr;
    struct AxesView; AxesView* axes = nullptr;
    struct JournalList; JournalList* journalList = nullptr;
    // corpus page
    struct JobList; JobList* jobList = nullptr; juce::Component* offlineView = nullptr;
    // gain page
    struct GainView; GainView* gainView = nullptr;
    float outHold = -120; int outHoldAge = 0;
    // space page (Neurosonde Space)
    ui::SphereView* sphere = nullptr; ui::Goniometer* gonio = nullptr; ui::SpatialFeatures* spatFeat = nullptr;
    bool dragOver = false, built = false;
    std::unique_ptr<juce::Component> captureButton, presetButton;
    int tick = 0;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Editor)
};
} // namespace ns
