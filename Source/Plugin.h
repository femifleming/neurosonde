// Neurosonde — the plug-in / app processor: parameters, the shared brain, file feeding and processing, the journal.
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>
#ifndef NS_SPATIAL
 #define NS_SPATIAL 0
#endif
#include "Core.h"

namespace ns
{
/** one brain per process, shared by every instance (and the standalone app); other processes share it through
    ~/Library/Application Support/Neurosonde/brain.nsb */
struct BrainHolder
{
    std::shared_ptr<Brain> brain;
    BrainHolder()
    {
        auto dir = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory).getChildFile ("Application Support/Neurosonde");
        if (auto* env = std::getenv ("NEUROSONDE_BRAIN_DIR")) dir = juce::File (juce::String (env));   // tests use their own brain
        dir.createDirectory();
        brain = std::make_shared<Brain> (dir.getFullPathName().toStdString());
    }
};

/** a file being fed (learned) or processed (rendered through the network) */
struct FileJob
{
    juce::File file; bool process = false; bool train = false;   // train: offline training on `files`
    juce::Array<juce::File> files;
    std::atomic<float> progress { 0 }; std::atomic<int> state { 0 };   // 0 queued, 1 running, 2 done, 3 failed
    juce::String result;
};

class Processor : public juce::AudioProcessor, private juce::Thread
{
public:
    Processor();
    ~Processor() override;

    void prepareToPlay (double sr, int block) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout&) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
#if NS_SPATIAL
    const juce::String getName() const override { return "Neurosonde Space"; }
#else
    const juce::String getName() const override { return "Neurosonde"; }
#endif
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 8.0; }
    int getNumPrograms() override;
    int getCurrentProgram() override { return currentProgram; }
    void setCurrentProgram (int) override;
    const juce::String getProgramName (int) override;
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    // ---- for the editor ----
    juce::AudioProcessorValueTreeState apvts;
    Instance& instance() { return *inst; }
    Brain& brain() { return *holder->brain; }
    juce::String notes;                      // the patch notes (also saved in the session)
    void setNotes (const juce::String& t) { notes = t; inst->setNotes (t.toStdString()); }
    void logJournal();                       // append the notes to the shared journal
    juce::StringArray journal() const;       // newest first: "time · steps · text"
    std::vector<int> route;                  // constellation route (star indices)
    void setRoute (const std::vector<int>& r) { route = r; inst->setRoute (r); }

    void addFiles (const juce::StringArray& paths, bool process);
    /** offline training: one job over all the files (folders are searched) */
    void trainOnFiles (const juce::StringArray& paths);
    std::vector<std::shared_ptr<FileJob>> jobs() { const juce::ScopedLock l (jobLock); return jobList; }
    juce::File outputFolder() const { return juce::File::getSpecialLocation (juce::File::userMusicDirectory).getChildFile ("Neurosonde"); }

    // ---- capture: record the output to ~/Music/Neurosonde
    bool capturing() const { return writer.load() != nullptr; }
    void toggleCapture();
    juce::File lastCapture;

    /** layer / engine parameter ids */
    static juce::String lid (int layer, const char* what) { return "l" + juce::String (layer + 1) + what; }

private:
    void run() override;   // file jobs
    void feedFile (FileJob&);
    void processFile (FileJob&);
    void trainOfflineJob (FileJob&);
    void paceOfflineRender (bool offlineLearning);
    /** analyse a file: feed items (with the stereo image when it is stereo) to `sink`, audio + units to the corpus */
    bool analyseFile (FileJob&, const juce::File&, const std::function<void (const FeedItem&)>& sink, float p0, float p1);
    void readParams (Params& p) const;
    static juce::AudioProcessorValueTreeState::ParameterLayout layout();

    juce::SharedResourcePointer<BrainHolder> holder;
    std::unique_ptr<Instance> inst;
    juce::AudioFormatManager formats;
    juce::CriticalSection jobLock;
    std::vector<std::shared_ptr<FileJob>> jobList;
    juce::AudioBuffer<float> silentIn, spareSc, silentZero; int captureCh = 2;
    std::atomic<bool> trainingHere { false };
    int currentProgram = 0;
    juce::TimeSliceThread captureThread { "Neurosonde capture" };
    std::unique_ptr<juce::AudioFormatWriter::ThreadedWriter> threaded;
    std::atomic<juce::AudioFormatWriter::ThreadedWriter*> writer { nullptr };
    juce::CriticalSection captureLock;
    double sampleRate = 48000;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Processor)
};
} // namespace ns
