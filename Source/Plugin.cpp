#include "Plugin.h"
#include "Editor.h"

namespace ns
{
using APVTS = juce::AudioProcessorValueTreeState;

static juce::StringArray modeNames() { juce::StringArray s; for (int m = 0; m < (int) Mode::Count; ++m) s.add (modeName (m)); return s; }

APVTS::ParameterLayout Processor::layout()
{
    using namespace juce;
    APVTS::ParameterLayout l;
    auto f = [&] (const String& id, const String& name, float lo, float hi, float def, float skew = 1.0f, const String& unit = {})
    {
        NormalisableRange<float> r (lo, hi); r.skew = skew;
        const int dec = hi - lo >= 20 ? 1 : 2;
        auto fmt = [dec, unit, lo, hi] (float v, int)
        {
            if (unit.isEmpty() && lo >= 0 && hi <= 2.0f) return String (v * 100.0f, 0) + "%";
            return String (v, dec) + (unit.isNotEmpty() ? " " + unit : String());
        };
        l.add (std::make_unique<AudioParameterFloat> (ParameterID { id, 1 }, name, r, def,
            AudioParameterFloatAttributes().withLabel (unit).withStringFromValueFunction (fmt)));
    };
    auto c = [&] (const String& id, const String& name, StringArray items, int def) { l.add (std::make_unique<AudioParameterChoice> (ParameterID { id, 1 }, name, items, def)); };
    auto b = [&] (const String& id, const String& name, bool def) { l.add (std::make_unique<AudioParameterBool> (ParameterID { id, 1 }, name, def)); };
    const int defModes[4] = { 0, 3, 6, 1 };
    for (int k = 0; k < Synth::kLayers; ++k)
    {
        b (lid (k, "on"), "layer " + String (k + 1) + " on", k == 0);
        c (lid (k, "mode"), "layer " + String (k + 1) + " engine", modeNames(), defModes[k]);
        f (lid (k, "level"), "layer " + String (k + 1) + " level", 0, 1.5f, 1.0f);
        f (lid (k, "pan"), "layer " + String (k + 1) + " pan", -1, 1, 0);
        f (lid (k, "char"), "layer " + String (k + 1) + " character", 0, 1, 0.3f);
    }
    c ("combine", "combine", { "stack", "morph", "merge", "imprint" }, 1);
    f ("morph", "morph", 0, 1, 0);
    c ("morphsrc", "morph source", { "hand", "network", "stars", "lfo" }, 0);
    f ("morphrate", "morph rate", 0.005f, 2, 0.05f, 0.3f, "hz");
    f ("lucid", "lucidity", 0, 1, 0.7f);
    f ("temp", "temperature", 0, 2, 0.5f);
    f ("drift", "drift", 0, 1, 0.1f);
    c ("dream", "dream engine", { "gru", "reservoir", "both" }, 0);
    f ("gravity", "probe gravity", 0, 1, 0);
    f ("probex", "probe x", 0, 1, 0.5f);
    f ("probey", "probe y", 0, 1, 0.5f);
    b ("route", "autopilot", false);
    f ("routespeed", "autopilot speed", 0.5f, 60, 4, 0.4f, "stars/min");
    c ("learnmode", "learning", { "online", "offline" }, 0);   // online: learns while playing; offline: frozen, trained on files
    f ("epochs", "offline epochs", 2, 200, 40, 0.5f);
    b ("listen", "listen", true);
    f ("plastic", "plasticity", 0, 2, 1);
    f ("sleep", "sleep", 0, 1, 0.25f);
    f ("betamax", "kl weight", 0.01f, 1, 0.2f, 0.4f);
    f ("gate", "gate", -90, -20, -58, 1, "db");
    f ("ingain", "input gain", -24, 24, 0, 1, "db");
    f ("mix", "dry / wet", 0, 1, 1);
    f ("out", "output", -36, 12, 0, 1, "db");
    // gain staging: the limiter keeps every sample under the ceiling (clean: transparent; soft: saturates into it first)
    f ("ceiling", "ceiling", -12, 0, -1, 1, "db");
    c ("limiter", "limiter", { "clean", "soft" }, 0);   // always on: nothing leaves above the ceiling
    b ("autogain", "auto gain", false);
    b ("match", "match engine levels", true);
    f ("transpose", "transpose", -24, 24, 0, 1, "st");
    c ("pitchsrc", "pitch source", { "input", "network" }, 0);
    f ("grainsize", "grain size", 10, 1000, 120, 0.4f, "ms");
    f ("spray", "spray", 0, 1, 0.3f);
    f ("focus", "recall focus", 0.5f, 40, 6, 0.4f);
    c ("space", "similarity", { "latent", "timbre", "chroma", "loudness" }, 0);
    c ("playback", "mosaic playback", { "point", "ring", "path" }, 2);
    f ("ring", "ring size", 1, 16, 6);
    f ("pitchfollow", "pitch follow", 0, 1, 0.5f);
    f ("imprint", "imprint", 0, 1, 0);
    f ("tilt", "tilt", -1, 1, 0);
    f ("notesinf", "notes influence", 0, 1.5f, 0.6f);
    f ("width", "width", 0, 1.5f, 0.7f);
    f ("spin", "spin", 0, 1, 0);
    f ("nebula", "nebula", 0, 1, 0.25f);
    f ("decay", "nebula decay", 0, 1, 0.6f);
    f ("learnwidth", "learned width", 0, 1, 0.0f);
#if NS_SPATIAL
    c ("sformat", "output", { "ambix", "binaural", "stereo", "quad", "5.1", "7.1.4" }, 1);
    c ("sorder", "ambisonic order", { "1", "2", "3", "4", "5", "6", "7" }, 2);
    f ("follow", "follow the network", 0, 1, 0.7f);
    f ("span", "azimuth span", 0, 1, 0.75f);
    f ("elev", "elevation span", 0, 1, 0.5f);
    f ("orbit", "orbit", -12, 12, 0, 1, "rpm");
    f ("distance", "distance", 0.5f, 12, 2, 0.5f, "m");
    f ("lspread", "layer spread", 0, 1, 0.35f);
    f ("room", "room size", 0.3f, 3, 1, 0.6f);
    f ("early", "early reflections", 0, 1, 0.35f);
    f ("late", "late field", 0, 1, 0.3f);
    f ("rdecay", "room decay", 0, 1, 0.6f);
    f ("yaw", "scene yaw", -180, 180, 0, 1, "deg");
    f ("stilt", "scene tilt", -90, 90, 0, 1, "deg");
#endif
    return l;
}

Processor::Processor()
    : AudioProcessor (BusesProperties().withInput ("Input", juce::AudioChannelSet::stereo(), true)
                                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)   // spatial: binaural by default; up to 64 (7th-order AmbiX)
                                       .withInput ("Sidechain", juce::AudioChannelSet::stereo(), false)),
      juce::Thread ("Neurosonde files"),
      apvts (*this, nullptr, "NEUROSONDE", layout())
{
    formats.registerBasicFormats();
    inst = std::make_unique<Instance> (holder->brain);
    startThread();
}

Processor::~Processor()
{
    if (capturing()) toggleCapture();
    captureThread.stopThread (2000);
    signalThreadShouldExit(); notify();
    if (trainingHere) brain().offline.cancel = true;   // closing during offline training: stop it at the next batch
    stopThread (-1);                                   // and wait for it: the brain must not go away under it
    // chunks still in the instance's queues
    while (auto* c = inst->takeDone()) delete c;
}

bool Processor::isBusesLayoutSupported (const BusesLayout& l) const
{
    const auto out = l.getMainOutputChannelSet();
#if NS_SPATIAL
    // stereo (binaural / stereo), quad, 5.1, 7.1.4, or any AmbiX order 1..7 (4..64 channels), or discrete channels
    const int nOut = out.size();
    if (nOut < 2 || nOut > 64) return false;
#else
    if (out != juce::AudioChannelSet::stereo() && out != juce::AudioChannelSet::mono()) return false;
#endif
    const auto in = l.getMainInputChannelSet();
    if (! (in.isDisabled() || in == juce::AudioChannelSet::stereo() || in == juce::AudioChannelSet::mono())) return false;
    if (l.inputBuses.size() > 1)
    {
        const auto sc = l.getChannelSet (true, 1);
        if (! (sc.isDisabled() || sc == juce::AudioChannelSet::stereo() || sc == juce::AudioChannelSet::mono())) return false;
    }
    return true;
}

void Processor::prepareToPlay (double sr, int block)
{
    sampleRate = sr;
    inst->prepare (sr, std::max (block, 512));
    setLatencySamples (inst->latencySamples());   // the output limiter's look-ahead
    silentIn.setSize (2, std::max (block, 8192)); spareSc.setSize (2, std::max (block, 8192));
    silentZero.setSize (1, std::max (block, 8192)); silentZero.clear();

}

void Processor::readParams (Params& p) const
{
    auto v = [&] (const char* id) { return apvts.getRawParameterValue (id)->load(); };
    auto vs = [&] (const juce::String& id) { return apvts.getRawParameterValue (id)->load(); };
    for (int k = 0; k < Synth::kLayers; ++k)
    {
        auto& L = p.layer[(size_t) k];
        L.on = vs (lid (k, "on")) > 0.5f; L.mode = (int) vs (lid (k, "mode")); L.level = vs (lid (k, "level"));
        L.pan = vs (lid (k, "pan")); L.character = vs (lid (k, "char"));
    }
    p.combine = (int) v ("combine"); p.morph = v ("morph"); p.morphSource = (int) v ("morphsrc"); p.morphRate = v ("morphrate");
    p.lucid = v ("lucid"); p.temperature = v ("temp"); p.drift = v ("drift"); p.dreamEngine = (int) v ("dream");
    p.gravity = v ("gravity"); p.probeX = v ("probex"); p.probeY = v ("probey"); p.route = v ("route") > 0.5f; p.routeSpeed = v ("routespeed");
    p.learn = (int) v ("learnmode") == 0; p.listen = v ("listen") > 0.5f; p.gateDb = v ("gate"); p.inGainDb = v ("ingain");
    p.mix = v ("mix"); p.outDb = v ("out"); p.ceilingDb = v ("ceiling"); p.limiter = (int) v ("limiter"); p.autoGain = v ("autogain") > 0.5f; p.matchLevels = v ("match") > 0.5f; p.transpose = v ("transpose"); p.pitchSource = (int) v ("pitchsrc"); p.keysOnly = false;
    p.grainSize = v ("grainsize"); p.spray = v ("spray"); p.focus = v ("focus"); p.space = (int) v ("space"); p.playback = (int) v ("playback");
    p.ringSize = v ("ring"); p.pitchFollow = v ("pitchfollow"); p.imprint = v ("imprint"); p.tilt = v ("tilt"); p.notesInfluence = v ("notesinf");
    p.width = v ("width"); p.spin = v ("spin"); p.nebula = v ("nebula"); p.decay = v ("decay");
    p.learnedWidth = v ("learnwidth");
#if NS_SPATIAL
    p.spatial = true; p.format = (int) v ("sformat"); p.order = (int) v ("sorder") + 1;
    p.follow = v ("follow"); p.span = v ("span"); p.elevSpan = v ("elev"); p.orbit = v ("orbit"); p.distance = v ("distance");
    p.layerSpread = v ("lspread"); p.room = v ("room"); p.early = v ("early"); p.late = v ("late"); p.roomDecay = v ("rdecay");
    p.yaw = v ("yaw") * (float) M_PI / 180; p.sceneTilt = v ("stilt") * (float) M_PI / 180;
#endif
}

/** A DAW bounce runs faster than real time: the trainer would fall behind and frames would be dropped from the full
    feed ring. When the host renders offline, wait (bounded) until the brain has taken what this block fed it, so a
    bounce learns what playing it live would have learned. */
void Processor::paceOfflineRender (bool offlineLearning)
{
    if (! isNonRealtime() || offlineLearning || apvts.getRawParameterValue ("listen")->load() < 0.5f) return;
    brain().wake();
    for (int k = 0; k < 400 && brain().backlog() > 1.5; ++k) std::this_thread::sleep_for (std::chrono::microseconds (250));
}

void Processor::processBlock (juce::AudioBuffer<float>& buf, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals nd;
    readParams (inst->params);
    auto& s = brain().settings;
    s.plasticity = apvts.getRawParameterValue ("plastic")->load();
    s.sleepRate = apvts.getRawParameterValue ("sleep")->load();
    s.betaMax = apvts.getRawParameterValue ("betamax")->load();
    // offline learning: the shared brain is frozen while playing (it is trained on files instead)
    const bool offline = (int) apvts.getRawParameterValue ("learnmode")->load() == 1;
    s.frozen = offline;
    const int n = buf.getNumSamples();
    auto mainIn = getBusBuffer (buf, true, 0);
    const int nIn = getBus (true, 0) != nullptr && getBus (true, 0)->isEnabled() ? mainIn.getNumChannels() : 0;
    const float* inL = nIn > 0 ? mainIn.getReadPointer (0) : silentIn.getReadPointer (0);
    const float* inR = nIn > 1 ? mainIn.getReadPointer (1) : inL;
    if (nIn == 0) silentIn.clear();
    // sidechain: when enabled, it is what Neurosonde listens to (learns from, follows)
    const float* scL = nullptr; const float* scR = nullptr;
    if (auto* sc = getBus (true, 1); sc != nullptr && sc->isEnabled() && sc->getNumberOfChannels() > 0)
    {
        auto scBuf = getBusBuffer (buf, true, 1);
        scL = scBuf.getReadPointer (0); scR = scBuf.getNumChannels() > 1 ? scBuf.getReadPointer (1) : scL;
    }
    // the instance writes the output in place; it reads each input sample before writing that sample's output
    float* outL = buf.getWritePointer (0);
    float* outR = buf.getNumChannels() > 1 ? buf.getWritePointer (1) : outL;
    // the instance reads each input sample before writing that sample's output; a sidechain may share memory with
    // the output in some hosts, so copy it first
    if (scL) { silentIn.copyFrom (0, 0, scL, n); silentIn.copyFrom (1, 0, scR, n); scL = silentIn.getReadPointer (0); scR = silentIn.getReadPointer (1); }
#if NS_SPATIAL
    {
        // the input is read before the output is written, segment by segment; with more outputs than inputs the
        // extra channels are free to write. Keep a copy of the input anyway (hosts may alias buffers).
        silentIn.copyFrom (0, 0, inL, n); silentIn.copyFrom (1, 0, inR, n);
        inL = silentIn.getReadPointer (0); inR = silentIn.getReadPointer (1);
        if (scL) { spareSc.copyFrom (0, 0, scL, n); spareSc.copyFrom (1, 0, scR, n); scL = spareSc.getReadPointer (0); scR = spareSc.getReadPointer (1); }
        // never trust the bus alone: some hosts (and auval) hand over fewer channels than the bus reports
        const int nOut = std::min (getBus (false, 0)->getNumberOfChannels(), buf.getNumChannels());
        if (nOut < 2) { buf.clear(); midi.clear(); return; }
        float* outs[64]; for (int c = 0; c < std::min (nOut, 64); ++c) outs[c] = buf.getWritePointer (c);
        // AmbiX: the order follows the bus when it is smaller than the chosen one
        if ((int) apvts.getRawParameterValue ("sformat")->load() == 0) inst->params.order = std::min (inst->params.order, std::max (1, (int) std::floor (std::sqrt ((double) nOut)) - 1));
        inst->setSpatialOutput (outs, std::min (nOut, 64));
        inst->process (inL, inR, outs[0], outs[1], n, {}, scL, scR);
        paceOfflineRender (offline);
        inst->setSpatialOutput (nullptr, 0);
        if (auto* w = writer.load())
        {
            const float* ch[64];
            for (int c = 0; c < captureCh; ++c) ch[c] = c < std::min (nOut, 64) ? outs[c] : silentZero.getReadPointer (0);
            if (n <= silentZero.getNumSamples()) w->write (ch, n);
        }
        for (int c = std::min (nOut, 64); c < buf.getNumChannels(); ++c) buf.clear (c, 0, n);
        midi.clear();
        return;
    }
#endif
    inst->process (inL, inR, outL, outR, n, {}, scL, scR);
    paceOfflineRender (offline);
    if (auto* w = writer.load()) { const float* ch[2] = { outL, outR }; w->write (ch, n); }
    for (int c = 2; c < buf.getNumChannels(); ++c) buf.clear (c, 0, n);
    midi.clear();
}

// ---- factory presets ---------------------------------------------------------------------------------------------
namespace
{
struct Preset
{
    const char* name; const char* notes;
    std::vector<std::pair<const char*, float>> values;   // real (unnormalised) values; choice = index
};
const std::vector<Preset>& presets()
{
    static const std::vector<Preset> p = {
        { "first light", "first light. warm, golden, slow drift over a quiet room.",
          { { "l1on", 1 }, { "l1mode", 0 }, { "l2on", 1 }, { "l2mode", 3 }, { "l3on", 0 }, { "l4on", 0 }, { "combine", 1 }, { "morphsrc", 1 },
            { "lucid", 0.75f }, { "temp", 0.4f }, { "drift", 0.12f }, { "nebula", 0.35f } } },
        { "hydrophone", "underwater, a deep room, wet and slow. pressure.",
          { { "l1on", 1 }, { "l1mode", 1 }, { "l2on", 1 }, { "l2mode", 0 }, { "l3on", 0 }, { "l4on", 0 }, { "combine", 2 }, { "morph", 0.5f },
            { "lucid", 0.5f }, { "temp", 0.6f }, { "drift", 0.2f }, { "nebula", 0.6f }, { "decay", 0.85f }, { "tilt", -0.4f } } },
        { "glass folding", "glass, bright, metal ticks folding over each other. sparse.",
          { { "l1on", 1 }, { "l1mode", 2 }, { "l2on", 1 }, { "l2mode", 4 }, { "l3on", 0 }, { "l4on", 0 }, { "combine", 0 },
            { "playback", 1 }, { "lucid", 0.8f }, { "temp", 0.5f }, { "tilt", 0.4f } } },
        { "tungsten room", "tungsten, sepia, dust on the windows. a memory of a warehouse.",
          { { "l1on", 1 }, { "l1mode", 10 }, { "l2on", 1 }, { "l2mode", 3 }, { "l3on", 1 }, { "l3mode", 0 }, { "l4on", 0 }, { "combine", 1 }, { "morphsrc", 3 },
            { "morphrate", 0.03f }, { "lucid", 0.6f }, { "temp", 0.7f }, { "drift", 0.25f } } },
        { "probe sweep", "probe the stars. signal, wide, a gate somewhere.",
          { { "l1on", 1 }, { "l1mode", 4 }, { "l2on", 1 }, { "l2mode", 1 }, { "l3on", 0 }, { "l4on", 0 }, { "combine", 1 }, { "morphsrc", 2 },
            { "gravity", 0.6f }, { "lucid", 0.3f }, { "temp", 0.6f }, { "width", 1.1f } } },
        { "ear in the dark", "night. the pitch is made in the ear. still.",
          { { "l1on", 1 }, { "l1mode", 5 }, { "l2on", 1 }, { "l2mode", 0 }, { "l3on", 0 }, { "l4on", 0 }, { "combine", 0 },
            { "l2level", 0.35f }, { "lucid", 0.7f }, { "temp", 0.3f }, { "nebula", 0.2f } } },
        { "tanpura memory", "strings, sympathetic, a drone held for a long time. warm.",
          { { "l1on", 1 }, { "l1mode", 6 }, { "l2on", 1 }, { "l2mode", 3 }, { "l3on", 1 }, { "l3mode", 7 }, { "l4on", 0 }, { "combine", 1 }, { "morphsrc", 1 },
            { "lucid", 0.65f }, { "temp", 0.5f }, { "l1char", 0.5f } } },
        { "skipping archive", "an old cd, skipping. broken machine, rough, fast.",
          { { "l1on", 1 }, { "l1mode", 10 }, { "l2on", 1 }, { "l2mode", 8 }, { "l3on", 0 }, { "l4on", 0 }, { "combine", 3 },
            { "l1char", 0.8f }, { "lucid", 0.55f }, { "temp", 0.9f }, { "drift", 0.3f } } },
        { "wormhole", "a wormhole: dense, dark, swirling, vast.",
          { { "l1on", 1 }, { "l1mode", 1 }, { "l2on", 1 }, { "l2mode", 3 }, { "l3on", 1 }, { "l3mode", 8 }, { "l4on", 0 }, { "combine", 2 }, { "morphsrc", 3 },
            { "morphrate", 0.12f }, { "imprint", 0.5f }, { "lucid", 0.2f }, { "temp", 1.2f }, { "drift", 0.5f }, { "spin", 0.4f }, { "nebula", 0.55f } } },
        { "uncanny valley", "half learned. the network's own voice, semi-congealed, failing to mimic.",
          { { "l1on", 1 }, { "l1mode", 1 }, { "l2on", 1 }, { "l2mode", 9 }, { "l3on", 0 }, { "l4on", 0 }, { "combine", 1 }, { "morphsrc", 1 },
            { "lucid", 0.45f }, { "temp", 1.5f }, { "drift", 0.35f }, { "plastic", 0.3f }, { "l1char", 0.9f } } },
    };
    return p;
}
}
int Processor::getNumPrograms() { return (int) presets().size(); }
const juce::String Processor::getProgramName (int i) { return juce::isPositiveAndBelow (i, (int) presets().size()) ? juce::String (presets()[(size_t) i].name) : juce::String(); }
void Processor::setCurrentProgram (int i)
{
    if (! juce::isPositiveAndBelow (i, (int) presets().size())) return;
    currentProgram = i;
    // start from the defaults, then the preset's values
    for (auto* prm : getParameters())
        if (auto* r = dynamic_cast<juce::RangedAudioParameter*> (prm)) r->setValueNotifyingHost (r->getDefaultValue());
    for (auto& [id, v] : presets()[(size_t) i].values)
        if (auto* r = apvts.getParameter (id)) r->setValueNotifyingHost (r->convertTo0to1 (v));
    setNotes (presets()[(size_t) i].notes);
    if (auto* ed = dynamic_cast<Editor*> (getActiveEditor())) ed->notesChangedExternally();
}

// ---- capture -------------------------------------------------------------------------------------------------------
void Processor::toggleCapture()
{
    const juce::ScopedLock l (captureLock);
    if (auto* w = writer.exchange (nullptr))
    {
        juce::Thread::sleep (30);   // let the audio thread finish a block that may still hold it
        threaded.reset(); (void) w;
        return;
    }
    auto dir = outputFolder(); dir.createDirectory();
    // Space records every output channel (AmbiX keeps ACN order, SN3D: the file name says the order)
    captureCh = NS_SPATIAL ? juce::jlimit (2, 64, getTotalNumOutputChannels()) : 2;
    juce::String tag;
#if NS_SPATIAL
    {
        const int fmt = (int) apvts.getRawParameterValue ("sformat")->load();
        static const char* names[] = { "ambix", "binaural", "stereo", "quad", "5.1", "7.1.4" };
        tag = juce::String (" ") + names[juce::jlimit (0, 5, fmt)];
        if (fmt == 0) tag << " o" << (int) std::floor (std::sqrt ((double) captureCh) - 1) << " acn sn3d";
    }
#endif
    lastCapture = dir.getNonexistentChildFile ("capture " + juce::Time::getCurrentTime().formatted ("%Y-%m-%d %H.%M.%S") + tag, ".wav");
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::OutputStream> os (lastCapture.createOutputStream());
    if (! os) return;
    auto* w = wav.createWriterFor (os.release(), sampleRate, (unsigned int) captureCh, 24, {}, 0);
    if (! w) return;
    if (! captureThread.isThreadRunning()) captureThread.startThread();
    threaded = std::make_unique<juce::AudioFormatWriter::ThreadedWriter> (w, captureThread, 1 << 17);
    writer.store (threaded.get());
}

// ---- state ------------------------------------------------------------------------------------------------------
void Processor::getStateInformation (juce::MemoryBlock& dest)
{
    auto st = apvts.copyState();
    st.setProperty ("notes", notes, nullptr);
    juce::StringArray r; for (int x : route) r.add (juce::String (x));
    st.setProperty ("route", r.joinIntoString (","), nullptr);
    st.setProperty ("program", currentProgram, nullptr);   // the preset's name comes back with the session
    if (auto xml = st.createXml()) copyXmlToBinary (*xml, dest);
}
void Processor::setStateInformation (const void* d, int size)
{
    if (auto xml = getXmlFromBinary (d, size))
    {
        auto st = juce::ValueTree::fromXml (*xml);
        if (! st.isValid()) return;
        setNotes (st.getProperty ("notes").toString());
        std::vector<int> r; for (auto& t : juce::StringArray::fromTokens (st.getProperty ("route").toString(), ",", "")) if (t.isNotEmpty()) r.push_back (t.getIntValue());
        setRoute (r);
        // only the index: the parameters in the state already hold the sound (applying the preset would undo edits)
        currentProgram = juce::jlimit (0, std::max (0, getNumPrograms() - 1), (int) st.getProperty ("program", 0));
        st.removeProperty ("notes", nullptr); st.removeProperty ("route", nullptr); st.removeProperty ("program", nullptr);
        apvts.replaceState (st);
        updateHostDisplay (ChangeDetails().withProgramChanged (true));
    }
}

// ---- journal ----------------------------------------------------------------------------------------------------
void Processor::logJournal()
{
    if (notes.trim().isEmpty()) return;
    auto f = juce::File (juce::String (brain().folder())).getChildFile ("journal.txt");
    const auto line = juce::Time::getCurrentTime().formatted ("%Y-%m-%d %H:%M") + "\t" + juce::String ((juce::int64) brain().stats.vaeSteps.load())
                    + "\t" + notes.replaceCharacters ("\t\n\r", "   ") + "\n";
    f.appendText (line, false, false, "\n");
    brain().remember (notes.upToFirstOccurrenceOf ("\n", false, false).substring (0, 60).toStdString());
}
juce::StringArray Processor::journal() const
{
    auto f = juce::File (juce::String (holder->brain->folder())).getChildFile ("journal.txt");
    juce::StringArray lines; f.readLines (lines); lines.removeEmptyStrings();
    juce::StringArray out; for (int i = lines.size(); --i >= 0;) out.add (lines[i]);
    return out;
}

// ---- files ------------------------------------------------------------------------------------------------------
void Processor::addFiles (const juce::StringArray& paths, bool process)
{
    {
        const juce::ScopedLock l (jobLock);
        for (auto& p : paths)
        {
            juce::File f (p);
            if (f.isDirectory())
            {
                for (auto& c : f.findChildFiles (juce::File::findFiles, true, formats.getWildcardForAllFormats()))
                { auto j = std::make_shared<FileJob>(); j->file = c; j->process = process; jobList.push_back (j); }
            }
            else if (formats.findFormatForFileExtension (f.getFileExtension()) != nullptr)
            { auto j = std::make_shared<FileJob>(); j->file = f; j->process = process; jobList.push_back (j); }
        }
    }
    notify();
}

void Processor::run()
{
    while (! threadShouldExit())
    {
        while (auto* c = inst->takeDone()) delete c;
        std::shared_ptr<FileJob> next;
        {
            const juce::ScopedLock l (jobLock);
            for (auto& j : jobList) if (j->state == 0) { next = j; break; }
        }
        if (! next) { wait (200); continue; }
        next->state = 1;
        if (next->train) trainOfflineJob (*next); else if (next->process) processFile (*next); else feedFile (*next);
    }
}

bool Processor::analyseFile (FileJob& job, const juce::File& file, const std::function<void (const FeedItem&)>& sink, float p0, float p1)
{
    std::unique_ptr<juce::AudioFormatReader> r (formats.createReaderFor (file));
    if (! r) return false;
    const double fsr = r->sampleRate; const auto len = r->lengthInSamples;
    Analyser an; an.prepare (fsr);
    StereoAnalyser st; st.prepare (fsr);
    const bool stereo = r->numChannels >= 2;
    std::array<float, kSpatial> spatNow {};
    const auto cond = TextEmbed::embed (notes.toStdString());
    const int block = (int) std::lround (fsr * 2);   // 2 s chunks
    juce::AudioBuffer<float> b ((int) std::max (1u, r->numChannels), block);
    const double targetSr = inst->sampleRate();
    juce::LagrangeInterpolator li;
    int64_t pos = 0; int hopIdx = 0;
    while (pos < len && ! threadShouldExit() && ! brain().offline.cancel)
    {
        const int n = (int) std::min<juce::int64> (block, len - pos);
        r->read (&b, 0, n, pos, true, true);
        std::vector<float> mono ((size_t) n);
        for (int i = 0; i < n; ++i) { float s2 = 0; for (int c = 0; c < b.getNumChannels(); ++c) s2 += b.getSample (c, i); mono[(size_t) i] = s2 / (float) b.getNumChannels(); }
        auto* chunk = new Instance::Chunk();
        // resample the chunk to the session rate for the corpus
        const double ratio = fsr / targetSr; const int outN = (int) (n / ratio);
        chunk->audio.resize ((size_t) outN); li.process (ratio, mono.data(), chunk->audio.data(), outN);
        const Snapshot& S = brain().acquire();
        // the stereo analyser runs ahead in small steps alongside the mono analysis (both hop every 10 ms)
        int stPos = 0;
        an.push (mono.data(), n, [&] (const RawFrame& raw)
        {
            const int upto = std::min (n, (hopIdx + 1) * an.hopSize() - (int) pos);
            if (stereo && upto > stPos) { st.push (b.getReadPointer (0) + stPos, b.getReadPointer (1) + stPos, upto - stPos, [&] (const std::array<float, kSpatial>& f) { spatNow = f; }); stPos = upto; }
            FeedItem it; toFeatures (raw, it.x);
            std::copy (raw.harm, raw.harm + kHarm, it.harm); std::copy (raw.noise, raw.noise + kNoise, it.noise);
            std::copy (cond.begin(), cond.end(), it.cond); it.voiced = raw.voicing; it.learn = raw.loud > -60;
            // a stereo image to learn from, unless the file is a mono source in two identical channels
            it.hasSpat = stereo && raw.loud > -60 && spatNow[16] > 1e-7f;   // width 0 exactly: identical channels
            std::copy (spatNow.begin(), spatNow.end(), it.spat);
            sink (it);
            ++hopIdx;
            if (raw.loud > -60 && (raw.onset || hopIdx % 25 == 0))
            {
                Unit u; u.len = (int) (0.3 * targetSr);
                const int64_t endInChunk = (int64_t) (((double) (hopIdx * an.hopSize()) - (double) pos) / ratio);
                u.start = std::clamp<int64_t> (endInChunk - u.len, 0, std::max<int64_t> (0, outN - u.len));
                float xs[kFeat]; S.feat.apply (it.x, xs); S.vae.encode (xs, it.cond, u.z.data());
                melCepstrum (raw.mel, u.timbre.data()); std::copy (raw.chroma, raw.chroma + kChroma, u.chroma.begin());
                u.loud = raw.loud; u.f0 = raw.f0;
                chunk->units.push_back (u);
            }
        });
        if (stereo && n > stPos) st.push (b.getReadPointer (0) + stPos, b.getReadPointer (1) + stPos, n - stPos, [&] (const std::array<float, kSpatial>& f) { spatNow = f; });
        brain().release();
        while (! inst->offerChunk (chunk) && ! threadShouldExit()) juce::Thread::sleep (20);
        pos += n;
        job.progress = p0 + (p1 - p0) * (float) pos / (float) std::max<juce::int64> (1, len);
    }
    return true;
}

/** analyse a file into the brain (as fast as the trainer takes it) and hand its audio + units to the corpus */
void Processor::feedFile (FileJob& job)
{
    auto ring = brain().addFeeder();
    brain().settings.intensive = true;
    const bool ok = analyseFile (job, job.file, [&] (const FeedItem& it) { while (! ring->push (it) && ! threadShouldExit()) juce::Thread::sleep (2); }, 0, 1);
    // let the trainer drain the queue
    for (int i = 0; i < 400 && ! threadShouldExit() && brain().backlog() > 1.5; ++i) juce::Thread::sleep (10);
    brain().removeFeeder (ring);
    brain().settings.intensive = false;
    if (! ok) { job.state = 3; job.result = "unreadable"; return; }
    job.state = 2; job.result = brain().settings.frozen ? "remembered (offline: not learned)" : "learned";
}

/** offline training: analyse every file, then train in epochs with a held-out split (Brain::trainOffline) */
void Processor::trainOfflineJob (FileJob& job)
{
    auto& off = brain().offline;
    off.cancel = false; off.state = 1; off.epoch = 0;
    std::vector<std::vector<FeedItem>> data;
    int k = 0;
    for (auto& f : job.files)
    {
        std::vector<FeedItem> items;
        const float p0 = 0.25f * (float) k / (float) job.files.size(), p1 = 0.25f * (float) (k + 1) / (float) job.files.size();
        if (analyseFile (job, f, [&] (const FeedItem& it) { items.push_back (it); }, p0, p1)) data.push_back (std::move (items));
        ++k;
        if (threadShouldExit() || off.cancel) break;
    }
    if (off.cancel || threadShouldExit()) { off.state = 4; job.state = 3; job.result = "cancelled"; return; }
    trainingHere = true;
    std::atomic<bool> done { false };
    std::thread watch ([&] { while (! done) { job.progress = 0.25f + 0.75f * off.progress.load(); std::this_thread::sleep_for (std::chrono::milliseconds (50)); } });
    brain().trainOffline (data, (int) std::lround (apvts.getRawParameterValue ("epochs")->load()), 5);
    done = true; watch.join();
    trainingHere = false;
    job.progress = 1;
    if (off.state == 5) { job.state = 3; job.result = "too little audio to train (needs a few seconds)"; return; }
    job.state = off.state == 4 ? 3 : 2;
    job.result = juce::String (off.state == 4 ? "cancelled at epoch " : "trained: best epoch ") + juce::String (off.bestEpoch.load()) + " of " + juce::String (off.epoch.load())
               + juce::String::fromUTF8 (" · validation ") + juce::String (off.bestVal.load(), 4);
}

void Processor::trainOnFiles (const juce::StringArray& paths)
{
    auto j = std::make_shared<FileJob>(); j->train = true;
    for (auto& p : paths)
    {
        juce::File f (p);
        if (f.isDirectory()) for (auto& c : f.findChildFiles (juce::File::findFiles, true, formats.getWildcardForAllFormats())) j->files.add (c);
        else if (formats.findFormatForFileExtension (f.getFileExtension()) != nullptr) j->files.add (f);
    }
    if (j->files.isEmpty()) return;
    j->file = j->files.size() == 1 ? j->files[0] : j->files[0].getParentDirectory();
    { const juce::ScopedLock l (jobLock); jobList.push_back (j); }
    notify();
}

/** render a file through the network (a separate instance on the same brain) into ~/Music/Neurosonde */
void Processor::processFile (FileJob& job)
{
    std::unique_ptr<juce::AudioFormatReader> r (formats.createReaderFor (job.file));
    if (! r) { job.state = 3; job.result = "unreadable"; return; }
    const double sr = r->sampleRate; const auto len = r->lengthInSamples;
    Instance off (holder->brain);
    off.prepare (sr, 1024);
    readParams (off.params);
    off.params.learn = false;     // processing doesn't teach (feed does)
    off.setNotes (notes.toStdString());
    off.setRoute (route);
    auto outDir = outputFolder(); outDir.createDirectory();
    auto outFile = outDir.getNonexistentChildFile (job.file.getFileNameWithoutExtension() + " neurosonde", ".wav");
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (new juce::FileOutputStream (outFile), sr, 2, 24, {}, 0));
    if (! w) { job.state = 3; job.result = "cannot write"; return; }
    const int B = 1024; const auto total = len + (juce::int64) (sr * 6);
    juce::AudioBuffer<float> in (2, B), out (2, B);
    for (juce::int64 pos = 0; pos < total && ! threadShouldExit(); pos += B)
    {
        in.clear();
        const int n = (int) std::min<juce::int64> (B, std::max<juce::int64> (0, len - pos));
        if (n > 0) { juce::AudioBuffer<float> t ((int) r->numChannels, n); r->read (&t, 0, n, pos, true, true);
                     in.copyFrom (0, 0, t, 0, 0, n); in.copyFrom (1, 0, t, std::min (1, t.getNumChannels() - 1), 0, n); }
        off.process (in.getReadPointer (0), in.getReadPointer (1), out.getWritePointer (0), out.getWritePointer (1), B);
        w->writeFromAudioSampleBuffer (out, 0, B);
        job.progress = (float) pos / (float) total;
    }
    w.reset();
    job.state = 2; job.result = outFile.getFileName();
}
} // namespace ns

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new ns::Processor(); }
