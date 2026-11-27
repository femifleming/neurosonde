// HostCheck: load the INSTALLED Neurosonde plug-ins (VST3 and AU, from ~/Library/Audio/Plug-Ins) the way a DAW does,
// and put them through a battery modelled on pluginval's strictest level, plus more:
//   identity (an audio effect: not an instrument, no MIDI in), bus layouts, sample rates x block sizes (incl. odd and
//   1-sample blocks, variable sizes), silence / DC / full-scale / impulses / denormal-level input, every parameter
//   (text round trip, random automation every block, ramps), state round trip into a fresh instance, prepare /
//   release / reset cycles, bypass, non-realtime rendering, editors opened and closed while audio runs, several
//   instances processing at once on their own threads, real-time factor, and a soak (minutes of audio: finite,
//   bounded, no memory growth, no CPU creep).
//   HostCheck [--soak seconds] [--only vst3|au] [--plugin name]
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <iostream>
#include <mach/mach.h>

static int fails = 0, checks = 0;
static void check (bool ok, const juce::String& what) { ++checks; std::cout << (ok ? "  ok   " : "  FAIL ") << what << std::endl; if (! ok) ++fails; }
static size_t residentBytes()
{
    mach_task_basic_info info; mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    return task_info (mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info, &n) == KERN_SUCCESS ? (size_t) info.resident_size : 0;
}

/** a host hands the plug-in max(inputs, outputs) channels (the sidechain counts) */
static int chans (const juce::AudioProcessor& p) { return std::max ({ 2, p.getTotalNumInputChannels(), p.getTotalNumOutputChannels() }); }

struct Stats { double peak = 0; bool finite = true; void take (const juce::AudioBuffer<float>& b, int ch) { for (int c = 0; c < ch; ++c) for (int i = 0; i < b.getNumSamples(); ++i) { const float v = b.getSample (c, i); if (! std::isfinite (v)) finite = false; else peak = std::max (peak, (double) std::abs (v)); } } };

static void fill (juce::AudioBuffer<float>& b, int kind, int64_t& pos, juce::Random& r)
{
    for (int c = 0; c < b.getNumChannels(); ++c)
        for (int i = 0; i < b.getNumSamples(); ++i)
        {
            const double t = (double) (pos + i) / 48000.0; float x = 0;
            switch (kind)
            {
                case 0: x = 0; break;                                                                              // silence
                case 1: x = (float) (0.3 * std::sin (2 * M_PI * 220 * t * (1 + 0.01 * c)) + 0.05 * (r.nextFloat() - 0.5f)); break;  // music-ish
                case 2: x = 1.0f; break;                                                                           // DC full scale
                case 3: x = std::sin (2 * M_PI * 50 * t) > 0 ? 1.0f : -1.0f; break;                                // full-scale square
                case 4: x = ((pos + i) % 4800 == 0) ? 1.0f : 0.0f; break;                                          // impulses
                case 5: x = 1e-30f * (r.nextFloat() - 0.5f); break;                                                // denormal-level
                case 6: x = r.nextFloat() * 2 - 1; break;                                                          // full-scale noise
            }
            b.setSample (c, i, x);
        }
    pos += b.getNumSamples();
}

static void runSuite (juce::AudioPluginFormatManager& fm, const juce::PluginDescription& d, double soakSeconds)
{
    std::cout << "\n== " << d.name << " (" << d.pluginFormatName << ")  " << d.fileOrIdentifier << std::endl;
    juce::String err;
    auto make = [&] (double sr, int bs) { return fm.createPluginInstance (d, sr, bs, err); };
    auto p = make (48000, 512);
    check (p != nullptr, "loads" + (err.isNotEmpty() ? " (" + err + ")" : juce::String()));
    if (! p) return;
    const bool spatial = d.name.containsIgnoreCase ("space");

    // ---- identity: an audio effect
    check (! d.isInstrument, "is an audio effect, not an instrument (category: " + d.category + ")");
    check (! p->acceptsMidi() && ! p->producesMidi(), "no MIDI in or out");
    check (p->getTotalNumInputChannels() >= 1 && p->getMainBusNumInputChannels() == 2, "has a stereo audio input (" + juce::String (p->getMainBusNumInputChannels()) + " ch)");

    // ---- layouts
    {
        auto lay = p->getBusesLayout();
        auto tryOut = [&] (juce::AudioChannelSet set) { auto l = lay; l.outputBuses.getReference (0) = set; return p->checkBusesLayoutSupported (l); };
        check (tryOut (juce::AudioChannelSet::stereo()), "stereo out supported");
        if (spatial)
        {
            int maxOrder = 0; for (int o = 1; o <= 7; ++o) if (tryOut (juce::AudioChannelSet::ambisonic (o))) maxOrder = o; else break;
            const bool au = d.pluginFormatName == "AudioUnit";
            // VST3 carries any order; an AU host only negotiates the HOA layouts it knows (Logic: up to 3rd order)
            check (maxOrder >= (au ? 1 : 7), "ambisonic output up to order " + juce::String (maxOrder) + " (" + juce::String ((maxOrder + 1) * (maxOrder + 1)) + " ch)" + (au ? " as AU" : ""));
            check (tryOut (juce::AudioChannelSet::create5point1()) && tryOut (juce::AudioChannelSet::quadraphonic()), "5.1 and quad supported");
        }
        else check (! tryOut (juce::AudioChannelSet::create5point1()), "5.1 out refused (stereo effect)");
    }

    // ---- every parameter: text round trip
    {
        int bad = 0; juce::String which;
        for (auto* prm : p->getParameters())
            for (float v : { 0.0f, 0.25f, 0.5f, 1.0f })
            {
                const auto txt = prm->getText (v, 64);
                const float back = prm->getValueForText (txt);
                if (txt.isEmpty() || ! std::isfinite (back)) { ++bad; which = prm->getName (64); }
            }
        check (bad == 0, juce::String (p->getParameters().size()) + " parameters: text round trip" + (bad ? " (" + which + ")" : juce::String()));
    }

    // ---- rates x block sizes x signals
    juce::Random rnd (1);
    {
        int runs = 0; Stats s; bool shape = true;
        for (double sr : { 22050.0, 44100.0, 48000.0, 88200.0, 96000.0, 192000.0 })
            for (int bs : { 1, 7, 64, 511, 1024, 4096 })
            {
                p->releaseResources(); p->setRateAndBufferSizeDetails (sr, bs); p->prepareToPlay (sr, bs);
                const int ch = std::max (p->getTotalNumInputChannels(), p->getTotalNumOutputChannels());
                juce::AudioBuffer<float> b (ch, bs); juce::MidiBuffer m; int64_t pos = 0;
                const int blocks = std::max (4, (int) (sr * 0.4 / bs));
                for (int k = 0; k < blocks; ++k)
                {
                    fill (b, (k / 3) % 7, pos, rnd);
                    p->processBlock (b, m);
                    s.take (b, p->getTotalNumOutputChannels());
                    if (b.getNumSamples() != bs) shape = false;
                }
                ++runs;
            }
        check (s.finite, juce::String (runs) + " rate/block combinations x 7 signals (silence, music, DC, square, impulses, denormal, noise): all output finite");
        check (s.peak < 4.0, "output bounded (peak " + juce::String (s.peak, 2) + ")");
        check (shape, "buffers left intact");
    }

    // ---- variable block sizes within the prepared maximum (hosts do this)
    {
        p->releaseResources(); p->prepareToPlay (48000, 1024);
        juce::AudioBuffer<float> b (chans (*p), 1024); juce::MidiBuffer m; Stats s; int64_t pos = 0;
        for (int k = 0; k < 600; ++k) { const int n = 1 + rnd.nextInt (1024); juce::AudioBuffer<float> v (b.getArrayOfWritePointers(), b.getNumChannels(), n); fill (v, 1, pos, rnd); p->processBlock (v, m); s.take (v, 2); }
        check (s.finite && s.peak < 4, "600 random block sizes 1..1024: finite, peak " + juce::String (s.peak, 2));
    }

    // ---- parameter automation: every parameter random every block, and slow ramps
    {
        p->releaseResources(); p->prepareToPlay (48000, 256);
        juce::AudioBuffer<float> b (chans (*p), 256); juce::MidiBuffer m; Stats s; int64_t pos = 0;
        auto& params = p->getParameters();
        for (int k = 0; k < 1500; ++k)
        {
            if (k < 750) { for (auto* prm : params) if (rnd.nextFloat() < 0.5f) prm->setValueNotifyingHost (rnd.nextFloat()); }
            else for (int q = 0; q < params.size(); ++q) params[q]->setValueNotifyingHost (0.5f + 0.5f * std::sin (k * 0.01f + q));
            fill (b, 1 + (k / 100) % 6, pos, rnd); p->processBlock (b, m); s.take (b, 2);
        }
        check (s.finite && s.peak < 4, "1500 blocks of random + ramped automation on all parameters: finite, peak " + juce::String (s.peak, 2));
        for (auto* prm : params) prm->setValueNotifyingHost (prm->getDefaultValue());
    }

    // ---- state round trip into a fresh instance
    {
        for (auto* prm : p->getParameters()) prm->setValueNotifyingHost (rnd.nextFloat());
        juce::MemoryBlock mb; p->getStateInformation (mb);
        auto q = make (48000, 512);
        bool same = q != nullptr && mb.getSize() > 0; juce::String mismatch;
        if (q)
        {
            q->setStateInformation (mb.getData(), (int) mb.getSize());
            // compare what each parameter shows (choices snap to their steps: the raw float may differ, the value not)
            for (int i = 0; i < p->getParameters().size(); ++i)
            {
                auto* a2 = p->getParameters()[i]; auto* b2 = q->getParameters()[i];
                if (a2->getCurrentValueAsText() != b2->getCurrentValueAsText() && std::abs (a2->getValue() - b2->getValue()) > 1e-3f)
                { if (same) mismatch = a2->getName (64) + " " + a2->getCurrentValueAsText() + " vs " + b2->getCurrentValueAsText(); same = false; }
            }
        }
        check (same, "state round trip: " + juce::String ((int) mb.getSize()) + " bytes, every parameter restored in a new instance" + (same ? juce::String() : " (" + mismatch + ")"));
        for (auto* prm : p->getParameters()) prm->setValueNotifyingHost (prm->getDefaultValue());
        // garbage state must not crash
        juce::MemoryBlock junk (777); for (size_t i = 0; i < junk.getSize(); ++i) junk[i] = (char) rnd.nextInt (256);
        if (q) q->setStateInformation (junk.getData(), (int) junk.getSize());
        check (true, "survives a corrupt state blob");
    }

    // ---- prepare / release / reset cycles, bypass, non-realtime
    {
        juce::MidiBuffer m; Stats s; int64_t pos = 0;
        for (int k = 0; k < 20; ++k)
        {
            p->releaseResources(); const double sr = k % 2 ? 44100 : 96000; p->setRateAndBufferSizeDetails (sr, 256); p->prepareToPlay (sr, 256); p->reset();
            juce::AudioBuffer<float> b (chans (*p), 256);
            for (int i = 0; i < 10; ++i) { fill (b, 1, pos, rnd); if (i % 3 == 0) p->processBlockBypassed (b, m); else p->processBlock (b, m); s.take (b, 2); }
        }
        check (s.finite, "20 prepare / release / reset cycles at alternating rates, with bypass");
        p->setNonRealtime (true); p->prepareToPlay (48000, 4096);
        juce::AudioBuffer<float> b (chans (*p), 4096); Stats s2;
        const auto t0 = juce::Time::getMillisecondCounterHiRes();
        for (int k = 0; k < 120; ++k) { fill (b, 1, pos, rnd); p->processBlock (b, m); s2.take (b, 2); }
        const double secs = (juce::Time::getMillisecondCounterHiRes() - t0) / 1000.0;
        p->setNonRealtime (false);
        check (s2.finite, "non-realtime render of " + juce::String (120 * 4096 / 48000.0, 1) + " s took " + juce::String (secs, 2) + " s");
    }

    // ---- real-time factor at 48 kHz / 512
    {
        p->releaseResources(); p->prepareToPlay (48000, 512);
        juce::AudioBuffer<float> b (chans (*p), 512); juce::MidiBuffer m; int64_t pos = 0;
        double worst = 0, total = 0; const int N = 2000;
        for (int k = 0; k < N; ++k)
        {
            fill (b, 1, pos, rnd);
            const auto t0 = juce::Time::getMillisecondCounterHiRes(); p->processBlock (b, m); const double ms = juce::Time::getMillisecondCounterHiRes() - t0;
            total += ms; if (k > 50) worst = std::max (worst, ms);
        }
        const double blockMs = 512 / 48.0;
        check (total / N < blockMs * 0.5, "cpu: mean " + juce::String (100 * total / N / blockMs, 1) + "% of real time, worst block " + juce::String (100 * worst / blockMs, 0) + "% (default patch, 48 kHz / 512)");
    }

    // ---- editor open / close while audio runs on another thread
    if (p->hasEditor())
    {
        std::atomic<bool> stop { false }; Stats s;
        std::thread audio ([&]
        {
            juce::AudioBuffer<float> b (chans (*p), 256); juce::MidiBuffer m; int64_t pos = 0; juce::Random r2 (5);
            while (! stop) { fill (b, 1, pos, r2); p->processBlock (b, m); s.take (b, 2); std::this_thread::sleep_for (std::chrono::microseconds (500)); }
        });
        p->prepareToPlay (48000, 256);
        bool opened = true;
        for (int k = 0; k < 12; ++k)
        {
            std::unique_ptr<juce::AudioProcessorEditor> ed (p->createEditorIfNeeded());
            opened = opened && ed != nullptr;
            if (ed) { ed->setSize (ed->getWidth(), ed->getHeight()); juce::MessageManager::getInstance()->runDispatchLoopUntil (120); for (auto* prm : p->getParameters()) if (rnd.nextFloat() < 0.1f) prm->setValueNotifyingHost (rnd.nextFloat()); }
            p->editorBeingDeleted (ed.get());
        }
        stop = true; audio.join();
        check (opened && s.finite, "editor opened and closed 12 times while audio ran and parameters changed");
        for (auto* prm : p->getParameters()) prm->setValueNotifyingHost (prm->getDefaultValue());
    }

    // ---- several instances at once, each on its own audio thread (one shared brain)
    {
        std::vector<std::unique_ptr<juce::AudioPluginInstance>> many;
        for (int k = 0; k < 4; ++k) if (auto q = make (48000, 256)) { q->prepareToPlay (48000, 256); many.push_back (std::move (q)); }
        std::atomic<bool> ok { true };
        std::vector<std::thread> th;
        for (size_t k = 0; k < many.size(); ++k)
            th.emplace_back ([&, k]
            {
                juce::AudioBuffer<float> b (chans (*many[k]), 256); juce::MidiBuffer m; int64_t pos = 0; juce::Random r2 ((int) k); Stats s;
                for (int i = 0; i < 1500; ++i) { fill (b, 1 + (int) (k % 3), pos, r2); many[k]->processBlock (b, m); s.take (b, 2); }
                if (! s.finite) ok = false;
            });
        for (auto& t : th) t.join();
        check (ok && many.size() == 4, "4 instances processing at once on 4 threads (shared brain): finite");
        many.clear();
    }

    // ---- soak
    if (soakSeconds > 0)
    {
        p->releaseResources(); p->prepareToPlay (48000, 512);
        juce::AudioBuffer<float> b (chans (*p), 512); juce::MidiBuffer m; int64_t pos = 0; Stats s;
        const int blocks = (int) (soakSeconds * 48000 / 512);
        size_t memStart = 0; double cpuFirst = 0, cpuLast = 0; const int window = blocks / 10;
        for (int k = 0; k < blocks; ++k)
        {
            if (k == blocks / 10) memStart = residentBytes();
            if (k % 3000 == 0) for (auto* prm : p->getParameters()) if (rnd.nextFloat() < 0.2f) prm->setValueNotifyingHost (rnd.nextFloat());
            fill (b, 1 + (k / 2000) % 6, pos, rnd);
            const auto t0 = juce::Time::getMillisecondCounterHiRes(); p->processBlock (b, m); const double ms = juce::Time::getMillisecondCounterHiRes() - t0;
            if (k >= blocks / 10 && k < blocks / 10 + window) cpuFirst += ms; if (k >= blocks - window) cpuLast += ms;
            s.take (b, 2);
        }
        const double grow = (double) residentBytes() - (double) memStart;
        check (s.finite && s.peak < 4, "soak: " + juce::String (soakSeconds / 60.0, 1) + " min of audio with parameter changes, finite, peak " + juce::String (s.peak, 2));
        check (grow < 64.0 * 1024 * 1024, "soak: memory after warm-up grew " + juce::String (grow / 1048576.0, 1) + " MB");
        check (cpuLast < cpuFirst * 1.6 + 5, "soak: no cpu creep (last tenth " + juce::String (cpuLast / window, 2) + " ms/block vs " + juce::String (cpuFirst / window, 2) + ")");
    }
    p.reset();
}

int main (int argc, char** argv)
{
    ::setenv ("NEUROSONDE_BRAIN_DIR", "/private/tmp/neurosonde_hostcheck_brain", 1);   // never the user's real brain
    juce::File ("/private/tmp/neurosonde_hostcheck_brain").deleteRecursively();
    juce::ScopedJuceInitialiser_GUI gui;
    double soak = 0; juce::String only, name;
    for (int i = 1; i < argc; ++i)
    {
        const juce::String a (argv[i]);
        if (a == "--soak" && i + 1 < argc) soak = juce::String (argv[++i]).getDoubleValue();
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--plugin" && i + 1 < argc) name = argv[++i];
    }
    juce::AudioPluginFormatManager fm; juce::addDefaultFormatsToManager (fm);
    // --vartest NAME: only the variable-block-size test, on any installed AU by name (to tell host bugs from plug-in bugs)
    for (int i = 1; i + 1 < argc; ++i) if (juce::String (argv[i]) == "--vartest")
    {
        const juce::String want (argv[i + 1]);
        for (auto* f : fm.getFormats()) if (f->getName() == "AudioUnit")
            for (auto& id : f->searchPathsForPlugins ({}, false, false))
            {
                juce::OwnedArray<juce::PluginDescription> ds; f->findAllTypesForFile (ds, id);
                for (auto* d : ds) if (d->name == want && d->pluginFormatName == "AudioUnit")
                {
                    juce::String err; auto p = fm.createPluginInstance (*d, 48000, 1024, err);
                    if (! p) { std::cout << "cannot load " << want << "\n"; return 1; }
                    const int mode = argc > i + 2 ? juce::String (argv[i + 2]).getIntValue() : 0;
                    if (mode == 1) { p->prepareToPlay (192000, 4096); p->releaseResources(); }
                    if (mode != 5) p->enableAllBuses(); std::cout << "in " << p->getTotalNumInputChannels() << " out " << p->getTotalNumOutputChannels() << std::endl;
                    p->setRateAndBufferSizeDetails (48000, 1024); p->prepareToPlay (48000, 1024);
                    juce::AudioBuffer<float> b (2, 1024); juce::MidiBuffer m; juce::Random r (1);
                    for (int k = 0; k < 600; ++k) { const int n = mode == 3 ? 1024 : mode == 4 ? 512 : 1 + r.nextInt (1024); juce::AudioBuffer<float> v (2, n); if (mode >= 2) v = juce::AudioBuffer<float> (b.getArrayOfWritePointers(), 2, n); for (int c = 0; c < 2; ++c) for (int q = 0; q < n; ++q) v.setSample (c, q, 0.1f * (r.nextFloat() - 0.5f)); p->processBlock (v, m); }
                    std::cout << want << " mode " << mode << ": variable blocks ok\n"; return 0;
                }
            }
        std::cout << "not found\n"; return 1;
    }
    const auto home = juce::File::getSpecialLocation (juce::File::userHomeDirectory);
    std::vector<juce::PluginDescription> found;
    for (auto* f : fm.getFormats())
    {
        const bool vst3 = f->getName() == "VST3", au = f->getName() == "AudioUnit";
        if (! vst3 && ! au) continue;
        if (only.isNotEmpty() && ! (only == "vst3" ? vst3 : au)) continue;
        juce::StringArray ids;
        if (vst3) for (auto* n : { "Neurosonde.vst3", "Neurosonde Space.vst3" }) ids.add (home.getChildFile ("Library/Audio/Plug-Ins/VST3").getChildFile (n).getFullPathName());
        else ids.addArray ({ "AudioUnit:Effects/aufx,Nsnd,Frkp", "AudioUnit:Effects/aufx,Nssp,Frkp" });   // only ours (a full scan loads every AU on the machine)
        for (auto& id : ids)
        {
            juce::OwnedArray<juce::PluginDescription> ds; f->findAllTypesForFile (ds, id);
            for (auto* d : ds)
                if (d->manufacturerName == "Neurosonde" && d->name.startsWith ("Neurosonde") && ! d->name.containsIgnoreCase ("midi")
                    && (name.isEmpty() || d->name == name))
                    found.push_back (*d);
        }
    }
    check (found.size() >= (only.isEmpty() ? 4u : 2u), "found the installed plug-ins: " + juce::String ((int) found.size()));
    for (auto& d : found) runSuite (fm, d, soak);
    std::cout << "\n" << checks << " checks, " << (fails ? juce::String (fails) + " FAILED" : juce::String ("all passed")) << std::endl;
    juce::File ("/private/tmp/neurosonde_hostcheck_brain").deleteRecursively();
    return fails ? 1 : 0;
}
