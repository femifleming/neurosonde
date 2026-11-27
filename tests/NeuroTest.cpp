// NeuroTest: instantiate the processor like a host would, play into it, check the output, and screenshot every page.
//   NeuroTest [--shots dir]
#include "../Source/Editor.h"
#include <iostream>
int main (int argc, char** argv)
{
    ::setenv ("NEUROSONDE_BRAIN_DIR", "/private/tmp/neurosonde_test_brain", 1);   // never train the user's real brain
    juce::File ("/private/tmp/neurosonde_test_brain").deleteRecursively();
    juce::ScopedJuceInitialiser_GUI gui;
    juce::File shots;
    for (int i = 1; i < argc; ++i) if (juce::String (argv[i]) == "--shots" && i + 1 < argc) shots = juce::File::getCurrentWorkingDirectory().getChildFile (argv[++i]);
    int fails = 0;
    auto check = [&] (bool ok, const juce::String& what) { std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n"; if (! ok) ++fails; };
    auto proc = std::make_unique<ns::Processor>();
    proc->setPlayConfigDetails (2, 2, 48000, 512);
    proc->prepareToPlay (48000, 512);
    proc->setNotes ("warm tungsten room, dust on the windows, golden light, a slow drift");
    juce::AudioBuffer<float> buf (2, 512); juce::MidiBuffer midi;
    double e = 0; bool finite = true; int64_t pos = 0;
    for (int b = 0; b < 48000 * 12 / 512; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            const double t = (pos + i) / 48000.0; const double f = 196.0 * std::pow (2.0, ((int) (t * 1.5) % 4) * 3 / 12.0);
            const float x = (float) (0.25 * std::exp (-std::fmod (t, 0.66) * 2) * (std::sin (2 * M_PI * f * t) + 0.4 * std::sin (6 * M_PI * f * t)));
            buf.setSample (0, i, x); buf.setSample (1, i, x);
        }
        pos += 512;
        proc->processBlock (buf, midi);
        for (int i = 0; i < 512; ++i) { const float v = buf.getSample (0, i); if (! std::isfinite (v)) finite = false; e += v * v; }
        if (b % 20 == 0) juce::Thread::sleep (2);
    }
    const double rms = std::sqrt (e / (48000 * 12));
    check (finite && rms > 1e-4, "12 s through the processor: finite, rms " + juce::String (rms, 4));
    check (proc->brain().stats.vaeSteps > 50, "the shared brain learned: " + juce::String ((juce::int64) proc->brain().stats.vaeSteps.load()) + " steps");
    // state round trip
    juce::MemoryBlock mb; proc->getStateInformation (mb);
    auto p2 = std::make_unique<ns::Processor>(); p2->setStateInformation (mb.getData(), (int) mb.getSize());
    check (p2->notes == proc->notes, "session keeps the patch notes");
#if NS_SPATIAL
    // 7th-order AmbiX on a 64-channel bus, then binaural
    {
        juce::AudioProcessor::BusesLayout lay; lay.inputBuses.add (juce::AudioChannelSet::stereo()); lay.inputBuses.add (juce::AudioChannelSet::disabled());
        lay.outputBuses.add (juce::AudioChannelSet::ambisonic (7));
        check (proc->setBusesLayout (lay), "accepts a 7th-order ambisonic bus (64 channels)");
        proc->apvts.getParameter ("sformat")->setValueNotifyingHost (0.0f); proc->apvts.getParameter ("sorder")->setValueNotifyingHost (1.0f);
        proc->prepareToPlay (48000, 512);
        juce::AudioBuffer<float> big (64, 512); double e64 = 0; bool fin = true;
        for (int b = 0; b < 200; ++b)
        {
            big.clear(); for (int i = 0; i < 512; ++i) { big.setSample (0, i, 0.2f * std::sin ((b * 512 + i) * 0.03f)); big.setSample (1, i, 0.2f * std::sin ((b * 512 + i) * 0.031f)); }
            proc->processBlock (big, midi);
            for (int c = 0; c < 64; ++c) for (int i = 0; i < 512; ++i) { const float v = big.getSample (c, i); if (! std::isfinite (v)) fin = false; if (c >= 49) e64 += v * v; }
        }
        check (fin && e64 > 1e-6, "7th order: finite, energy in the 7th-order channels " + juce::String (e64, 3));
        // record 64 channels for a moment, check the file
        proc->toggleCapture();
        for (int b = 0; b < 40; ++b) { for (int i = 0; i < 512; ++i) { big.setSample (0, i, 0.2f * std::sin ((b * 512 + i) * 0.03f)); big.setSample (1, i, 0.1f * std::sin ((b * 512 + i) * 0.05f)); } proc->processBlock (big, midi); }
        proc->toggleCapture();
        {
            juce::AudioFormatManager fm; fm.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> rd (fm.createReaderFor (proc->lastCapture));
            check (rd != nullptr && rd->numChannels == 64 && rd->lengthInSamples > 10000 && proc->lastCapture.getFileName().contains ("ambix o7"),
                   "records all 64 channels: " + proc->lastCapture.getFileName() + " (" + juce::String (rd ? (int) rd->numChannels : 0) + " ch)");
            rd.reset(); proc->lastCapture.deleteFile();   // a test render: remove it
        }
        lay.outputBuses.set (0, juce::AudioChannelSet::stereo()); proc->setBusesLayout (lay);
        proc->apvts.getParameter ("sformat")->setValueNotifyingHost (proc->apvts.getParameter ("sformat")->convertTo0to1 (1));
        proc->apvts.getParameter ("sorder")->setValueNotifyingHost (proc->apvts.getParameter ("sorder")->convertTo0to1 (2));
        proc->prepareToPlay (48000, 512);
    }
#endif
    // fuzz: like auval — sample rates, block sizes, every parameter at random, MIDI
    if (std::getenv ("NS_FUZZ"))
    {
        std::mt19937 rng (7); std::uniform_real_distribution<float> u (0, 1);
        for (double sr : { 44100.0, 22050.0, 96000.0, 11025.0, 192000.0 })
        {
            proc->prepareToPlay (sr, 4096);
            juce::AudioBuffer<float> fb (2, 4096); juce::MidiBuffer mm;
            for (int k = 0; k < 120; ++k)
            {
                for (auto* prm : proc->getParameters()) if (u (rng) < 0.3f) prm->setValueNotifyingHost (u (rng));
                const int n = 1 + (int) (u (rng) * 4095);
                for (int c = 0; c < 2; ++c) for (int i = 0; i < n; ++i) fb.setSample (c, i, 0.3f * (u (rng) - 0.5f));
                mm.clear(); mm.addEvent (juce::MidiMessage::noteOn (1, 40 + (int) (u (rng) * 40), (juce::uint8) 100), 0); mm.addEvent (juce::MidiMessage::noteOff (1, 50), n / 2);
                juce::AudioBuffer<float> view (fb.getArrayOfWritePointers(), 2, n);
                proc->processBlock (view, mm);
            }
        }
        check (true, "fuzz survived");
        proc->prepareToPlay (48000, 512);
    }

    // ---- a DAW bounce (non-realtime): every frame reaches the brain, none dropped
    {
        auto& st = proc->brain().stats;
        proc->apvts.getParameter ("learnmode")->setValueNotifyingHost (0.0f);
        proc->setNonRealtime (true);
        const auto seen0 = st.framesSeen.load();
        juce::AudioBuffer<float> bb (2, 2048); const auto t0 = juce::Time::getMillisecondCounterHiRes();
        for (int k = 0; k < 48000 * 30 / 2048; ++k)
        {
            for (int i = 0; i < 2048; ++i) { const double t = (k * 2048 + i) / 48000.0; const float x = (float) (0.2 * std::sin (2 * M_PI * (220 + 40 * std::sin (t)) * t)); bb.setSample (0, i, x); bb.setSample (1, i, 0.7f * x); }
            proc->processBlock (bb, midi);
        }
        proc->setNonRealtime (false);
        const double secs = (juce::Time::getMillisecondCounterHiRes() - t0) / 1000.0;
        for (int k = 0; k < 100 && proc->brain().backlog() > 1.5; ++k) juce::Thread::sleep (10);
        const double got = (double) (st.framesSeen.load() - seen0), want = 30.0 * 100;
        check (got > 0.97 * want, "offline bounce: 30 s rendered in " + juce::String (secs, 1) + " s, the brain saw " + juce::String ((int) got) + " of ~" + juce::String ((int) want) + " frames");
    }
    // ---- offline learning: train on files in epochs, frozen while playing
    {
        auto dir = juce::File ("/private/tmp/neurosonde_test_files"); dir.deleteRecursively(); dir.createDirectory();
        juce::WavAudioFormat wav;
        for (int f = 0; f < 3; ++f)
        {
            auto file = dir.getChildFile ("take" + juce::String (f) + ".wav");
            std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (file.createOutputStream().release(), 48000, 2, 24, {}, 0));
            juce::AudioBuffer<float> fb (2, 48000 * 15); juce::Random rr (f);
            for (int i = 0; i < fb.getNumSamples(); ++i)
            {
                const double t = i / 48000.0; float l = 0, r = 0;
                if (f == 0) { l = (float) (0.3 * std::sin (2 * M_PI * 196 * std::pow (2.0, ((int) (t * 2) % 5) / 12.0) * t)); r = 0.3f * l; }
                else if (f == 1) { l = 0.15f * (rr.nextFloat() - 0.5f); r = 0.15f * (rr.nextFloat() - 0.5f); }
                else { l = r = (float) (0.2 * std::sin (2 * M_PI * 82.4 * t) * (0.6 + 0.4 * std::sin (t * 0.7))); }
                fb.setSample (0, i, l); fb.setSample (1, i, r);
            }
            w->writeFromAudioSampleBuffer (fb, 0, fb.getNumSamples());
        }
        proc->apvts.getParameter ("learnmode")->setValueNotifyingHost (1.0f);   // offline
        proc->apvts.getParameter ("epochs")->setValueNotifyingHost (proc->apvts.getParameter ("epochs")->convertTo0to1 (30));
        juce::AudioBuffer<float> one (2, 512); one.clear(); proc->processBlock (one, midi);   // the mode takes effect
        proc->trainOnFiles ({ dir.getFullPathName() });
        auto& off = proc->brain().offline;
        for (int k = 0; k < 360000 && ! (off.state >= 3); ++k)
        {
            for (int i = 0; i < 512; ++i) { one.setSample (0, i, 0.1f * std::sin (i * 0.1f)); one.setSample (1, i, 0.1f * std::sin (i * 0.1f)); }
            proc->processBlock (one, midi);   // playing while it trains offline
            juce::Thread::sleep (5);
        }
        check (off.state == 3 && off.epoch >= 6 && off.bestVal < off.valCurve[0], "trained offline on 3 files: " + juce::String (off.epoch.load()) + " epochs, best " + juce::String (off.bestEpoch.load())
               + ", validation " + juce::String (off.valCurve[0], 4) + " -> " + juce::String (off.bestVal.load(), 4) + ", " + juce::String (off.spatFrames.load()) + " stereo frames");
        const auto steps = proc->brain().stats.vaeSteps.load();
        for (int k = 0; k < 200; ++k) { for (int i = 0; i < 512; ++i) one.setSample (0, i, 0.2f * std::sin ((k * 512 + i) * 0.02f)); proc->processBlock (one, midi); }
        juce::Thread::sleep (300);
        check (proc->brain().stats.vaeSteps == steps, "offline mode: playing does not train the network");
        proc->apvts.getParameter ("learnmode")->setValueNotifyingHost (0.0f);
        dir.deleteRecursively();
    }
    // ---- threads: several instances on their own audio threads + a UI reader + offline training, all at once
    if (std::getenv ("NS_STRESS"))
    {
        std::vector<std::unique_ptr<ns::Processor>> many;
        for (int k = 0; k < 3; ++k) { auto q = std::make_unique<ns::Processor>(); q->setPlayConfigDetails (2, 2, 48000, 256); q->prepareToPlay (48000, 256); many.push_back (std::move (q)); }
        std::atomic<bool> stop { false }, finite { true };
        std::vector<std::thread> th;
        for (size_t k = 0; k < many.size(); ++k)
            th.emplace_back ([&, k]
            {
                juce::AudioBuffer<float> b (2, 256); juce::MidiBuffer m; juce::Random r ((int) k); int64_t pos = 0;
                while (! stop)
                {
                    for (int i = 0; i < 256; ++i) { const float x = 0.2f * std::sin ((pos + i) * (0.01f + 0.01f * k)) + 0.02f * (r.nextFloat() - 0.5f); b.setSample (0, i, x); b.setSample (1, i, 0.8f * x); }
                    pos += 256;
                    if (r.nextFloat() < 0.05f) for (auto* prm : many[k]->getParameters()) if (r.nextFloat() < 0.1f && prm != many[k]->apvts.getParameter ("learnmode")) prm->setValueNotifyingHost (r.nextFloat());
                    many[k]->processBlock (b, m);
                    for (int i = 0; i < 256; ++i) if (! std::isfinite (b.getSample (0, i))) finite = false;
                }
            });
        th.emplace_back ([&]   // the UI: telemetry, snapshots, journal, memories
        {
            while (! stop)
            {
                for (auto& q : many) { auto t = q->instance().telemetry(); (void) t; }
                const auto& S = many[0]->brain().acquireUi(); volatile float z = S.som.w.empty() ? 0 : S.som.w[0]; (void) z; many[0]->brain().releaseUi();
                (void) many[0]->brain().memories(); (void) many[1]->journal();
                std::this_thread::sleep_for (std::chrono::milliseconds (10));
            }
        });
        // meanwhile: notes changing, a memory saved, files fed and an offline training run
        auto dir = juce::File ("/private/tmp/neurosonde_stress_files"); dir.deleteRecursively(); dir.createDirectory();
        { juce::WavAudioFormat wav; auto file = dir.getChildFile ("a.wav"); std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (file.createOutputStream().release(), 48000, 2, 24, {}, 0));
          juce::AudioBuffer<float> fb (2, 48000 * 8); for (int i = 0; i < fb.getNumSamples(); ++i) { const float x = 0.3f * std::sin (i * 0.02f); fb.setSample (0, i, x); fb.setSample (1, i, 0.5f * x); } w->writeFromAudioSampleBuffer (fb, 0, fb.getNumSamples()); }
        many[1]->setNotes ("rain on glass, slow"); many[2]->addFiles ({ dir.getChildFile ("a.wav").getFullPathName() }, false);
        many[0]->trainOnFiles ({ dir.getFullPathName() });
        many[0]->brain().remember ("stress");
        for (int k = 0; k < 80 && many[0]->brain().offline.state < 3; ++k) { many[2]->setNotes (k % 2 ? "dark" : "bright wide"); juce::Thread::sleep (100); }
        juce::Thread::sleep (500);
        stop = true; for (auto& t : th) t.join();
        check (finite, "3 instances on 3 audio threads + a UI reader + feeding + offline training + memories, concurrently: finite");
        many.clear(); dir.deleteRecursively();
    }
    // ---- the lab: the trainer's instrumentation is filled with real, finite numbers
    {
        auto& L = proc->brain().lab;
        std::lock_guard<std::mutex> l (L.m);
        bool fin = std::isfinite (L.last.total) && std::isfinite (L.last.gnorm) && L.last.lr > 0;
        int named = 0; bool layersOk = true;
        for (auto& st : L.layers) if (! st.name.empty()) { ++named; layersOk = layersOk && std::isfinite (st.wRms) && st.wRms > 0 && std::isfinite (st.gRms); }
        double kl = 0; for (float v : L.klDim) kl += v;
        check (L.count > 50 && fin, "lab: " + juce::String ((juce::int64) L.count) + " optimiser steps recorded, last L " + juce::String (L.last.total, 4) + ", |g| " + juce::String (L.last.gnorm, 3));
        check (named >= 8 && layersOk, "lab: " + juce::String (named) + " layers with finite weight / gradient statistics");
        check (L.W[0].rows == 128 && L.W[0].cols == ns::ml::kFeat && L.W[0].v.size() == (size_t) 128 * ns::ml::kFeat && L.A[0].rows > 0, "lab: weight matrices and batch activations captured");
        check (L.exValid && std::isfinite (L.exErr) && std::isfinite (kl), "lab: held-out reconstruction (mse " + juce::String (L.exErr, 4) + ") and per-dimension KL");
        check (L.log.size() > 20, "lab: training log has " + juce::String ((int) L.log.size()) + " lines");
    }
    // ---- the path from the latent into the synthesis is reported
    {
        const auto t = proc->instance().telemetry();
        double hs = 0; for (float v : t.synHarm) hs += v;
        bool fin = std::isfinite (t.synF0) && std::isfinite (t.synLoud) && std::isfinite (t.lucidMix);
        check (fin && std::abs (hs - 1.0) < 0.05 && t.netSteps > 0, "telemetry: decoder -> engines (harmonics sum " + juce::String (hs, 3) + ", f0 " + juce::String (t.synF0, 1) + " hz, network drives: " + (t.netDrives ? "yes" : "no") + ")");
    }
    // ---- the editor: every page builds, and clicks on the tab bar reach the editor (not the map underneath)
    {
        ns::Editor ed (*proc);
        ed.setVisible (true);   // (hit tests only answer for visible components)
        bool ok = true;
        for (int pg : { 0, 1, 2, 3, 4, 5, 6 })
        {
            if (pg == 5 && ! NS_SPATIAL) continue;
            ed.setPage (pg);
            int vis = 0; for (auto* c : ed.getChildren()) vis += c->isVisible();
            ok = ok && vis >= 3;
        }
        check (ok, "editor: every page shows its controls");
        bool tabs = true;
        for (int x = 0; x < ed.getWidth(); x += 40) tabs = tabs && ed.getComponentAt (x, 15) != nullptr && ed.getComponentAt (x, 15)->findParentComponentOfClass<ns::ui::MapView>() == nullptr && dynamic_cast<ns::ui::MapView*> (ed.getComponentAt (x, 15)) == nullptr;
        for (int x = 0; x < ed.getWidth(); x += 40)
            if (auto* c = ed.getComponentAt (x, 15); c == nullptr || dynamic_cast<ns::ui::MapView*> (c) != nullptr || c->findParentComponentOfClass<ns::ui::MapView>() != nullptr)
                std::cout << "    header x " << x << ": " << (c ? typeid (*c).name() : "null") << "\n";
        check (tabs, "editor: the header strip (tabs, presets, rec) is not covered by the map");
    }
    if (shots != juce::File())
    {
        shots.createDirectory();
        for (int pg : { 0, 1, 2, 3, 4, 5, 6 })
        {
            if (pg == 5 && ! NS_SPATIAL) continue;
            const auto f = shots.getChildFile ("page" + juce::String (pg) + ".png");
            check (ns::Editor::snapshot (*proc, pg, f), "screenshot " + f.getFileName());
        }
    }
    std::cout << (fails ? juce::String (fails) + " FAILED" : juce::String ("all passed")) << "\n";
    p2.reset(); proc.reset();
    return fails ? 1 : 0;
}
