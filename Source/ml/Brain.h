// Neurosonde — the brain: one per process, shared by every plug-in instance and the app in that process, and with
// other processes through a brain file (federated delta merging under a file lock).
//
//   audio threads ──(lock-free rings of analysed frames)──▶ trainer thread ──(double-buffered snapshot)──▶ inference
//
// The trainer standardises features (running statistics), keeps a replay reservoir (protection against catastrophic
// forgetting), trains the conditional VAE every few frames, the latent GRU on replayed sequences, the echo state
// network's RLS readout and the SOM on the live latent stream, and keeps consolidating from replay when the input is
// quiet ("sleep"). Weights are published to the audio side ten times a second.
#pragma once
#include "Models.h"
#include "Lab.h"
#include "../dsp/Analysis.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <fstream>
#include <chrono>
#include <set>
#include <numeric>
#include <sstream>
#include <filesystem>
#if defined(_WIN32)
 // the C runtime's file locking: no <windows.h> in a header everything includes (its macros, near, far, IN, OUT,
 // small ..., break ordinary code)
 #include <io.h>
 #include <fcntl.h>
 #include <share.h>
 #include <sys/locking.h>
 #include <sys/stat.h>
#else
 #include <sys/file.h>
 #include <fcntl.h>
 #include <unistd.h>
#endif

namespace ns::ml
{
/** a path from a UTF-8 string (on Windows a narrow string would be read in the ANSI code page) */
inline std::filesystem::path fsPath (const std::string& s)
{
   #if defined(_WIN32)
    return std::filesystem::path (std::u8string (s.begin(), s.end()));
   #else
    return std::filesystem::path (s);
   #endif
}

/** an exclusive lock on a file, held for its lifetime (flock on the Mac and Linux, LockFileEx on Windows): the
    brain file is merged by every process that uses it */
struct FileLock
{
   #if defined(_WIN32)
    int fd = -1; bool locked = false;
    explicit FileLock (const std::string& path)
    {
        if (_wsopen_s (&fd, fsPath (path).wstring().c_str(), _O_CREAT | _O_RDWR | _O_BINARY, _SH_DENYNO, _S_IREAD | _S_IWRITE) != 0) { fd = -1; return; }
        // _LK_LOCK retries for about 10 s before it gives up; keep trying (another process holds the brain)
        for (int k = 0; k < 30 && ! locked; ++k) { _lseek (fd, 0, SEEK_SET); locked = _locking (fd, _LK_LOCK, 1) == 0; }
    }
    ~FileLock() { if (fd >= 0) { if (locked) { _lseek (fd, 0, SEEK_SET); _locking (fd, _LK_UNLCK, 1); } _close (fd); } }
    bool ok() const { return fd >= 0 && locked; }
   #else
    int fd = -1;
    explicit FileLock (const std::string& path) { fd = ::open (path.c_str(), O_CREAT | O_RDWR, 0644); if (fd >= 0) ::flock (fd, LOCK_EX); }
    ~FileLock() { if (fd >= 0) { ::flock (fd, LOCK_UN); ::close (fd); } }
    bool ok() const { return fd >= 0; }
   #endif
    FileLock (const FileLock&) = delete; FileLock& operator= (const FileLock&) = delete;
};

/** single-producer single-consumer ring (wait-free on both sides) */
template <typename T, size_t Cap>
struct SpscRing
{
    std::array<T, Cap> buf {};
    std::atomic<size_t> head { 0 }, tail { 0 };
    bool push (const T& v)
    {
        const size_t h = head.load (std::memory_order_relaxed), n = (h + 1) % Cap;
        if (n == tail.load (std::memory_order_acquire)) return false;
        buf[h] = v; head.store (n, std::memory_order_release); return true;
    }
    bool empty() const { return head.load (std::memory_order_acquire) == tail.load (std::memory_order_acquire); }
    bool pop (T& v)
    {
        const size_t t = tail.load (std::memory_order_relaxed);
        if (t == head.load (std::memory_order_acquire)) return false;
        v = buf[t]; tail.store ((t + 1) % Cap, std::memory_order_release); return true;
    }
};

/** what an audio thread sends for each analysed frame */
struct FeedItem
{
    float x[kFeat];          // raw (unstandardised) features
    float harm[kHarm];
    float noise[kNoise];     // dB
    float cond[kCond];
    float voiced = 0;
    bool learn = true;       // false: only observed, not learned (frozen / below gate)
    float spat[kSpatial] {}; // raw spatial features
    bool hasSpat = false;    // the input was stereo (and not silent)
};

/** everything the audio side needs to run the network, copied from the trainer */
struct Snapshot
{
    VAE vae; LatentGRU gru; EchoState esn; SOM som; SpatialNet spatial;
    RunningNorm feat, noise, spat;
    uint64_t steps = 0, frames = 0, spatSteps = 0;
};

struct Stats
{
    std::atomic<uint64_t> vaeSteps { 0 }, gruSteps { 0 }, framesSeen { 0 }, framesLearned { 0 }, esnUpdates { 0 };
    std::atomic<float> recon { 0 }, kl { 0 }, harm { 0 }, gruNll { 0 }, esnErr { 0 }, surprise { 0 }, beta { 0 }, lr { 0 }, spatNll { 0 };
    std::atomic<uint64_t> spatSteps { 0 }, spatFrames { 0 };
    // held-out evaluation (every 10th learned frame is never trained on): generalisation, not just fit
    std::atomic<float> trainRecon { 0 }, valRecon { 0 }, spatValNll { 0 }, spatCoverage { 0 };
    std::atomic<int> activeUnits { 0 }, holdout { 0 };
    std::array<std::atomic<float>, 16> unitVar {};   // Cov_x E_q[z_u] per latent dimension (Burda et al. 2016)
    float evTrainHist[128] {}, evValHist[128] {}; std::atomic<int> evalPos { 0 };   // the same eval-mode loss on both sets
    std::atomic<int> lastBmu { -1 }, replaySize { 0 }, merges { 0 };
    std::atomic<bool> sleeping { false };
    // loss history for the UI (written by the trainer, read approximately by the UI)
    static constexpr int kHist = 256;
    float reconHist[kHist] {}, gruHist[kHist] {};
    std::atomic<int> histPos { 0 };
};

class Brain
{
public:
    static constexpr uint32_t kMagic = 0x3142534e;   // "NSB1"
    static constexpr uint32_t kVersion = 2;   // 2: + spatial network

    struct Settings
    {
        std::atomic<float> plasticity { 1.0f };     // learning-rate scale (0 = frozen)
        std::atomic<float> sleepRate { 0.25f };     // consolidation speed when idle (0..1)
        std::atomic<float> betaMax { 0.2f };        // KL weight ceiling
        std::atomic<bool> frozen { false };
        std::atomic<bool> intensive { false };      // files being fed: train as fast as possible
    } settings;
    Stats stats;
    Lab lab;   // instrumentation of the training, for the lab page (ml/Lab.h)

    explicit Brain (std::string folder) : dir (std::move (folder))
    {
        initModels (1234);
        load (dir + "/brain.nsb");
        publish (true);
        running = true;
        worker = std::thread ([this] { run(); });
    }
    ~Brain()
    {
        running = false; cv.notify_all();
        if (worker.joinable()) worker.join();
        sync();
    }

    // ---- feeders (one per plug-in instance) ----
    using Ring = SpscRing<FeedItem, 2048>;
    std::shared_ptr<Ring> addFeeder() { auto r = std::make_shared<Ring>(); std::lock_guard<std::mutex> l (feedLock); feeders.push_back (r); return r; }
    void removeFeeder (const std::shared_ptr<Ring>& r) { std::lock_guard<std::mutex> l (feedLock); feeders.erase (std::remove (feeders.begin(), feeders.end(), r), feeders.end()); }

    // ---- inference side ----
    /** The audio thread calls acquire() once per hop and release() when done with the snapshot. */
    /** Readers (every instance's audio thread, the UI, file jobs) count themselves in on a slot. The re-check closes
        the gap between reading `published` and counting in; the slot a thread holds is remembered per thread, so
        several instances on several threads never let go of each other's snapshot. */
    const Snapshot& acquire() { auto& h = heldSlot(); h = enter(); return slots[(size_t) h]; }
    void release() { auto& h = heldSlot(); if (h >= 0) { readers[(size_t) h].fetch_sub (1); h = -1; } }
    /** the UI's own claim on a snapshot (never the audio thread's flag) */
    const Snapshot& acquireUi() { auto& h = heldSlotUi(); h = enter(); return slots[(size_t) h]; }
    void releaseUi() { auto& h = heldSlotUi(); if (h >= 0) { readers[(size_t) h].fetch_sub (1); h = -1; } }
    uint64_t snapshotVersion() const { return version.load(); }

    // ---- star names (words the user wrote while a node was active) ----
    std::vector<std::pair<std::string, float>> starWords (int node, int maxWords = 2)
    {
        std::lock_guard<std::mutex> l (wordLock);
        std::vector<std::pair<std::string, float>> out;
        if (auto it = nodeWords.find (node); it != nodeWords.end())
        {
            for (auto& kv : it->second) out.push_back (kv);
            std::sort (out.begin(), out.end(), [] (auto& a, auto& b) { return a.second > b.second; });
            if ((int) out.size() > maxWords) out.resize ((size_t) maxWords);
        }
        return out;
    }
    void setNoteWords (std::vector<std::string> w) { std::lock_guard<std::mutex> l (wordLock); currentWords = std::move (w); }

    // ---- memories (snapshots on disk) ----
    struct Memory { std::string file, label; uint64_t steps = 0; int64_t time = 0; };
    std::vector<Memory> memories() const
    {
        std::vector<Memory> out;
        std::ifstream idx (fsPath (dir + "/memories/index.txt"));
        std::string line;
        while (std::getline (idx, line))
        {
            Memory m; std::stringstream ss (line); std::string steps, t;
            if (std::getline (ss, m.file, '\t') && std::getline (ss, steps, '\t') && std::getline (ss, t, '\t'))
            {
                std::getline (ss, m.label);
                // a damaged line is skipped, never thrown (this runs on the UI thread)
                char* e1 = nullptr; char* e2 = nullptr;
                m.steps = std::strtoull (steps.c_str(), &e1, 10); m.time = std::strtoll (t.c_str(), &e2, 10);
                if (e1 != steps.c_str() && e2 != t.c_str()) out.push_back (m);
            }
        }
        return out;
    }
    /** save the current brain as a named memory */
    void remember (const std::string& label)
    {
        std::lock_guard<std::mutex> l (modelLock);
        const auto t = (int64_t) std::chrono::duration_cast<std::chrono::seconds> (std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string name = "m" + std::to_string (t) + "_" + std::to_string (vaeSteps) + ".nsb";
        mkdirs (dir + "/memories");
        save (dir + "/memories/" + name);
        std::ofstream idx (fsPath (dir + "/memories/index.txt"), std::ios::app);
        std::string clean = label; for (auto& c : clean) if (c == '\t' || c == '\n') c = ' ';
        idx << name << '\t' << vaeSteps << '\t' << t << '\t' << clean << '\n';
    }
    /** go back to a memory (the shared brain becomes that state at the next sync) */
    bool recall (const std::string& file)
    {
        std::lock_guard<std::mutex> l (modelLock);
        if (! load (dir + "/memories/" + file)) return false;
        forceOverwrite = true;
        publishNow = true;
        return true;
    }
    /** forget everything: fresh random weights (the brain file is overwritten at the next sync) */
    void reset()
    {
        std::lock_guard<std::mutex> l (modelLock);
        initModels ((uint32_t) std::chrono::steady_clock::now().time_since_epoch().count());
        forceOverwrite = true; publishNow = true;
    }
    void requestSync() { syncNow = true; cv.notify_all(); }
    const std::string& folder() const { return dir; }

    /** for tests: run the trainer loop body synchronously */
    void trainFor (int iterations) { for (int i = 0; i < iterations; ++i) iterate (true); publish (true); }

    /** how much learning is still queued: frames waiting in the feeders + gradient steps owed (0 = caught up) */
    double backlog()
    {
        size_t pending = 0;
        { std::lock_guard<std::mutex> l (feedLock); for (auto& r : feeders) pending += r->empty() ? 0 : 1; }
        return (double) pending * 1000 + owedNow.load();
    }
    void wake() { cv.notify_one(); }

    // ---- offline training -------------------------------------------------------------------------------------
    /** progress / result of the last offline training run (read by the UI) */
    struct Offline
    {
        std::atomic<int> state { 0 };          // 0 idle, 1 analysing, 2 training, 3 done, 4 cancelled, 5 too little data
        std::atomic<int> epoch { 0 }, epochs { 0 }, bestEpoch { 0 }, trainFrames { 0 }, valFrames { 0 }, spatFrames { 0 };
        std::atomic<float> progress { 0 }, trainLoss { 0 }, valLoss { 0 }, bestVal { 0 }, spatVal { 0 };
        std::atomic<bool> cancel { false };
        float trainCurve[256] {}, valCurve[256] {};
    } offline;

    /** Batch training on a fixed dataset, as opposed to the online learning that runs while playing:
        - the frames are standardised with statistics of the whole set (two passes),
        - split into training / validation by contiguous 2 s blocks (every 10th block validates), so neighbouring,
          nearly identical frames never straddle the split,
        - shuffled minibatch epochs (VAE every batch, spatial net every 2nd, GRU every 4th on 25-frame windows that lie
          inside training blocks), learning rate decaying per epoch,
        - validation after each epoch; early stopping after `patience` epochs without improvement; the best weights
          (by validation loss) are restored at the end,
        - then the constellation (SOM) is refitted on the training latents and both sets join the replay / held-out
          memories, so live learning continues from here. */
    void trainOffline (const std::vector<std::vector<FeedItem>>& files, int maxEpochs, int patience = 5)
    {
        offline.state = 2; offline.epoch = 0; offline.epochs = maxEpochs; offline.bestEpoch = 0; offline.progress = 0;
        { std::lock_guard<std::mutex> ll (lab.m); lab.line ("== offline training: %d file(s), up to %d epochs, patience %d (live learning paused)", (int) files.size(), maxEpochs, patience); }
        std::vector<Frame> train, val; std::vector<SeqChunk> trainSeq;
        {
            std::lock_guard<std::mutex> lk (modelLock);
            // pass 1: statistics
            for (auto& fl : files) for (auto& it : fl) if (it.learn)
            { featNorm.update (it.x); noiseNorm.update (it.noise); if (it.hasSpat) spatNorm.update (it.spat); }
            // pass 2: frames, split by block
            int block = 0;
            for (auto& fl : files)
            {
                std::vector<Frame> run;   // consecutive frames of this block (for sequences)
                for (size_t i = 0; i < fl.size(); ++i)
                {
                    if (i > 0 && i % 200 == 0) { if (block % 10 != 9) addSeqs (run, trainSeq); run.clear(); ++block; }
                    if (! fl[i].learn) { run.clear(); continue; }
                    const Frame f = makeFrame (fl[i]);
                    (block % 10 == 9 ? val : train).push_back (f);
                    run.push_back (f);
                }
                if (block % 10 != 9) addSeqs (run, trainSeq);
                ++block;
            }
            // too little to hold out a block: take the last tenth
            if (val.empty() && train.size() >= 50) { const size_t k = train.size() / 10; val.assign (train.end() - (long) k, train.end()); train.resize (train.size() - k); }
        }
        offline.trainFrames = (int) train.size(); offline.valFrames = (int) val.size();
        int ns = 0; for (auto& f : train) ns += f.hasSpat > 0; offline.spatFrames = ns;
        if (train.size() < 64 || val.empty()) { offline.state = 5; return; }
        { std::lock_guard<std::mutex> ll (lab.m); lab.line ("== split by 2 s blocks (every 10th held out): %d training frames, %d held-out frames, %d stereo, %d GRU sequences of 25 frames",
                                                            (int) train.size(), (int) val.size(), offline.spatFrames.load(), (int) trainSeq.size()); }

        std::vector<float> bestV, bestG, bestS; double best = 1e30; int bestEpoch = 0;
        std::vector<size_t> order (train.size()); std::iota (order.begin(), order.end(), 0);
        std::vector<size_t> spatIdx; for (size_t i = 0; i < train.size(); ++i) if (train[i].hasSpat > 0) spatIdx.push_back (i);
        const int batches = (int) (train.size() / 32);
        std::mt19937 orng (0x0ff1); // its own generator: the shared one belongs to the trainer thread (under modelLock)
        for (int ep = 0; ep < maxEpochs && ! offline.cancel; ++ep)
        {
            std::shuffle (order.begin(), order.end(), orng);
            const float lr = std::max (1e-4f, 1e-3f * std::pow (0.9f, (float) ep)) * std::clamp (settings.plasticity.load(), 0.25f, 4.0f);
            double sum = 0;
            for (int b = 0; b < batches && ! offline.cancel; ++b)
            {
                std::lock_guard<std::mutex> lk (modelLock);
                std::vector<const Frame*> batch; for (int k = 0; k < 32; ++k) batch.push_back (&train[order[(size_t) (b * 32 + k)]]);
                vae.beta = std::min (settings.betaMax.load(), settings.betaMax.load() * (float) vaeSteps / 3000.0f);
                optV.lr = lr; vae.ps.zeroGrad();
                const bool deep = (vaeSteps + 1) % 10 == 0;
                if (deep) vae.ps.copyWeightsTo (labPrev);
                const auto t0 = std::chrono::steady_clock::now();
                const auto L = vae.trainBatch (batch, rng);
                const double gn = optV.step (vae.ps);
                const double us = std::chrono::duration<double, std::micro> (std::chrono::steady_clock::now() - t0).count();
                if (! vae.ps.finite() || ! std::isfinite (L.total)) { labRestore(); if (! good.empty()) vae.ps.setWeightsFrom (good); optV.reset (vae.ps); continue; }
                ++vaeSteps; stats.vaeSteps = vaeSteps; sum += L.recon;
                labObserve (L, gn, us, batch, 0, deep, true);
                if (vaeSteps % 25 == 0) labExample (val[(size_t) (rng() % val.size())]);
                stats.recon = (float) (stats.recon * 0.98 + 0.02 * L.recon); stats.kl = (float) (stats.kl * 0.98 + 0.02 * L.kl); stats.beta = vae.beta; stats.lr = lr;
                if (vaeSteps % 500 == 0) vae.ps.copyWeightsTo (good);
                if (b % 2 == 0 && spatIdx.size() >= 32)
                {
                    Vec zs, cs, ts;
                    for (int k = 0; k < 32; ++k)
                    {
                        const Frame& f = train[spatIdx[(size_t) (rng() % spatIdx.size())]];
                        float mu[kZ]; vae.encode (f.x.data(), f.cond.data(), mu);
                        zs.insert (zs.end(), mu, mu + kZ); cs.insert (cs.end(), f.cond.begin(), f.cond.end()); ts.insert (ts.end(), f.spat.begin(), f.spat.end());
                    }
                    optS.lr = lr; spatial.ps.zeroGrad(); const double nll = spatial.trainBatch (zs, cs, ts, 32); optS.step (spatial.ps);
                    if (spatial.ps.finite()) { ++spatSteps; stats.spatSteps = spatSteps; stats.spatNll = (float) (stats.spatNll * 0.97 + 0.03 * nll); }
                    else { spatial.init (91); optS.reset (spatial.ps); }
                }
                if (b % 4 == 0 && trainSeq.size() >= 8) trainGruOn (trainSeq, 0.5f * lr);
                offline.progress = ((float) ep + (float) (b + 1) / (float) batches) / (float) maxEpochs;
            }
            // validation
            double tr = 0, va = 0, sv = 0; int sn = 0;
            {
                std::lock_guard<std::mutex> lk (modelLock);
                const size_t nv = std::min<size_t> (val.size(), 4000), nt = std::min<size_t> (train.size(), 2000);
                for (size_t i = 0; i < nv; ++i) { const Frame& f = val[i * val.size() / nv]; va += evalRecon (f); if (f.hasSpat > 0 && spatSteps > 0) { sv += evalSpat (f); ++sn; } }
                for (size_t i = 0; i < nt; ++i) tr += evalRecon (train[i * train.size() / nt]);
                va /= (double) nv; tr /= (double) nt;
                if (va < best - 1e-5) { best = va; bestEpoch = ep + 1; vae.ps.copyWeightsTo (bestV); gru.ps.copyWeightsTo (bestG); spatial.ps.copyWeightsTo (bestS); }
            }
            (void) sum;
            offline.epoch = ep + 1; offline.trainLoss = (float) tr; offline.valLoss = (float) va; offline.bestVal = (float) best; offline.bestEpoch = bestEpoch;
            if (sn > 0) offline.spatVal = (float) (sv / sn);
            offline.trainCurve[ep % 256] = (float) tr; offline.valCurve[ep % 256] = (float) va;
            {
                std::lock_guard<std::mutex> ll (lab.m);
                lab.line ("== epoch %d/%d  %d batches x 32  lr %.2e  |  eval-mode mse  train %.5f  held-out %.5f  gap %+.5f  |  best %.5f @ epoch %d%s",
                          ep + 1, maxEpochs, batches, (double) lr, tr, va, va - tr, best, bestEpoch, bestEpoch == ep + 1 ? "  (new best: weights kept)" : "");
                if (ep + 1 - bestEpoch >= patience) lab.line ("== early stop: %d epochs without improvement on the held-out block; restoring the weights of epoch %d", patience, bestEpoch);
            }
            publish (true);
            if (ep + 1 - bestEpoch >= patience) break;   // early stopping
        }
        {
            std::lock_guard<std::mutex> lk (modelLock);
            if (! bestV.empty()) { vae.ps.setWeightsFrom (bestV); gru.ps.setWeightsFrom (bestG); spatial.ps.setWeightsFrom (bestS); }
            optV.reset (vae.ps); optG.reset (gru.ps); optS.reset (spatial.ps);
            vae.ps.copyWeightsTo (good);
            // the constellation, refitted on the training latents
            for (int pass = 0; pass < 3; ++pass)
                for (size_t i = 0; i < train.size(); i += 2) { float mu[kZ]; const Frame& f = train[(i * 7919 + (size_t) pass) % train.size()]; vae.encode (f.x.data(), f.cond.data(), mu); som.learn (mu); }
            // live learning continues from here: the data joins the replay and held-out memories
            for (auto& f : train) replay.add (f);
            for (auto& f : val) held.add (f);
            for (auto& c : trainSeq) seqs.add (c);
            stats.holdout = (int) held.items.size(); stats.replaySize = (int) replay.items.size();
            frames += train.size() + val.size(); stats.framesLearned = frames;
        }
        publish (true); syncNow = true; wake();
        offline.state = offline.cancel ? 4 : 3;
    }

private:
    Vec labPrev;   // the VAE's weights before a "deep" step (for the per-layer update ratio)
    std::chrono::steady_clock::time_point labLast = std::chrono::steady_clock::now();
    size_t vaeParamCount() { size_t n = 0; for (auto* q : vae.ps.list) n += q->size(); return n; }
    size_t gruParamCount() { size_t n = 0; for (auto* q : gru.ps.list) n += q->size(); return n; }

    static void layerStat (LayerStat& L, const Param& W, const Param& b, const float* prev, const Vec* act)
    {
        double ww = 0, gg = 0, dd = 0; float mx = 1e-9f;
        for (size_t i = 0; i < W.size(); ++i)
        {
            ww += (double) W.w[i] * W.w[i]; gg += (double) W.g[i] * W.g[i]; mx = std::max (mx, std::abs (W.w[i]));
            if (prev) { const double d = W.w[i] - prev[i]; dd += d * d; }
        }
        const double n = (double) std::max<size_t> (1, W.size());
        L.wRms = (float) std::sqrt (ww / n); L.gRms = (float) std::sqrt (gg / n); L.wMax = mx;
        L.upd = prev ? (float) std::sqrt (dd / std::max (1e-30, ww)) : L.upd;
        L.params = W.size() + b.size();
        L.wHist.fill (0); L.gHist.fill (0);
        for (size_t i = 0; i < W.size(); ++i)
        {
            const int k = std::clamp ((int) std::lround ((W.w[i] / mx * 0.5f + 0.5f) * 32), 0, 32); L.wHist[(size_t) k] += 1;
            const float lg = std::log10 (std::abs (W.g[i]) + 1e-12f);
            const int q = std::clamp ((int) std::lround ((lg + 9.0f) / 8.0f * 32), 0, 32); L.gHist[(size_t) q] += 1;
        }
        const float hm = *std::max_element (L.wHist.begin(), L.wHist.end()), gm = *std::max_element (L.gHist.begin(), L.gHist.end());
        for (auto& v : L.wHist) v /= std::max (1.0f, hm);
        for (auto& v : L.gHist) v /= std::max (1.0f, gm);
        if (act && ! act->empty())
        {
            double s = 0, s2 = 0; size_t neg = 0;
            for (float y : *act) { s += y; s2 += (double) y * y; neg += y < 0; }
            const double m = s / (double) act->size();
            L.actMean = (float) m; L.actStd = (float) std::sqrt (std::max (0.0, s2 / (double) act->size() - m * m)); L.negFrac = (float) neg / (float) act->size(); L.hasAct = true;
        }
    }

    /** after each VAE step: the losses, the optimiser, the latent per dimension, the error per output; every 10th
        step also the layers, the weight matrices and the batch's activations */
    void labObserve (const VAE::Losses& L, double gn, double us, const std::vector<const Frame*>& batch, int nRecent, bool deep, bool offlineMode)
    {
        std::lock_guard<std::mutex> lk (lab.m);
        const int B = (int) batch.size();
        Lab::Step s; s.total = (float) L.total; s.recon = (float) L.recon; s.harm = (float) L.harm; s.noise = (float) L.noise; s.kl = (float) L.kl;
        s.beta = vae.beta; s.lr = optV.lr; s.gnorm = (float) gn; s.clipK = gn > optV.clip ? (float) (optV.clip / gn) : 1.0f; s.us = (float) us;
        lab.series[(size_t) (lab.pos % Lab::N)] = s; lab.pos = (lab.pos + 1) % Lab::N; ++lab.count; lab.last = s;
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::max (1e-6, std::chrono::duration<double> (now - labLast).count()); labLast = now;
        lab.stepsPerSec = (float) (0.95 * lab.stepsPerSec + 0.05 * std::min (20000.0, 1.0 / dt));
        int voiced = 0; for (auto* f : batch) voiced += f->voiced > 0.5f;
        lab.batchRecent = nRecent; lab.batchReplay = B - nRecent; lab.batchVoiced = voiced; lab.offlineMode = offlineMode;
        lab.adamT = optV.t; lab.adamClip = optV.clip; lab.adamWd = optV.wd;
        // the error per output, from this batch's forward pass
        const float* o = vae.dout.Y.data();
        if (vae.dout.Y.size() >= (size_t) B * kOut)
        {
            for (int i = 0; i < kFeat; ++i)
            { double e = 0; for (int r = 0; r < B; ++r) { const double d = o[(size_t) r * kOut + i] - batch[(size_t) r]->x[(size_t) i]; e += d * d; } lab.featErr[(size_t) i] = (float) (0.97 * lab.featErr[(size_t) i] + 0.03 * e / B); }
            for (int i = 0; i < kNoise; ++i)
            { double e = 0; for (int r = 0; r < B; ++r) { const double d = o[(size_t) r * kOut + kFeat + kHarm + i] - batch[(size_t) r]->noise[(size_t) i]; e += d * d; } lab.noiseErr[(size_t) i] = (float) (0.97 * lab.noiseErr[(size_t) i] + 0.03 * e / B); }
            std::array<double, kHarm> he {}; double vw = 0;
            for (int r = 0; r < B; ++r)
            {
                const float* h = o + (size_t) r * kOut + kFeat; float mx = -1e9f; for (int i = 0; i < kHarm; ++i) mx = std::max (mx, h[i]);
                double z = 0; float p[kHarm]; for (int i = 0; i < kHarm; ++i) { p[i] = std::exp (h[i] - mx); z += p[i]; }
                const float v = batch[(size_t) r]->voiced; vw += v;
                for (int i = 0; i < kHarm; ++i) he[(size_t) i] += v * std::abs (p[i] / z - batch[(size_t) r]->harm[(size_t) i]);
            }
            if (vw > 0.5) for (int i = 0; i < kHarm; ++i) lab.harmErr[(size_t) i] = (float) (0.97 * lab.harmErr[(size_t) i] + 0.03 * he[(size_t) i] / vw);
        }
        // the posterior per latent dimension
        if (vae.mu.size() >= (size_t) B * kZ)
            for (int j = 0; j < kZ; ++j)
            {
                double kl = 0, m = 0, m2 = 0, sg = 0;
                for (int r = 0; r < B; ++r)
                {
                    const float mu = vae.mu[(size_t) r * kZ + j], lv = vae.lv[(size_t) r * kZ + j];
                    kl += -0.5 * (1 + lv - mu * mu - std::exp (lv)); m += mu; m2 += (double) mu * mu; sg += std::exp (0.5 * lv);
                }
                m /= B;
                lab.klDim[(size_t) j] = (float) (0.95 * lab.klDim[(size_t) j] + 0.05 * kl / B);
                lab.muMean[(size_t) j] = (float) (0.95 * lab.muMean[(size_t) j] + 0.05 * m);
                lab.muStd[(size_t) j] = (float) (0.95 * lab.muStd[(size_t) j] + 0.05 * std::sqrt (std::max (0.0, m2 / B - m * m)));
                lab.sigMean[(size_t) j] = (float) (0.95 * lab.sigMean[(size_t) j] + 0.05 * sg / B);
            }
        lab.line ("%s#%06llu  B%d %dr/%dp %dv  L %.5f = rec %.5f + .25*harm %.4f + noise %.5f + %.3f*kl %.4f  |g| %.3f%s  lr %.2e  t%ld  %.0fus",
                  offlineMode ? "off " : "vae ", (unsigned long long) vaeSteps, B, nRecent, B - nRecent, voiced, L.total, L.recon, L.harm, L.noise, (double) vae.beta, L.kl,
                  gn, gn > optV.clip ? " (clipped)" : "", (double) optV.lr, optV.t, us);
        if (! deep) return;
        // ---- the layers
        struct Ly { Dense* d; const char* name; const char* act; };
        const Ly ly[] = { { &vae.e1, "enc.1", "lrelu" }, { &vae.fe.map, "film.enc", "affine" }, { &vae.e2, "enc.2", "lrelu" }, { &vae.eh, "enc.head", "mu|logvar" },
                          { &vae.d1, "dec.1", "lrelu" }, { &vae.fd.map, "film.dec", "affine" }, { &vae.d2, "dec.2", "lrelu" }, { &vae.dout, "dec.out", "x|harm|noise" } };
        if (lab.layers.size() < 11) lab.layers.resize (11);
        size_t off = 0;
        for (int k = 0; k < 8; ++k)
        {
            auto& st = lab.layers[(size_t) k]; st.name = ly[k].name; st.act = ly[k].act; st.in = ly[k].d->in; st.out = ly[k].d->out;
            const float* prev = labPrev.size() >= off + ly[k].d->W.size() ? labPrev.data() + off : nullptr;
            layerStat (st, ly[k].d->W, ly[k].d->b, prev, &ly[k].d->Y);
            off += ly[k].d->W.size() + ly[k].d->b.size();
        }
        lab.layersAt = vaeSteps;
        auto mat = [] (Lab::Mat& M, const char* name, const Param& W, int rows, int cols)
        {
            M.name = name; M.rows = rows; M.cols = cols; M.v.assign (W.w.begin(), W.w.end());
            double ss = 0; for (float x : M.v) ss += (double) x * x; M.scale = (float) std::sqrt (ss / std::max<size_t> (1, M.v.size()));
        };
        mat (lab.W[0], "enc.1  W  128 x 84", vae.e1.W, 128, kFeat); mat (lab.W[1], "dec.2  W  128 x 128", vae.d2.W, 128, 128); mat (lab.W[2], "dec.out  W  164 x 128", vae.dout.W, kOut, 128);
        auto actm = [B] (Lab::Mat& M, const char* name, const Vec& Y, int units, bool expHalf = false)
        {
            M.name = name; M.rows = B; M.cols = units; M.v.assign (Y.begin(), Y.begin() + (long) std::min (Y.size(), (size_t) B * units));
            if (expHalf) for (auto& x : M.v) x = std::exp (0.5f * x);
            double s = 0, s2 = 0; for (float x : M.v) { s += x; s2 += (double) x * x; }
            const double m = s / std::max<size_t> (1, M.v.size()); M.scale = (float) std::max (1e-6, std::sqrt (std::max (0.0, s2 / std::max<size_t> (1, M.v.size()) - m * m)));
        };
        actm (lab.A[0], "enc.1", vae.e1.Y, 128); actm (lab.A[1], "enc.2", vae.e2.Y, 64); actm (lab.A[2], "mu", vae.mu, kZ); actm (lab.A[3], "sigma", vae.lv, kZ, true);
        actm (lab.A[4], "dec.1", vae.d1.Y, 128); actm (lab.A[5], "dec.2", vae.d2.Y, 128);
        double gsum = 0; for (int k = 0; k < 8; ++k) gsum += lab.layers[(size_t) k].upd;
        lab.line ("deep #%06llu  per-layer |dW|/|W|: enc.1 %.1e  enc.2 %.1e  head %.1e  dec.1 %.1e  dec.2 %.1e  out %.1e  |  lrelu negative: enc.1 %.0f%%  dec.2 %.0f%%",
                  (unsigned long long) vaeSteps, (double) lab.layers[0].upd, (double) lab.layers[2].upd, (double) lab.layers[3].upd, (double) lab.layers[4].upd, (double) lab.layers[6].upd,
                  (double) lab.layers[7].upd, 100.0 * lab.layers[0].negFrac, 100.0 * lab.layers[6].negFrac);
        (void) gsum;
    }
    void labGru (double nll, double gn)
    {
        std::lock_guard<std::mutex> lk (lab.m);
        lab.gruNll = (float) nll; lab.gruGnorm = (float) gn; ++lab.gruStepsSeen;
        lab.line ("gru #%06llu  bptt T=25 B=8  in z|c %d -> h %d -> mu|logvar %d  gaussian nll %.4f / dim  |g| %.3f  lr %.2e",
                  (unsigned long long) gruSteps, kZ + kCond, kGruH, 2 * kZ, nll, gn, (double) optG.lr);
        if (lab.layers.size() < 11) lab.layers.resize (11);
        if (gruSteps % 5 == 0)
        {
            auto& a = lab.layers[8]; a.name = "gru.Wx"; a.act = "gates z,r,n"; a.in = kZ + kCond; a.out = 3 * kGruH; layerStat (a, gru.gru.Wx, gru.gru.bx, nullptr, nullptr);
            auto& b = lab.layers[9]; b.name = "gru.Uh"; b.act = "recurrent"; b.in = kGruH; b.out = 3 * kGruH; layerStat (b, gru.gru.Uh, gru.gru.bh, nullptr, gru.gru.hs.empty() ? nullptr : &gru.gru.hs.back());
            auto& c = lab.layers[10]; c.name = "gru.head"; c.act = "mu|logvar"; c.in = kGruH; c.out = 2 * kZ; layerStat (c, gru.head.W, gru.head.b, nullptr, &gru.head.Y);
        }
    }
    /** a held-out frame through the network in eval mode (the latent mean) */
    void labExample (const Frame& f)
    {
        float mu[kZ], lv[kZ], out[kOut];
        vae.encode (f.x.data(), f.cond.data(), mu, lv); vae.decode (mu, f.cond.data(), out);
        std::lock_guard<std::mutex> lk (lab.m);
        double e = 0;
        for (int i = 0; i < kFeat; ++i) { lab.exX[(size_t) i] = f.x[(size_t) i]; lab.exXh[(size_t) i] = out[i]; const double d = out[i] - f.x[(size_t) i]; e += d * d; }
        float mx = -1e9f; for (int i = 0; i < kHarm; ++i) mx = std::max (mx, out[kFeat + i]);
        double z = 0; for (int i = 0; i < kHarm; ++i) { lab.exHp[(size_t) i] = std::exp (out[kFeat + i] - mx); z += lab.exHp[(size_t) i]; }
        for (int i = 0; i < kHarm; ++i) { lab.exHp[(size_t) i] = (float) (lab.exHp[(size_t) i] / z); lab.exHt[(size_t) i] = f.harm[(size_t) i]; }
        for (int i = 0; i < kNoise; ++i) { lab.exNt[(size_t) i] = f.noise[(size_t) i]; lab.exNp[(size_t) i] = out[kFeat + kHarm + i]; }
        for (int j = 0; j < kZ; ++j) { lab.exMu[(size_t) j] = mu[j]; lab.exSig[(size_t) j] = std::exp (0.5f * lv[j]); }
        lab.exErr = (float) (e / kFeat); lab.exVoiced = f.voiced; lab.exValid = true; lab.exAt = vaeSteps;
        lab.line ("held #%06llu  a held-out frame (never trained on) through enc -> mu -> dec:  feature mse %.5f  voiced %.2f  |mu| %.2f",
                  (unsigned long long) vaeSteps, e / kFeat, (double) f.voiced, [&] { double a = 0; for (float x : mu) a += (double) x * x; return std::sqrt (a); }());
    }
    void labRestore()
    {
        std::lock_guard<std::mutex> lk (lab.m); ++lab.restores;
        lab.line ("!!   non-finite loss or weights at step %llu: weights restored from the last good copy, Adam moments reset", (unsigned long long) vaeSteps);
    }

    struct SeqChunk { std::array<std::array<float, kFeat>, 25> x; std::array<float, kCond> cond; };

    /** Eval-mode losses (latent mean, no sampling, unweighted feature MSE) on training vs held-out frames; the
        active units of the latent; the spatial network's held-out NLL and the coverage of its ±1σ interval. */
    void evaluate()
    {
        if (held.items.size() < 32 || replay.items.size() < 32) return;
        auto recon = [&] (const Frame& f)
        {
            float mu[kZ], out[kOut]; vae.encode (f.x.data(), f.cond.data(), mu); vae.decode (mu, f.cond.data(), out);
            double e = 0; for (int i = 0; i < kFeat; ++i) { const double d = out[i] - f.x[(size_t) i]; e += d * d; }
            return e / kFeat;
        };
        const size_t n = std::min<size_t> (128, held.items.size());
        double tr = 0, va = 0;
        double m1[kZ] {}, m2[kZ] {};
        double snll = 0, cover = 0; int sn = 0;
        for (size_t i = 0; i < n; ++i)
        {
            tr += recon (replay.sample());
            const Frame& h = held.items[(i * 7919) % held.items.size()];
            va += recon (h);
            float mu[kZ]; vae.encode (h.x.data(), h.cond.data(), mu);
            for (int j = 0; j < kZ; ++j) { m1[j] += mu[j]; m2[j] += (double) mu[j] * mu[j]; }
            if (h.hasSpat > 0 && spatSteps > 0)
            {
                float m[kSpatial], lv[kSpatial]; spatial.infer (mu, h.cond.data(), m, lv);
                for (int j = 0; j < kSpatial; ++j)
                {
                    const double e = h.spat[(size_t) j] - m[j], v = std::exp ((double) lv[j]);
                    snll += 0.5 * (std::log (2 * M_PI * v) + e * e / v); cover += std::abs (e) <= std::sqrt (v) ? 1 : 0;
                }
                ++sn;
            }
        }
        stats.trainRecon = (float) (tr / n); stats.valRecon = (float) (va / n);
        int active = 0;
        for (int j = 0; j < kZ; ++j) { const double v = m2[j] / n - (m1[j] / n) * (m1[j] / n); stats.unitVar[(size_t) j] = (float) v; active += v > 0.01 ? 1 : 0; }
        stats.activeUnits = active;
        if (sn > 0) { stats.spatValNll = (float) (snll / (sn * kSpatial)); stats.spatCoverage = (float) (cover / (sn * kSpatial)); }
        {
            std::lock_guard<std::mutex> ll (lab.m);
            lab.line ("eval @%llu  eval-mode (latent mean) mse over %d frames:  train %.5f  held-out %.5f  gap %+.5f  |  active units %d/%d (Cov_x E_q[z] > 0.01)%s",
                      (unsigned long long) vaeSteps, (int) n, tr / n, va / n, (va - tr) / n, active, kZ,
                      sn > 0 ? "" : "");
            if (sn > 0) lab.line ("eval @%llu  spatial net: held-out gaussian nll %.3f / dim, ±1σ coverage %.0f%% (68%% if calibrated)", (unsigned long long) vaeSteps, snll / (sn * kSpatial), 100.0 * cover / (sn * kSpatial));
        }
        const int ep = stats.evalPos.load(); stats.evTrainHist[ep % 128] = stats.trainRecon; stats.evValHist[ep % 128] = stats.valRecon; stats.evalPos = ep + 1;
    }

    void initModels (uint32_t seed)
    {
        vae.init (seed); gru.init (seed + 1); esn.init (seed + 2); som.init (seed + 3); spatial.init (seed + 4);
        featNorm.init (kFeat); noiseNorm.init (kNoise); spatNorm.init (kSpatial);
        optV = {}; optG = {}; optG.lr = 5e-4f; optS = {}; optS.lr = 1e-3f; spatSteps = 0;
        replay.init (20000); seqs.init (400); held.init (512); heldCount = 0;
        vaeSteps = gruSteps = frames = 0;
        esnState.assign (EchoState::N, 0.0f); std::fill (prevMu, prevMu + kZ, 0.0f);
        {
            std::lock_guard<std::mutex> l (wordLock); nodeWords.clear();
        }
        good.clear();
        baseValid = false;
    }

    void run()
    {
        auto lastPublish = std::chrono::steady_clock::now(), lastSync = lastPublish;
        while (running)
        {
            const bool worked = iterate (false);
            const auto now = std::chrono::steady_clock::now();
            if (publishNow || now - lastPublish > std::chrono::milliseconds (100)) { publish (false); lastPublish = now; publishNow = false; }
            if (syncNow || now - lastSync > std::chrono::seconds (30)) { sync(); lastSync = now; syncNow = false; }
            if (! worked)
            {
                std::unique_lock<std::mutex> l (sleepLock);
                cv.wait_for (l, std::chrono::milliseconds (settings.intensive ? 1 : 8));
            }
        }
    }

    /** one round: take new frames, learn from them, take gradient steps. Returns true if it did work. */
    bool iterate (bool test)
    {
        std::lock_guard<std::mutex> lk (modelLock);
        int got = 0;
        {
            std::lock_guard<std::mutex> l (feedLock);
            FeedItem it;
            for (auto& r : feeders)
                for (int n = 0; n < 256 && r->pop (it); ++n) { take (it); ++got; }
        }
        const bool frozen = settings.frozen.load() || settings.plasticity.load() <= 0.0f;
        if (frozen || replay.items.size() < 64) return got > 0;

        // how many gradient steps are owed: ~1 per 2 new frames live, continuous when files are fed, a slow trickle
        // of consolidation from replay when idle ("sleep")
        owed += got * 0.5;
        const bool idle = got == 0;
        stats.sleeping = idle && ! settings.intensive;
        if (settings.intensive) owed = std::max (owed, 4.0);
        else if (idle) owed += 0.12 * settings.sleepRate.load();
        if (test) owed = std::max (owed, 1.0);
        int steps = 0;
        while (owed >= 1.0 && steps < 8) { trainStep(); owed -= 1.0; ++steps; }
        owedNow = settings.intensive ? 0.0 : owed;
        return got > 0 || steps > 0;
    }

    /** a standardised training frame from a feed item (statistics as they are now) */
    Frame makeFrame (const FeedItem& it) const
    {
        Frame f;
        featNorm.apply (it.x, f.x.data());
        std::copy (it.harm, it.harm + kHarm, f.harm.begin());
        noiseNorm.apply (it.noise, f.noise.data());
        std::copy (it.cond, it.cond + kCond, f.cond.begin());
        f.voiced = it.voiced;
        if (it.hasSpat) { spatNorm.apply (it.spat, f.spat.data()); for (auto& v : f.spat) v = std::clamp (v, -6.0f, 6.0f); f.hasSpat = 1; }
        for (auto& v : f.x) v = std::clamp (v, -6.0f, 6.0f);
        for (auto& v : f.noise) v = std::clamp (v, -6.0f, 6.0f);
        return f;
    }
    /** 25-frame windows (stride 12) of a run of consecutive frames, for the GRU */
    static void addSeqs (const std::vector<Frame>& run, std::vector<SeqChunk>& out)
    {
        for (size_t s = 0; s + 25 <= run.size(); s += 12)
        { SeqChunk c; for (int t = 0; t < 25; ++t) c.x[(size_t) t] = run[s + (size_t) t].x; c.cond = run[s].cond; out.push_back (c); }
    }
    void trainGruOn (const std::vector<SeqChunk>& pool, float lr)
    {
        std::vector<std::vector<std::array<float, kZ>>> seqZ (8, std::vector<std::array<float, kZ>> (25));
        std::vector<std::array<float, kCond>> cond (8);
        for (int b = 0; b < 8; ++b)
        {
            const auto& c = pool[(size_t) (rng() % pool.size())];
            cond[(size_t) b] = c.cond;
            for (int t = 0; t < 25; ++t) vae.encode (c.x[(size_t) t].data(), c.cond.data(), seqZ[(size_t) b][(size_t) t].data());
        }
        optG.lr = lr; gru.ps.zeroGrad();
        const double nll = gru.trainBatch (seqZ, cond);
        const double gn = optG.step (gru.ps);
        if (gru.ps.finite()) { ++gruSteps; stats.gruSteps = gruSteps; stats.gruNll = (float) (stats.gruNll * 0.95 + 0.05 * nll); labGru (nll, gn); }
        else { gru.init (77); optG.reset (gru.ps); }
    }
    /** eval mode: latent mean, no sampling, unweighted feature MSE */
    double evalRecon (const Frame& f) const
    {
        float mu[kZ], out[kOut]; vae.encode (f.x.data(), f.cond.data(), mu); vae.decode (mu, f.cond.data(), out);
        double e = 0; for (int i = 0; i < kFeat; ++i) { const double d = out[i] - f.x[(size_t) i]; e += d * d; }
        return e / kFeat;
    }
    /** Gaussian NLL per dimension of the spatial features (standardised units) */
    double evalSpat (const Frame& f) const
    {
        float mu[kZ], m[kSpatial], lv[kSpatial]; vae.encode (f.x.data(), f.cond.data(), mu); spatial.infer (mu, f.cond.data(), m, lv);
        double s = 0; for (int j = 0; j < kSpatial; ++j) { const double e = f.spat[(size_t) j] - m[j], v = std::exp ((double) lv[j]); s += 0.5 * (std::log (2 * M_PI * v) + e * e / v); }
        return s / kSpatial;
    }

    void take (const FeedItem& it)
    {
        stats.framesSeen++;
        if (! it.learn || settings.frozen) return;
        featNorm.update (it.x); noiseNorm.update (it.noise);
        Frame f;
        featNorm.apply (it.x, f.x.data());
        std::copy (it.harm, it.harm + kHarm, f.harm.begin());
        noiseNorm.apply (it.noise, f.noise.data());
        std::copy (it.cond, it.cond + kCond, f.cond.begin());
        f.voiced = it.voiced;
        if (it.hasSpat)
        {
            spatNorm.update (it.spat); spatNorm.apply (it.spat, f.spat.data());
            for (auto& v : f.spat) v = std::clamp (v, -6.0f, 6.0f);
            f.hasSpat = 1; stats.spatFrames++;
        }
        for (auto& v : f.x) v = std::clamp (v, -6.0f, 6.0f);
        for (auto& v : f.noise) v = std::clamp (v, -6.0f, 6.0f);
        // every 10th frame goes to the held-out set instead of the training sets (the live latent stream below still
        // sees it: the SOM and the reservoir are unsupervised and online)
        if (++heldCount % 10 == 0) { held.add (f); stats.holdout = (int) held.items.size(); }
        else
        {
            replay.add (f);
            recent[(size_t) (recentPos++ % recent.size())] = f;
        }
        // sequences for the GRU: 25 consecutive frames
        chunk.x[(size_t) chunkFill] = f.x; chunk.cond = f.cond;
        if (++chunkFill == 25) { seqs.add (chunk); chunkFill = 0; }
        // the live latent stream: SOM, ESN, star words
        float mu[kZ]; vae.encode (f.x.data(), f.cond.data(), mu);
        const int b = som.learn (mu);
        stats.lastBmu = b;
        float in[EchoState::I]; std::copy (prevMu, prevMu + kZ, in); std::copy (f.cond.begin(), f.cond.end(), in + kZ);
        esn.advance (esnState.data(), in);
        float pred[kZ]; esn.readout (esnState.data(), pred);
        double e = 0; for (int j = 0; j < kZ; ++j) e += (pred[j] - mu[j]) * (pred[j] - mu[j]);
        stats.esnErr = (float) (0.99 * stats.esnErr + 0.01 * e / kZ);
        stats.surprise = (float) std::min (1.0, std::sqrt (e / kZ) / 1.5);
        esn.learn (esnState.data(), mu);
        std::copy (mu, mu + kZ, prevMu);
        {
            std::lock_guard<std::mutex> l (wordLock);
            for (auto& w : currentWords)
            {
                auto& m = nodeWords[b];
                m[w] += 1.0f;
                if (m.size() > 12)   // keep the strongest
                {
                    auto weakest = std::min_element (m.begin(), m.end(), [] (auto& a, auto& c) { return a.second < c.second; });
                    m.erase (weakest);
                }
            }
        }
        ++frames;
        stats.framesLearned = frames;
        if (frames % 100 == 0)
        {
            std::lock_guard<std::mutex> ll (lab.m);
            lab.line ("in   +100 frames (%llu learned)  replay %d / held-out %d  |  som bmu %03x σ %.2f η %.3f  |  esn rls err %.4f  surprise %.0f%%  |  f0 feat %+.2f  loud %+.2f (standardised)",
                      (unsigned long long) frames, (int) replay.items.size(), (int) held.items.size(), b, som.sigma(), som.eta(), (double) stats.esnErr.load(), 100.0 * stats.surprise.load(),
                      (double) f.x[(size_t) kF0], (double) f.x[(size_t) kLoud]);
        }
        stats.esnUpdates = esn.updates;
        stats.replaySize = (int) replay.items.size();
    }

    void trainStep()
    {
        const float p = std::clamp (settings.plasticity.load(), 0.0f, 4.0f);
        // learning rate: warm start 1e-3, decaying to a floor of 2e-4 (never zero: plasticity), scaled by the user
        const float base = std::max (2e-4f, 1e-3f * std::exp (-(float) vaeSteps / 60000.0f));
        optV.lr = base * p; optG.lr = 0.5f * base * p;
        vae.beta = std::min (settings.betaMax.load(), settings.betaMax.load() * (float) vaeSteps / 3000.0f);
        // batch: half recent, half replay
        std::vector<const Frame*> batch;
        const size_t have = std::min<size_t> (recentPos, recent.size());
        for (int i = 0; i < 32; ++i)
            batch.push_back (i < 16 && have > 0 ? &recent[(size_t) (rng() % have)] : &replay.sample());
        const bool deep = (vaeSteps + 1) % 10 == 0;
        if (deep) vae.ps.copyWeightsTo (labPrev);
        vae.ps.zeroGrad();
        const auto t0 = std::chrono::steady_clock::now();
        const auto L = vae.trainBatch (batch, rng);
        const double gnV = optV.step (vae.ps);
        const double usV = std::chrono::duration<double, std::micro> (std::chrono::steady_clock::now() - t0).count();
        if (! vae.ps.finite() || ! std::isfinite (L.total))
        {
            labRestore();
            if (! good.empty()) vae.ps.setWeightsFrom (good);
            optV.reset (vae.ps);
            return;
        }
        ++vaeSteps;
        stats.vaeSteps = vaeSteps;
        labObserve (L, gnV, usV, batch, have > 0 ? 16 : 0, deep, false);
        if (vaeSteps % 25 == 0 && (held.items.size() > 0 || replay.items.size() > 0)) labExample (held.items.empty() ? replay.sample() : held.sample());
        stats.recon = (float) (stats.recon * 0.98 + 0.02 * L.recon);
        stats.kl = (float) (stats.kl * 0.98 + 0.02 * L.kl);
        stats.harm = (float) (stats.harm * 0.98 + 0.02 * L.harm);
        stats.beta = vae.beta; stats.lr = optV.lr;
        if (vaeSteps % 500 == 0) vae.ps.copyWeightsTo (good);
        if (vaeSteps % 200 == 0) evaluate();
        if (vaeSteps % 40 == 0)
        {
            const int hp = stats.histPos.load();
            stats.reconHist[hp % Stats::kHist] = stats.recon; stats.gruHist[hp % Stats::kHist] = stats.gruNll;
            stats.histPos = hp + 1;
        }
        // spatial network on replayed stereo frames, every 2nd step (latents from the current encoder, detached)
        if (vaeSteps % 2 == 0 && stats.spatFrames.load() > 64)
        {
            Vec zs, cs, ts; int B = 0;
            for (int tries = 0; tries < 160 && B < 32; ++tries)
            {
                const Frame& f = (tries % 2 == 0 && have > 0) ? recent[(size_t) (rng() % have)] : replay.sample();
                if (f.hasSpat <= 0) continue;
                float mu[kZ]; vae.encode (f.x.data(), f.cond.data(), mu);
                zs.insert (zs.end(), mu, mu + kZ); cs.insert (cs.end(), f.cond.begin(), f.cond.end()); ts.insert (ts.end(), f.spat.begin(), f.spat.end());
                ++B;
            }
            if (B >= 8)
            {
                optS.lr = 1e-3f * p;
                spatial.ps.zeroGrad();
                const double nll = spatial.trainBatch (zs, cs, ts, B);
                const double gnS = optS.step (spatial.ps);
                if (spatial.ps.finite())
                {
                    ++spatSteps; stats.spatSteps = spatSteps; stats.spatNll = (float) (stats.spatNll * 0.97 + 0.03 * nll);
                    std::lock_guard<std::mutex> ll (lab.m); lab.spatNll = (float) nll; lab.spatGnorm = (float) gnS;
                    if (spatSteps % 10 == 0) lab.line ("spat #%06llu  B%d stereo frames, z detached  β-nll %.4f  |g| %.3f  lr %.2e", (unsigned long long) spatSteps, B, nll, gnS, (double) optS.lr);
                }
                else { spatial.init (91); optS.reset (spatial.ps); }
            }
        }
        // GRU on replayed sequences, every 4th step
        if (vaeSteps % 4 == 0 && seqs.items.size() >= 8)
        {
            std::vector<std::vector<std::array<float, kZ>>> seqZ (8, std::vector<std::array<float, kZ>> (25));
            std::vector<std::array<float, kCond>> cond (8);
            for (int b = 0; b < 8; ++b)
            {
                const auto& c = seqs.sample();
                cond[(size_t) b] = c.cond;
                for (int t = 0; t < 25; ++t) vae.encode (c.x[(size_t) t].data(), c.cond.data(), seqZ[(size_t) b][(size_t) t].data());
            }
            gru.ps.zeroGrad();
            const double nll = gru.trainBatch (seqZ, cond);
            const double gnG = optG.step (gru.ps);
            if (gru.ps.finite()) { ++gruSteps; stats.gruSteps = gruSteps; stats.gruNll = (float) (stats.gruNll * 0.95 + 0.05 * nll); labGru (nll, gnG); }
            else { gru.init (77); optG.reset (gru.ps); }
        }
    }

    int enter()
    {
        for (;;)
        {
            const int i = published.load();
            readers[(size_t) i].fetch_add (1);
            if (published.load() == i) return i;
            readers[(size_t) i].fetch_sub (1);
        }
    }
    static int& heldSlot()   { static thread_local int h = -1; return h; }
    static int& heldSlotUi() { static thread_local int h = -1; return h; }

    void publish (bool force)
    {
        std::lock_guard<std::mutex> pl (publishLock);   // one publisher at a time (trainer thread, offline training)
        // never write the slot a reader holds: `force` waits a little for it to be let go
        const int target = 1 - published.load();
        for (int k = 0; readers[(size_t) target].load() > 0; ++k)
        {
            if (! force || k > 400) return;
            std::this_thread::sleep_for (std::chrono::microseconds (500));
        }
        std::lock_guard<std::mutex> lk (modelLock);
        Snapshot& s = slots[(size_t) target];
        copyInto (s);
        published.store (target);
        version++;
        {
            std::lock_guard<std::mutex> ll (lab.m); ++lab.publishes;
            if (lab.publishes % 50 == 1) lab.line ("pub  snapshot v%llu -> audio threads (double-buffered, lock-free): vae %d params, gru %d, step %llu",
                                                   (unsigned long long) version.load(), (int) vaeParamCount(), (int) gruParamCount(), (unsigned long long) vaeSteps);
        }
    }
    void copyInto (Snapshot& s)
    {
        s.vae = vae; s.gru = gru; s.esn.Win = esn.Win; s.esn.W = esn.W; s.esn.Wout = esn.Wout; s.esn.leak = esn.leak;
        s.som = som; s.feat = featNorm; s.noise = noiseNorm; s.steps = vaeSteps; s.frames = frames;
        s.spatial = spatial; s.spat = spatNorm; s.spatSteps = spatSteps;
        for (Dense* d : { &s.spatial.h1, &s.spatial.h2, &s.spatial.out }) { d->X.clear(); d->Y.clear(); }
        s.spatial.ps = {};
        // drop training caches from the copy (inference uses infer()/step())
        for (Dense* d : { &s.vae.e1, &s.vae.e2, &s.vae.eh, &s.vae.d1, &s.vae.d2, &s.vae.dout, &s.vae.fe.map, &s.vae.fd.map, &s.gru.head })
            { d->X.clear(); d->Y.clear(); }
        s.vae.fe.Hin.clear(); s.vae.fe.GB.clear(); s.vae.fd.Hin.clear(); s.vae.fd.GB.clear();
        for (auto* v : { &s.gru.gru.xs, &s.gru.gru.hs, &s.gru.gru.zs, &s.gru.gru.rs, &s.gru.gru.ns, &s.gru.gru.uhn }) v->clear();
        s.vae.ps = {}; s.gru.ps = {};
    }

    // ---- persistence ------------------------------------------------------------------------------------------
    struct Blob { Vec vae, gru, wout, esnW, esnWin, som, hits, fm, fv, nm, nv, spat, sm, sv; double fc = 0, nc = 0, sc = 0; uint64_t steps = 0, gsteps = 0, frames = 0, ssteps = 0; };
    void toBlob (Blob& b)
    {
        vae.ps.copyWeightsTo (b.vae); gru.ps.copyWeightsTo (b.gru);
        b.wout = esn.Wout; b.esnW = esn.W; b.esnWin = esn.Win; b.som = som.w; b.hits = som.hits;
        b.fm = featNorm.mean; b.fv = featNorm.var; b.nm = noiseNorm.mean; b.nv = noiseNorm.var;
        b.fc = featNorm.count; b.nc = noiseNorm.count; b.steps = vaeSteps; b.gsteps = gruSteps; b.frames = frames;
        spatial.ps.copyWeightsTo (b.spat); b.sm = spatNorm.mean; b.sv = spatNorm.var; b.sc = spatNorm.count; b.ssteps = spatSteps;
    }
    void fromBlob (const Blob& b)
    {
        vae.ps.setWeightsFrom (b.vae); gru.ps.setWeightsFrom (b.gru);
        esn.Wout = b.wout; esn.W = b.esnW; esn.Win = b.esnWin; som.w = b.som; som.hits = b.hits;
        som.steps = b.frames;
        featNorm.mean = b.fm; featNorm.var = b.fv; noiseNorm.mean = b.nm; noiseNorm.var = b.nv;
        featNorm.count = b.fc; noiseNorm.count = b.nc; vaeSteps = b.steps; gruSteps = b.gsteps; frames = b.frames;
        spatial.ps.setWeightsFrom (b.spat); spatNorm.mean = b.sm; spatNorm.var = b.sv; spatNorm.count = b.sc; spatSteps = b.ssteps; stats.spatSteps = spatSteps;
        stats.vaeSteps = vaeSteps; stats.gruSteps = gruSteps; stats.framesLearned = frames;
    }
    static void putVec (std::ostream& o, const Vec& v) { const uint64_t n = v.size(); o.write ((const char*) &n, 8); o.write ((const char*) v.data(), (std::streamsize) (n * 4)); }
    static bool getVec (std::istream& i, Vec& v, size_t expect)
    {
        uint64_t n = 0; i.read ((char*) &n, 8);
        if (! i || n != expect) return false;
        v.resize (n); i.read ((char*) v.data(), (std::streamsize) (n * 4)); return (bool) i;
    }
    uint32_t archHash() const { return (uint32_t) (kFeat * 131 + kZ * 7 + kCond * 3 + kGruH + kSom * 1000 + EchoState::N * 17 + (int) vae.ps.count() + (int) gru.ps.count() + (int) spatial.ps.count() * 3 + kSpatial); }
    bool writeBlob (const std::string& path, const Blob& b)
    {
        const std::string tmp = path + ".tmp";
        {
            std::ofstream o (fsPath (tmp), std::ios::binary);
            if (! o) return false;
            const uint32_t h[3] = { kMagic, kVersion, archHash() };
            o.write ((const char*) h, sizeof (h));
            o.write ((const char*) &b.steps, 8); o.write ((const char*) &b.gsteps, 8); o.write ((const char*) &b.frames, 8);
            o.write ((const char*) &b.fc, 8); o.write ((const char*) &b.nc, 8);
            for (const Vec* v : { &b.vae, &b.gru, &b.wout, &b.esnW, &b.esnWin, &b.som, &b.hits, &b.fm, &b.fv, &b.nm, &b.nv }) putVec (o, *v);
            o.write ((const char*) &b.sc, 8); o.write ((const char*) &b.ssteps, 8);
            for (const Vec* v : { &b.spat, &b.sm, &b.sv }) putVec (o, *v);
            o.flush();
            if (! o) return false;
        }
        // replaces the old file in one step (std::rename can't replace an existing file on Windows)
        std::error_code ec; std::filesystem::rename (fsPath (tmp), fsPath (path), ec);
        return ! ec;
    }
    bool readBlob (const std::string& path, Blob& b)
    {
        std::ifstream i (fsPath (path), std::ios::binary);
        if (! i) return false;
        uint32_t h[3] {}; i.read ((char*) h, sizeof (h));
        if (h[0] != kMagic || h[1] != kVersion || h[2] != archHash()) return false;
        i.read ((char*) &b.steps, 8); i.read ((char*) &b.gsteps, 8); i.read ((char*) &b.frames, 8);
        i.read ((char*) &b.fc, 8); i.read ((char*) &b.nc, 8);
        return getVec (i, b.vae, vae.ps.count()) && getVec (i, b.gru, gru.ps.count()) && getVec (i, b.wout, esn.Wout.size())
            && getVec (i, b.esnW, esn.W.size()) && getVec (i, b.esnWin, esn.Win.size()) && getVec (i, b.som, som.w.size())
            && getVec (i, b.hits, som.hits.size()) && getVec (i, b.fm, (size_t) kFeat) && getVec (i, b.fv, (size_t) kFeat)
            && getVec (i, b.nm, (size_t) kNoise) && getVec (i, b.nv, (size_t) kNoise)
            && (i.read ((char*) &b.sc, 8), i.read ((char*) &b.ssteps, 8), (bool) i)
            && getVec (i, b.spat, spatial.ps.count()) && getVec (i, b.sm, (size_t) kSpatial) && getVec (i, b.sv, (size_t) kSpatial);
    }
    bool load (const std::string& path)
    {
        Blob b;
        if (! readBlob (path, b)) return false;
        fromBlob (b);
        base = b; baseValid = true;
        vae.ps.copyWeightsTo (good);
        return true;
    }
    void save (const std::string& path) { Blob b; toBlob (b); writeBlob (path, b); }
    static void mkdirs (const std::string& p)
    {
        std::error_code ec; std::filesystem::create_directories (fsPath (p), ec);
    }

public:
    /** Federated sync with the brain file: merged = file + (mine - base). Holds an exclusive file lock. */
    void sync()
    {
        std::lock_guard<std::mutex> lk (modelLock);
        mkdirs (dir);
        const std::string path = dir + "/brain.nsb";
        FileLock lock (dir + "/brain.lock");
        if (! lock.ok()) return;
        Blob mine; toBlob (mine);
        Blob file;
        const bool haveFile = ! forceOverwrite && readBlob (path, file);
        if (haveFile && baseValid)
        {
            auto merge = [] (Vec& f, const Vec& m, const Vec& b) { for (size_t i = 0; i < f.size(); ++i) f[i] += m[i] - b[i]; };
            merge (file.vae, mine.vae, base.vae); merge (file.gru, mine.gru, base.gru); merge (file.wout, mine.wout, base.wout);
            merge (file.som, mine.som, base.som); merge (file.hits, mine.hits, base.hits); merge (file.spat, mine.spat, base.spat);
            // statistics: count-weighted
            auto wavg = [] (Vec& f, double fc, const Vec& m, double mc, const Vec& b, double bc)
            {
                const double add = std::max (0.0, mc - bc);
                for (size_t i = 0; i < f.size(); ++i) f[i] = (float) ((f[i] * fc + (m[i] * mc - b[i] * bc)) / std::max (1.0, fc + add));
                (void) mc;
            };
            const double addF = std::max (0.0, mine.fc - base.fc), addN = std::max (0.0, mine.nc - base.nc);
            if (addF > 0) { wavg (file.fm, file.fc, mine.fm, mine.fc, base.fm, base.fc); wavg (file.fv, file.fc, mine.fv, mine.fc, base.fv, base.fc); }
            if (addN > 0) { wavg (file.nm, file.nc, mine.nm, mine.nc, base.nm, base.nc); wavg (file.nv, file.nc, mine.nv, mine.nc, base.nv, base.nc); }
            file.fc += addF; file.nc += addN;
            { const double addS = std::max (0.0, mine.sc - base.sc);
              if (addS > 0) { wavg (file.sm, file.sc, mine.sm, mine.sc, base.sm, base.sc); wavg (file.sv, file.sc, mine.sv, mine.sc, base.sv, base.sc); }
              file.sc += addS; file.ssteps += mine.ssteps - std::min (mine.ssteps, base.ssteps); }
            file.steps += mine.steps - std::min (mine.steps, base.steps);
            file.gsteps += mine.gsteps - std::min (mine.gsteps, base.gsteps);
            file.frames += mine.frames - std::min (mine.frames, base.frames);
            bool ok = true;
            for (auto* v : { &file.vae, &file.gru, &file.wout, &file.som }) for (float x : *v) if (! std::isfinite (x)) { ok = false; break; }
            if (ok)
            {
                fromBlob (file);
                writeBlob (path, file);
                base = file;
                stats.merges++;
            }
        }
        else
        {
            writeBlob (path, mine);
            base = mine;
        }
        baseValid = true; forceOverwrite = false;
    }

private:
    std::string dir;
    // trainer-owned models
    VAE vae; LatentGRU gru; EchoState esn; SOM som;
    RunningNorm featNorm, noiseNorm, spatNorm;
    SpatialNet spatial; uint64_t spatSteps = 0;
    Adam optV, optG, optS;
    Reservoir<Frame> replay, held; uint64_t heldCount = 0;
    Reservoir<SeqChunk> seqs;
    std::array<Frame, 512> recent {}; size_t recentPos = 0;
    SeqChunk chunk {}; int chunkFill = 0;
    std::vector<float> esnState; float prevMu[kZ] {};
    uint64_t vaeSteps = 0, gruSteps = 0, frames = 0;
    double owed = 0; std::atomic<double> owedNow { 0 };
    Vec good;
    Blob base; bool baseValid = false, forceOverwrite = false;
    std::mt19937 rng { 42 };
    // words → stars
    std::mutex wordLock;
    std::map<int, std::map<std::string, float>> nodeWords;
    std::vector<std::string> currentWords;
    // threads
    std::mutex modelLock, feedLock, sleepLock, publishLock;
    std::array<std::atomic<int>, 2> readers {};
    std::vector<std::shared_ptr<Ring>> feeders;
    std::condition_variable cv;
    std::thread worker;
    std::atomic<bool> running { false }, publishNow { false }, syncNow { false };
    // double-buffered snapshot
    std::array<Snapshot, 2> slots;
    std::atomic<int> published { 0 }, inUse { -1 }, uiInUse { -1 };
    std::atomic<uint64_t> version { 0 };
};
} // namespace ns::ml
