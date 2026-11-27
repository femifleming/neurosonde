// Neurosonde — the lab: instrumentation of the real training, for the lab page. The trainer thread writes it (under
// its own mutex, never the model lock), the UI copies what it needs. Everything here is measured from the actual
// networks during the actual optimisation: losses per term, the optimiser's state, per-layer weight / gradient /
// update statistics and histograms, the weight matrices and a batch's activations, the latent posterior per
// dimension, the reconstruction of a held-out frame, and a verbose log of every event.
#pragma once
#include "Models.h"
#include <mutex>
#include <deque>
#include <string>
#include <cstdio>
#include <cstdarg>

namespace ns::ml
{
struct LayerStat
{
    std::string name, act;
    int in = 0, out = 0; size_t params = 0;
    float wRms = 0, wMax = 0, gRms = 0;
    float upd = 0;               // ‖Δw‖ / ‖w‖ for the last optimiser step (healthy ≈ 1e-3)
    float actMean = 0, actStd = 0, negFrac = 0; bool hasAct = false;
    std::array<float, 33> wHist {};   // weights, −wMax..+wMax, normalised to the fullest bin
    std::array<float, 33> gHist {};   // log10 |gradient|, −9..−1
};

struct Lab
{
    mutable std::mutex m;

    // ---- every optimiser step of the VAE --------------------------------------------------------------------------
    static constexpr int N = 600;
    struct Step { float total = 0, recon = 0, harm = 0, noise = 0, kl = 0, beta = 0, lr = 0, gnorm = 0, clipK = 1, us = 0; };
    std::array<Step, N> series {}; int pos = 0; uint64_t count = 0;
    Step last;
    float stepsPerSec = 0, framesPerSec = 0; int batchRecent = 0, batchReplay = 0, batchVoiced = 0; bool offlineMode = false;
    uint64_t restores = 0, publishes = 0, gruStepsSeen = 0;
    float gruNll = 0, gruGnorm = 0, spatNll = 0, spatGnorm = 0;
    long adamT = 0; float adamClip = 1, adamWd = 0;

    // ---- per layer (every 10 steps) -------------------------------------------------------------------------------
    std::vector<LayerStat> layers; uint64_t layersAt = 0;

    // ---- matrices: weights of three layers, and one batch's activations ----------------------------------------------
    struct Mat { std::string name; int rows = 0, cols = 0; std::vector<float> v; float scale = 1; };
    std::array<Mat, 3> W;          // e1 (128 × 84), d2 (128 × 128), dout (164 × 128)
    std::array<Mat, 6> A;          // activations [batch × units]: e1, e2, μ, σ, d1, d2

    // ---- the latent posterior, per dimension (EMA over batches) -----------------------------------------------------
    std::array<float, kZ> klDim {}, muMean {}, muStd {}, sigMean {};

    // ---- which outputs it gets right: squared error per output (EMA over batches, standardised units) ---------------
    std::array<float, kFeat> featErr {}; std::array<float, kNoise> noiseErr {}; std::array<float, kHarm> harmErr {};

    // ---- a held-out frame, reconstructed (eval mode: latent mean) -----------------------------------------------------
    bool exValid = false; uint64_t exAt = 0;
    std::array<float, kFeat> exX {}, exXh {};
    std::array<float, kHarm> exHt {}, exHp {};
    std::array<float, kNoise> exNt {}, exNp {};
    std::array<float, kZ> exMu {}, exSig {};
    float exErr = 0, exVoiced = 0;

    // ---- the log ---------------------------------------------------------------------------------------------------
    std::deque<std::string> log; uint64_t logSeq = 0;
    /** append a line (the caller holds m) */
    void line (const char* fmt, ...)
    {
        char buf[320];
        va_list ap; va_start (ap, fmt); std::vsnprintf (buf, sizeof (buf), fmt, ap); va_end (ap);
        log.emplace_back (buf);
        if (log.size() > 800) log.pop_front();
        ++logSeq;
    }
};
} // namespace ns::ml
