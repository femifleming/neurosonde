// The portable matrix product (Windows, Linux) against Accelerate's cblas_sgemm (the Mac) — or, where there is no
// Accelerate, against the product by its definition in double precision — every transpose and beta the networks
// use, at the networks' sizes.
//   clang++ -std=c++20 -O2 -ISource tests/GemmCheck.cpp -framework Accelerate -o /tmp/gemmcheck && /tmp/gemmcheck
#include "ml/NN.h"
#include <cstdio>
using namespace ns::ml;

int main()
{
    int fails = 0;
    std::mt19937 rng (3); std::normal_distribution<float> nd;
    for (int M : { 1, 7, 64 }) for (int N : { 1, 16, 97 }) for (int K : { 3, 84, 256 })
        for (int ta = 0; ta < 2; ++ta) for (int tb = 0; tb < 2; ++tb) for (float beta : { 0.0f, 1.0f, 0.5f })
        {
            std::vector<float> A ((size_t) M * K), B ((size_t) K * N), C0 ((size_t) M * N), C1, C2;
            for (auto& v : A) v = nd (rng); for (auto& v : B) v = nd (rng); for (auto& v : C0) v = nd (rng);
            C1 = C0; C2 = C0;
           #if defined(__APPLE__)
            gemm (ta, tb, M, N, K, 0.7f, A.data(), B.data(), beta, C1.data());     // Accelerate
           #else
            for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j)
            {
                double acc = 0; for (int k = 0; k < K; ++k) acc += (double) (ta ? A[(size_t) k * M + i] : A[(size_t) i * K + k]) * (tb ? B[(size_t) j * K + k] : B[(size_t) k * N + j]);
                C1[(size_t) i * N + j] = (float) (0.7 * acc + beta * C0[(size_t) i * N + j]);
            }
           #endif
            gemmPortable (ta, tb, M, N, K, 0.7f, A.data(), B.data(), beta, C2.data());
            // float rounding is bounded by the sum of the products' magnitudes (cancellation can make the result tiny)
            double worst = 0;
            for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j)
            {
                double bound = std::abs (beta * C0[(size_t) i * N + j]);
                for (int k = 0; k < K; ++k) bound += 0.7 * std::abs ((ta ? A[(size_t) k * M + i] : A[(size_t) i * K + k]) * (tb ? B[(size_t) j * K + k] : B[(size_t) k * N + j]));
                worst = std::max (worst, std::abs (C1[(size_t) i * N + j] - C2[(size_t) i * N + j]) / (bound + 1e-12));
            }
            if (worst > 1e-6) { ++fails; std::printf ("FAIL M %d N %d K %d tA %d tB %d beta %.1f  %.1e of the bound\n", M, N, K, ta, tb, beta, worst); }
        }
    std::printf (fails ? "FAILED (%d)\n" : "all passed (324 shapes)\n", fails);
    return fails;
}
