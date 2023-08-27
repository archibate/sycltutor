#include "autobench.h"
#include <execution>

void omp_sort(int *a, size_t vectorSize) {
    int numStages = 0;
    // 2^numStages should be equal to length
    // i.e number of times you halve the lenght to get 1 should be numStages
    for (size_t tmp = vectorSize; tmp > 1; tmp >>= 1) {
        ++numStages;
    }
    size_t maxThreadCount = vectorSize / 2;
    for (int stage = 0; stage < numStages; ++stage) {
        // Every stage has stage + 1 passes
        for (int passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            int pairDistanceShift = stage - passOfStage;
            int pairDistanceMask = ~((1 << pairDistanceShift) - 1);
            int pairDistance = 1 << pairDistanceShift;
            if (pairDistanceShift < 3) {
                #pragma omp parallel for
                for (int i = 0; i < maxThreadCount / 4 * 4; i += 4) {
                    int leftId = i + (i & pairDistanceMask);
                    int rightId = leftId + pairDistance;
                    int leftElement = a[leftId];
                    int rightElement = a[rightId];
                    bool needSwap = (leftElement > rightElement) ^ ((i >> stage) & 1);
                    a[leftId] = !needSwap ? leftElement : rightElement;
                    a[rightId] = !needSwap ? rightElement : leftElement;
                }
            } else {
                __m256i threadIdOffset = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                #pragma omp parallel for
                for (int i = 0; i < maxThreadCount / 8 * 8; i += 8) {
                    auto leftId = i + (i & pairDistanceMask);
                    auto rightId = leftId + pairDistance;
                    __m256i threadId = _mm256_add_epi32(_mm256_set1_epi32(i), threadIdOffset);
                    __m256i leftElement = _mm256_loadu_si256((__m256i *)(a + leftId));
                    __m256i rightElement = _mm256_loadu_si256((__m256i *)(a + rightId));
                    __m256i needSwap = _mm256_xor_si256(_mm256_cmpgt_epi32(rightElement, leftElement),
                                                     _mm256_sub_epi32(_mm256_setzero_si256(),
                                                                   _mm256_and_si256(_mm256_srli_epi32(threadId, stage),
                                                                                 _mm256_set1_epi32(1))));
                    _mm256_storeu_si256((__m256i *)(a + leftId), _mm256_blendv_epi8(leftElement, rightElement, needSwap));
                    _mm256_storeu_si256((__m256i *)(a + rightId), _mm256_blendv_epi8(rightElement, leftElement, needSwap));
                }
            }
        }
    }
}

/* template <class ValT> */
/* void sycl_sort(ValT *val_p, size_t val_n) { */
/*     buffer<ValT> val_buf{val_p, range<1>{val_n}}; */
/*     parallel_sort(g_queue, val_buf); */
/* } */

template <class ValT>
void cpu_sort(ValT *val_p, size_t val_n) {
    std::sort(val_p, val_p + val_n);
}

/* template <class ValT> */
/* void tbb_sort(ValT *val_p, size_t val_n) { */
/*     std::sort(std::execution::par, val_p, val_p + val_n); */
/* } */

constexpr size_t n = 65536 * 64;

static int bench_histogram_uint32_t = doAutoBench("histogram", std::array{
    cpu_sort<int>,
    // tbb_sort<int>,
    omp_sort,
    // sycl_sort<int>,
}, n, n);

/* int main() { */
/*     std::vector<uint32_t> buf(n); */
/*     _makeAutoBench_details::do_randomize(buf, 0u, 0xffffffffu, 0); */
/*     omp_sort(buf.data(), buf.size()); */
/*     if (std::is_sorted(buf.begin(), buf.end())) { */
/*         printf("sorted!\n"); */
/*     } else { */
/*         printf("not sorted!\n"); */
/*     } */
/* } */
