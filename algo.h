#pragma once

#include <sycl/sycl.hpp>

template <class T>
void parallel_sort(sycl::queue q, sycl::buffer<T> buf) {
    size_t vectorSize = buf.size();
    int numStages = 0;
    // 2^numStages should be equal to length
    // i.e number of times you halve the lenght to get 1 should be numStages
    for (size_t tmp = vectorSize; tmp > 1; tmp >>= 1) {
        ++numStages;
    }
    for (int stage = 0; stage < numStages; ++stage) {
        // Every stage has stage + 1 passes
        for (int passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            int pairDistanceShift = stage - passOfStage;
            int pairDistanceMask = ~((1 << pairDistanceShift) - 1);
            int pairDistance = 1 << pairDistanceShift;
            q.submit([&] (sycl::handler &cgh) {
                sycl::accessor a{buf, cgh, sycl::read_write};
                cgh.parallel_for(vectorSize / 2, [=](size_t threadId) {
                    int leftId = threadId + (threadId & pairDistanceMask);
                    int rightId = leftId + pairDistance;
                    T leftElement = a[leftId];
                    T rightElement = a[rightId];
                    bool needSwap = (leftElement > rightElement) ^ ((threadId >> stage) & 1);
                    if (needSwap) {
                        std::swap(leftElement, rightElement);
                    }
                    a[leftId] = leftElement;
                    a[rightId] = rightElement;
                });
            });
        }
    }
}

template <class HistT, class ValT>
void parallel_histogram(sycl::queue q, sycl::buffer<HistT> hist_buf, sycl::buffer<ValT> val_buf) {
    size_t hist_n = hist_buf.size();
    size_t val_n = val_buf.size();
    sycl::buffer<HistT> tmp_buf{sycl::range<1>{hist_n}};
    parallel_sort(q, val_buf);
    q.submit([&] (sycl::handler &cgh) {
        sycl::accessor tmp_axr{tmp_buf, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor val_axr{val_buf, cgh, sycl::read_only};
        cgh.parallel_for(sycl::range<1>(hist_n), [=] (size_t i) {
            tmp_axr[i] = std::upper_bound(val_axr.begin(), val_axr.end(), i) - val_axr.begin();
        });
    });
    q.submit([&] (sycl::handler &cgh) {
        sycl::accessor hist_axr{hist_buf, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor tmp_axr{tmp_buf, cgh, sycl::read_only};
        cgh.parallel_for(sycl::range<1>(hist_n), [=] (size_t i) {
            if (i != 0)
                hist_axr[i] = tmp_axr[i] - tmp_axr[i - 1];
            else
                hist_axr[i] = tmp_axr[i];
        });
    });
}
