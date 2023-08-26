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
    sycl::range<1> r{vectorSize / 2};
    for (int stage = 0; stage < numStages; ++stage) {
        // Every stage has stage + 1 passes
        for (int passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            q.submit([&] (sycl::handler &h) {
                sycl::accessor a{buf, h, sycl::read_write};
                h.parallel_for(
                    sycl::range<1>{r},
                    [a, stage, passOfStage](sycl::item<1> it) {
                        int sortIncreasing = 1;
                        int threadId = it.get_linear_id();

                        int pairDistance = 1 << (stage - passOfStage);
                        int blockWidth = 2 * pairDistance;

                        int leftId = (threadId % pairDistance) +
                            (threadId / pairDistance) * blockWidth;
                        int rightId = leftId + pairDistance;

                        T leftElement = a[leftId];
                        T rightElement = a[rightId];

                        int sameDirectionBlockWidth = 1 << stage;

                        if ((threadId / sameDirectionBlockWidth) % 2 == 1) {
                            sortIncreasing = 1 - sortIncreasing;
                        }

                        T greater;
                        T lesser;

                        if (leftElement > rightElement) {
                            greater = leftElement;
                            lesser = rightElement;
                        } else {
                            greater = rightElement;
                            lesser = leftElement;
                        }

                        a[leftId] = sortIncreasing ? lesser : greater;
                        a[rightId] = sortIncreasing ? greater : lesser;
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
