#pragma once

#include <sycl/sycl.hpp>

class KT_exclusive_scan;
class KT_exclusive_scan_paste;

inline void exclusive_scan(sycl::queue &q, sycl::buffer<unsigned> &hist_group) {
    if (hist_group.size() <= 1) return;
    sycl::buffer<unsigned> glob_sum{(hist_group.size() + 255) / 256};
    q.submit([&] (sycl::handler &cgh) {
        sycl::accessor hist{hist_group, cgh, sycl::read_write};
        sycl::accessor gsum{glob_sum, cgh, sycl::write_only, sycl::no_init};
        cgh.parallel_for<KT_exclusive_scan>(sycl::nd_range<1>{glob_sum.size() * 256, 256}, [=] (sycl::nd_item<1> it) {
            int ii = it.get_local_id(0);
            int gi = it.get_group(0);
            int i = it.get_global_id(0);
            unsigned val = sycl::inclusive_scan_over_group(it.get_group(), i < hist.size() ? hist[i] : 0u, std::plus<>{});
            if (ii == 255) {
                gsum[gi] = val;
            } else if (i + 1 < hist.size()) {
                hist[i + 1] = val;
            }
            if (ii == 0) {
                hist[i] = 0;
            }
        });
    });
    if (glob_sum.size() > 1) {
        exclusive_scan(q, glob_sum);
        q.submit([&] (sycl::handler &cgh) {
            sycl::accessor hist{hist_group, cgh, sycl::read_write};
            sycl::accessor gsum{glob_sum, cgh, sycl::read_only};
            cgh.parallel_for<KT_exclusive_scan_paste>(sycl::nd_range<1>{glob_sum.size() * 256, 256}, [=] (sycl::nd_item<1> it) {
                int gi = it.get_group(0);
                int i = it.get_global_id(0);
                if (i < hist.size())
                    hist[i] += gsum[gi];
            });
        });
    }
}
