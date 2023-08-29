#pragma once

#include <sycl/sycl.hpp>
#include "exclusive_scan.h"

class KT_radix_sort_histogram;
class KT_radix_sort_scatter;

template <
    sycl::memory_order memord = sycl::memory_order_relaxed,
    sycl::memory_scope memscp = sycl::memory_scope_work_group,
    class T>
sycl::atomic_ref<T, memord, memscp> atomic_ref(T &t) {
    return sycl::atomic_ref<T, memord, memscp>{t};
}

template <class T>
void lock(T &t) {
    while (atomic_ref<sycl::memory_order_acq_rel>(t).exchange(1) == 1);
}

template <class T>
void unlock(T &t) {
    atomic_ref<sycl::memory_order_acq_rel>(t).store(0);
}

inline void radix_sort(sycl::queue &q, sycl::buffer<unsigned> &buf) {
    sycl::buffer<unsigned> buf_next{buf.size()};
    sycl::buffer<unsigned> hist_group{buf.size()};
    for (int bit = 0; bit < 4; bit++) {
        q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<unsigned> count{256, cgh};
            sycl::accessor hist{hist_group, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor a{buf, cgh, sycl::read_only};
            cgh.parallel_for<KT_radix_sort_histogram>(sycl::nd_range<1>{buf.size(), 256}, [=] (sycl::nd_item<1> it) {
                int ii = it.get_local_id(0);
                int gi = it.get_group(0);
                int gn = it.get_group_range(0);
                int i = it.get_global_id(0);
                count[ii] = 0;
                it.barrier(sycl::access::fence_space::local_space);
                unsigned key = (a[i] >> bit * 8) & 0xff;
                atomic_ref(count[key]).fetch_add(1u);
                it.barrier(sycl::access::fence_space::local_space);
                hist[ii * gn + gi] = count[ii];
            });
        });
        exclusive_scan(q, hist_group);
        q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<unsigned> count{256, cgh};
            sycl::local_accessor<unsigned> bits{256 * 9, cgh};
            sycl::accessor hist{hist_group, cgh, sycl::read_only};
            sycl::accessor a{buf, cgh, sycl::read_only};
            sycl::accessor aout{buf_next, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for<KT_radix_sort_scatter>(sycl::nd_range<1>{buf.size(), 256}, [=] (sycl::nd_item<1> it) {
                int ii = it.get_local_id(0);
                int gi = it.get_group(0);
                int gn = it.get_group_range(0);
                int i = it.get_global_id(0);
                auto g = it.get_group();
                count[ii] = hist[ii * gn + gi];
                for (int k = 0; k < 8; k++) {
                    bits[ii * 9 + k] = 0u;
                }
                it.barrier(sycl::access::fence_space::local_space);
                unsigned key = (a[i] >> bit * 8) & 0xff;
                atomic_ref(bits[key * 9 + (ii >> 5)]).fetch_or(1u << (ii & 31));
                it.barrier(sycl::access::fence_space::local_space);
                int popcorns = 0;
                for (int k = 0; k <= (ii >> 5); k++) {
                    unsigned mask = bits[key * 9 + k];
                    if (ii < (k + 1) * 32)
                        mask &= (1u << (ii & 31)) - 1u;
                    popcorns += sycl::popcount(mask);
                }
                unsigned index = count[key] + popcorns;
                aout[index] = a[i];
            });
        });
        std::swap(buf, buf_next);
    }
}
