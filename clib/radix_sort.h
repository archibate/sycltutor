#pragma once

#include <sycl/sycl.hpp>
#include "exclusive_scan.h"

class KT_radix_sort_histogram;
class KT_radix_sort_compindex;
class KT_radix_sort_reorder;

inline void radix_sort(sycl::queue &q, sycl::buffer<unsigned> &buf) {
    using atomic_ref = sycl::atomic_ref<unsigned, sycl::memory_order_relaxed, sycl::memory_scope_work_group>;
    sycl::buffer<unsigned> buf_next{buf.size()};
    sycl::buffer<unsigned> hist_group{buf.size()};
    sycl::buffer<unsigned> index_buf{buf.size()};
    for (int bit = 0; bit < 4; bit++) {
        q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<unsigned> count{256, cgh};
            sycl::accessor<unsigned> hist{hist_group, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor<unsigned> a{buf, cgh, sycl::read_only};
            cgh.parallel_for<KT_radix_sort_histogram>(sycl::nd_range<1>{buf.size(), 256}, [=] (sycl::nd_item<1> it) {
                int ii = it.get_local_id(0);
                int gi = it.get_group(0);
                int gn = it.get_group_range(0);
                int i = it.get_global_id(0);
                count[ii] = 0;
                it.barrier(sycl::access::fence_space::local_space);
                unsigned key = (a[i] >> bit * 8) & 0xff;
                atomic_ref{count[key]}.fetch_add(1u);
                it.barrier(sycl::access::fence_space::local_space);
                hist[ii * gn + gi] = count[ii];
            });
        });
        exclusive_scan(q, hist_group);
        q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<unsigned> count{256, cgh};
            sycl::accessor<unsigned> hist{hist_group, cgh, sycl::read_only};
            sycl::accessor<unsigned> a{buf, cgh, sycl::read_only};
            sycl::accessor<unsigned> indices{index_buf, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for<KT_radix_sort_compindex>(sycl::nd_range<1>{buf.size(), 128}, [=] (sycl::nd_item<1> it) {
                int ii = it.get_local_id(0);
                int gi = it.get_group(0) >> 1;
                int gn = it.get_group_range(0) >> 1;
                int i = it.get_global_id(0);
                count[ii] = hist[ii * gn + gi];
                count[ii + 128] = hist[(ii + 128) * gn + gi];
                it.barrier(sycl::access::fence_space::local_space);
                unsigned key = (a[i] >> bit * 8) & 0xff;
                __hipsycl_if_target_host(
                indices[i] = atomic_ref{count[key]}.fetch_add(1u);
                );
                __hipsycl_if_target_device(
                for (int k = 0; k < 4; k++) {
                    if ((ii >> 5) == k) {
                        int mask = __match_any_sync(0xffffffff, key);
                        int lane_id = ii & 31;
                        int leader = __ffs(mask) - 1;
                        int res = 0;
                        if (lane_id == leader) {
                            res = count[key];
                            count[key] = res + __popc(mask);
                        }
                        res = __shfl_sync(mask, res, leader);
                        indices[i] = res + __popc(mask & ((1 << lane_id) - 1));
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                );
            });
        });
        q.submit([&] (sycl::handler &cgh) {
            sycl::accessor<unsigned> indices{index_buf, cgh, sycl::read_only};
            sycl::accessor<unsigned> a{buf, cgh, sycl::read_only};
            sycl::accessor<unsigned> aout{buf_next, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for<KT_radix_sort_reorder>(sycl::nd_range<1>{buf.size() / 2, 128}, [=] (sycl::nd_item<1> it) {
                int i = it.get_global_id(0) * 2;
                aout[indices[i]] = a[i];
                aout[indices[i + 1]] = a[i + 1];
            });
        });
        std::swap(buf, buf_next);
    }
}
