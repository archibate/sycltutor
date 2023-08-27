#include <sycl/sycl.hpp>
#include "autobench.h"
#include "algo.h"

using namespace sycl;

static queue g_queue{gpu_selector_v};

template <class HistT, class ValT>
void sycl_histogram(HistT *hist_p, size_t hist_n, ValT const *val_p, size_t val_n) {
    buffer<HistT> hist_buf{hist_p, range<1>{hist_n}};
    buffer<ValT> val_buf{val_p, range<1>{val_n}};
    return parallel_histogram(g_queue, hist_buf, val_buf);
}

template <class HistT, class ValT>
void dumb_parallel_histogram(HistT *hist_p, size_t hist_n, ValT const *val_p, size_t val_n) {
    buffer hist_buf{hist_p, range<1>{hist_n}};
    buffer val_buf{val_p, range<1>{val_n}};
    g_queue.submit([&] (handler &cgh) {
        constexpr size_t k_block_size = 256;
        constexpr size_t k_num_bins = 256;
        constexpr size_t k_group_size = 128;
        accessor hist_axr{hist_buf, cgh, read_write};
        accessor val_axr{val_buf, cgh, read_only};
        local_accessor<HistT> hist_loc{k_num_bins, cgh};
        cgh.parallel_for(nd_range<1>{((val_n + k_block_size - 1) / k_block_size + k_group_size - 1) / k_group_size * k_group_size, k_group_size}, [=] (nd_item<1> it) {
            auto i = it.get_global_id()[0];
            auto li = it.get_local_id()[0];
            hist_loc[li * 2] = 0;
            hist_loc[li * 2 + 1] = 0;
            it.barrier(access::fence_space::local_space);
            for (size_t j = 0; j < k_block_size; j++) {
                if (i * k_block_size + j < val_axr.size()) {
                    auto v = val_axr[i * k_block_size + j] % k_num_bins;
                    atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::local_space>{hist_loc[v]}.fetch_add(HistT{1});
                }
            }
            it.barrier(access::fence_space::local_space);
            atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::global_space>{hist_axr[li * 2]}.fetch_add(hist_loc[li * 2]);
            atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::global_space>{hist_axr[li * 2 + 1]}.fetch_add(hist_loc[li * 2 + 1]);
        });
    }).wait_and_throw();
}

template <class HistT, class ValT>
void cpu_histogram(HistT *hist_p, size_t hist_n, ValT const *val_p, size_t val_n) {
    memset(hist_p, 0, hist_n * sizeof(HistT));
    for (size_t i = 0; i < val_n; i++) {
        ++hist_p[val_p[i]];
    }
}

constexpr size_t n = 65536 * 256;
constexpr size_t maxval = 65536;

static int bench_histogram_uint32_t = doAutoBench("histogram", std::array{
    cpu_histogram<uint32_t, uint32_t>,
    sycl_histogram<uint32_t, uint32_t>,
}, maxval, maxval, {n, 0, maxval - 1}, n);
