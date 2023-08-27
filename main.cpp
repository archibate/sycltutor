#include <sycl/sycl.hpp>
#include "wangshash.h"
#include <vector>
#include <chrono>
#include <iostream>
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) std::cerr<<#x ": "<<std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-bench_##x).count();std::cerr<<"秒\n";

template <class T>
void print_buffer(std::vector<T> const &buf) {
    std::cout << "CPU [ ";
    for (size_t i = 0; i < buf.size(); i++) {
        std::cout << buf[i] << ' ';
    }
    std::cout << "]\n";
}

template <class T>
void print_buffer(sycl::buffer<T> buf) {
    sycl::host_accessor<T> hacc{buf, sycl::read_only};
    std::cout << "GPU [ ";
    for (size_t i = 0; i < hacc.size(); i++) {
        std::cout << hacc[i] << ' ';
    }
    std::cout << "]\n";
}

void parallel_scan(sycl::queue &q, sycl::buffer<unsigned> &hist_group) {
    if (hist_group.size() <= 1) return;
    sycl::buffer<unsigned> glob_sum{(hist_group.size() + 255) / 256};
    q.submit([&] (sycl::handler &cgh) {
        sycl::accessor<unsigned> hist{hist_group, cgh, sycl::read_write};
        sycl::accessor<unsigned> gsum{glob_sum, cgh, sycl::write_only, sycl::no_init};
        cgh.parallel_for(sycl::nd_range<1>{glob_sum.size() * 256, 256}, [=] (sycl::nd_item<1> it) {
            int ii = it.get_local_id(0);
            int gi = it.get_group(0);
            int i = it.get_global_id(0);
            unsigned val = sycl::inclusive_scan_over_group(it.get_group(), i < hist.size() ? hist[i] : 0u, 0u, std::plus<>{});
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
        parallel_scan(q, glob_sum);
        q.submit([&] (sycl::handler &cgh) {
            sycl::accessor<unsigned> hist{hist_group, cgh, sycl::read_write};
            sycl::accessor<unsigned> gsum{glob_sum, cgh, sycl::read_only};
            cgh.parallel_for(sycl::nd_range<1>{glob_sum.size() * 256, 256}, [=] (sycl::nd_item<1> it) {
                int gi = it.get_group(0);
                int i = it.get_global_id(0);
                if (i < hist.size())
                    hist[i] += gsum[gi];
            });
        });
    }
}

void radix_sort(sycl::queue &q, sycl::buffer<unsigned> &buf) {
    using atomic_ref = sycl::atomic_ref<unsigned, sycl::memory_order_relaxed, sycl::memory_scope_device>;
    sycl::buffer<unsigned> buf_next{buf.size()};
    sycl::buffer<unsigned> hist_group{buf.size()};
    for (int bit = 0; bit < 4; bit++) {
        q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<unsigned> count{256, cgh};
            sycl::accessor<unsigned> hist{hist_group, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor<unsigned> a{buf, cgh, sycl::read_only};
            cgh.parallel_for(sycl::nd_range<1>{buf.size(), 256}, [=] (sycl::nd_item<1> it) {
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
        parallel_scan(q, hist_group);
        q.submit([&] (sycl::handler &cgh) {
            sycl::accessor<unsigned> hist{hist_group, cgh, sycl::read_only};
            sycl::accessor<unsigned> a{buf, cgh, sycl::read_only};
            sycl::accessor<unsigned> aout{buf_next, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for(sycl::range<1>{buf.size() / 256}, [=] (sycl::item<1> it) {
                int gi = it.get_id(0);
                int gn = it.get_range(0);
                unsigned count[256];
                for (int ii = 0; ii < 256; ii++) {
                    count[ii] = hist[ii * gn + gi];
                }
                for (int ii = 0; ii < 256; ii++) {
                    int i = gi * 256 + ii;
                    auto ai = a[i];
                    unsigned key = (ai >> bit * 8) & 0xff;
                    auto index = count[key];
                    aout[index] = ai;
                    count[key] = index + 1;
                }
            });
        });
        q.submit([&] (sycl::handler &cgh) {
            sycl::accessor<unsigned> aout{buf_next, cgh, sycl::read_only};
            sycl::accessor<unsigned> a{buf, cgh, sycl::write_only, sycl::no_init};
            cgh.copy(aout, a);
        });
    }
}

int main() {
    sycl::queue q{sycl::cpu_selector_v};
    std::vector<unsigned int> arr(8 * 256);
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = wangshash(i)() % 10;
    }
    TICK(t);
    {
        sycl::buffer<unsigned int> buf{arr};
        radix_sort(q, buf);
    }
    TOCK(t);
    if (auto it = std::is_sorted_until(arr.begin(), arr.end()); it == arr.end()) {
        printf("sorted successfully\n");
    } else {
        printf("not sorted since %ld\n", it - arr.begin());
    }
    return 0;
}
