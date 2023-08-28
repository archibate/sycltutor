#include <sycl/sycl.hpp>
#include "utils/wangshash.h"
#include "clib/radix_sort.h"
#include <vector>
#include <execution>
#include "utils/ticktock.h"

int main() {
    constexpr size_t n = 4 * 256 * 256 * 256;
    {
        std::vector<unsigned int> arr(n);
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = wangshash(i)();
        }
        TICK(tbb);
        {
            std::sort(std::execution::par, arr.begin(), arr.end());
        }
        TOCK(tbb);
    }
    {
        sycl::queue q{sycl::gpu_selector_v};
        std::vector<unsigned int> arr(n);
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = wangshash(i)();
        }
        TICK(radix);
        {
            sycl::buffer<unsigned int> buf{arr};
            radix_sort(q, buf);
        }
        TOCK(radix);
        if (auto it = std::is_sorted_until(arr.begin(), arr.end()); it == arr.end()) {
            printf("sorted successfully\n");
        } else {
            printf("not sorted since %ld\n", it - arr.begin());
        }
    }
    return 0;
}
