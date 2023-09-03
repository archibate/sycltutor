#include "utils/wangshash.h"
#include "clib/print_buffer.h"
#include "clib/radix_sort.h"
#include <vector>
#include <execution>
#include "utils/ticktock.h"

int main() {
    constexpr size_t n = 4 * 256 * 256 * 256;
    {
        sycl::queue q{sycl::gpu_selector_v};
        std::cerr << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::vector<unsigned> arr(n);
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = wangshash(i)();
        }
        TICK(radix);
        {
            sycl::buffer<unsigned> buf{arr};
            radix_sort(q, buf);
            q.wait();
        }
        TOCK(radix);
        if (auto it = std::is_sorted_until(arr.begin(), arr.end()); it == arr.end()) {
            printf("sorted successfully\n");
        } else {
            printf("not sorted since %ld\n", it - arr.begin());
            print_buffer(arr);
        }
    }
    {
        std::vector<unsigned> arr(n);
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = wangshash(i)();
        }
        TICK(tbb);
        {
            std::sort(std::execution::par_unseq, arr.begin(), arr.end());
        }
        TOCK(tbb);
    }
    return 0;
}
