#include <sycl/sycl.hpp>
#include "utils/wangshash.h"
#include "clib/radix_sort.h"
#include <vector>
#include <execution>
#include "utils/ticktock.h"

/* template <class T> */
/* using Buffer = sycl::buffer<T>; */
/* using Vec3f = sycl::vec<float, 3>; */
/* using Vec3u = sycl::vec<unsigned, 3>; */
/*  */
/* struct Mesh { */
/*     Buffer<Vec3f> vertices; */
/*     Buffer<Vec3u> indices; */
/* }; */
/*  */
/* inline void compute_mesh_normal(sycl::queue &q, sycl::buffer<sycl::vec<float, 3>> &vertices, sycl::buffer<sycl::vec<unsigned, 3>> &indices) { */
/*     if (vertices.size() <= 1) return; */
/*     q.submit([&] (sycl::handler &cgh) { */
/*         sycl::accessor vert{vertices, cgh, sycl::read_only}; */
/*         cgh.parallel_for(sycl::nd_range<1>{vert.size(), 256}, [=] (sycl::nd_item<1> it) { */
/*         }); */
/*     }); */
/* } */

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
