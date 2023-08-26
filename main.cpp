#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
    queue q{cpu_selector_v};
    q.submit([&] (handler &cgh) {
        stream cl_cout{1024, 1024, cgh};
        cgh.parallel_for(range<1>{4}, [=] (item<1> it) {
            cl_cout << "I am " << it.get_id()[0] << '\n';
        });
    });
    return 0;
}
