#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <cassert>
#include <sycl/sycl.hpp>
#include "sycl_context_manager.h"

sycl::nd_range<1> sycl_decide_range
( size_t size
, size_t wgsize = 16
, size_t division = 1
) {
    return sycl::nd_range<1>(sycl::range<1>(size / division), sycl::range<1>(wgsize));
}

template <class T>
void sycl_exclusive_scan
( sycl::handler &cgh
, sycl::nd_range<1> const &range
, sycl::buffer<T, 1> &input
) {
    sycl::accessor inputAxr{input, cgh, sycl::read_write};
    cgh.parallel_for(
        range,
        [=] (sycl::nd_item<1> it) {
            T *first = inputAxr.get_pointer();
            T *last = first + inputAxr.size();
            sycl::joint_exclusive_scan(it.get_group(), first, last, first, std::plus<>());
        });
}

template <class T>
void sycl_inclusive_scan
( sycl::handler &cgh
, sycl::nd_range<1> const &range
, sycl::buffer<T, 1> &input
) {
    sycl::accessor inputAxr{input, cgh, sycl::read_write};
    cgh.parallel_for(
        range,
        [=] (sycl::nd_item<1> it) {
            T *first = inputAxr.get_pointer();
            T *last = first + inputAxr.size();
            sycl::joint_inclusive_scan(it.get_group(), first, last, first, std::plus<>());
        });
}


template <class T>
void sycl_exclusive_scan
( sycl::handler &cgh
, sycl::nd_range<1> const &range
, sycl::buffer<T, 1> &input
, sycl::buffer<T, 1> &output
) {
    sycl::accessor inputAxr{input, cgh, sycl::read_only};
    sycl::accessor outputAxr{output, cgh, sycl::write_only, sycl::no_init};
    cgh.parallel_for(
        range,
        [=] (sycl::nd_item<1> it) {
            T *first = inputAxr.get_pointer();
            T *last = first + inputAxr.size();
            T *result = outputAxr.get_pointer();
            sycl::joint_exclusive_scan(it.get_group(), first, last, result, std::plus<>());
        });
}

template <class T>
void sycl_inclusive_scan
( sycl::handler &cgh
, sycl::nd_range<1> const &range
, sycl::buffer<T, 1> &input
, sycl::buffer<T, 1> &output
) {
    sycl::accessor inputAxr{input, cgh, sycl::read_only};
    sycl::accessor outputAxr{output, cgh, sycl::write_only, sycl::no_init};
    cgh.parallel_for(
        range,
        [=] (sycl::nd_item<1> it) {
            T *first = inputAxr.get_pointer();
            T *last = first + inputAxr.size();
            T *result = outputAxr.get_pointer();
            sycl::joint_inclusive_scan(it.get_group(), first, last, result, std::plus<>());
        });
}

template <class T>
void sycl_reduce
( sycl::handler &cgh
, sycl::nd_range<1> const &range
, sycl::buffer<T, 1> &input
, sycl::buffer<T, 1> &output
) {
    sycl::accessor inputAxr{input, cgh, sycl::read_only};
    sycl::accessor outputAxr{output, cgh, sycl::write_only, sycl::no_init};
    cgh.parallel_for(
        range,
        [=] (sycl::nd_item<1> it) {
            T *first = inputAxr.get_pointer();
            T *last = first + inputAxr.size();
            T sum = sycl::joint_reduce(it.get_group(), first, last, 0, std::plus<>{});
            outputAxr[0] = sum;
        });
}

template <class T>
T sycl_reduce
( sycl::handler &cgh
, sycl::nd_range<1> const &range
, sycl::buffer<T, 1> &input
) {
    sycl::buffer<T, 1> output{1};
    sycl_reduce(cgh, range, input, output);
    sycl::host_accessor outputHostAxr{output, sycl::read_only};
    return outputHostAxr[0];
}

int main() {
    auto context = create_default_sycl_context();
    auto queue = sycl::queue(context, context.get_devices());

    auto arr = std::vector<int32_t>(100);
    auto mt_engine = std::mt19937(std::random_device{}());
    auto idist = std::uniform_int_distribution<int32_t>(0, 10);
    std::cout << "Data: ";
    for (auto &el : arr) {
        el = idist(mt_engine);
        std::cout << el << " ";
    }
    std::cout << std::endl;

    {
        auto buf = sycl::buffer<int32_t, 1>(arr.data(), sycl::range<1>(arr.size()));

        queue.submit([&] (sycl::handler &cgh) {
            sycl_inclusive_scan(cgh, sycl_decide_range(buf.size()), buf);
        }).wait_and_throw();
    }

    std::cout << "Scanned: ";
    for (auto const &el : arr) {
        std::cout << el << " ";
    }
    std::cout << std::endl;

    return 0;
}
