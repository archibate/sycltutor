#pragma once

#include <sycl/sycl.hpp>

template <class T>
void print_buffer(std::vector<T> const &buf) {
    std::cout << "CPU [ ";
    if (buf.size() > 1024) {
        for (size_t i = 0; i < 128; i++) {
            std::cout << buf[i] << ' ';
        }
        std::cout << "... ";
        for (size_t i = buf.size() - 128; i < buf.size(); i++) {
            std::cout << buf[i] << ' ';
        }
    } else {
        for (size_t i = 0; i < buf.size(); i++) {
            std::cout << buf[i] << ' ';
        }
    }
    std::cout << "]\n";
}

template <class T>
void print_buffer(sycl::buffer<T> buf) {
    sycl::host_accessor<T> hacc{buf, sycl::read_only};
    std::cout << "GPU [ ";
    if (hacc.size() > 1024) {
        for (size_t i = 0; i < 128; i++) {
            std::cout << hacc[i] << ' ';
        }
        std::cout << "... ";
        for (size_t i = hacc.size() - 128; i < hacc.size(); i++) {
            std::cout << hacc[i] << ' ';
        }
    } else {
        for (size_t i = 0; i < hacc.size(); i++) {
            std::cout << hacc[i] << ' ';
        }
    }
    std::cout << "]\n";
}
