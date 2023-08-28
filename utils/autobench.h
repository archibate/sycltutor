#pragma once

#include <type_traits>
#include <vector>
#include <random>
#include <iostream>
#include <benchmark/benchmark.h>
#include "wangshash.h"

namespace _makeAutoBench_details {

template <class T>
static std::enable_if_t<std::is_integral_v<T>> do_randomize(std::vector<T> &arr, T min, T max, uint32_t seed) {
    wangshash rng{seed};
    #pragma omp parallel for
    for (auto &a: arr) {
        a = std::uniform_int_distribution<T>{min, max}(rng);
    }
}

template <class T>
static std::enable_if_t<std::is_floating_point_v<T>> do_randomize(std::vector<T> &arr, T min, T max, uint32_t seed) {
    wangshash rng{seed};
    #pragma omp parallel for
    for (auto &a: arr) {
        a = std::uniform_real_distribution<T>{min, max}(rng);
    }
}

template <class T>
struct AutoBenchArg {
    T inner;

    using argument_type = T;

    T get() {
        return inner;
    }

    void init(T v, size_t) {
        inner = v;
    }

    bool test(AutoBenchArg const &, size_t, size_t) {
        return true;
    }
};

template <class T>
struct AutoBenchArg<T &> {
    T *inner;

    using argument_type = std::reference_wrapper<T>;

    T &get() {
        return *inner;
    }

    void init(std::reference_wrapper<T> v, size_t) {
        inner = &v.get();
    }

    bool test(AutoBenchArg const &, size_t, size_t) {
        return true;
    }
};

template <class T>
struct AutoBenchArg<T *> {
    std::vector<T> inner;

    using argument_type = size_t;

    T *get() {
        return inner.data();
    }

    void init(size_t n, size_t) {
        inner.clear();
        inner.resize(n);
    }

    bool test(AutoBenchArg const &that, size_t funcid, size_t index) {
        if (inner.size() != that.inner.size()) {
            std::cerr << "in function #" << funcid << ", argument #" << index << ": ";
            std::cerr << "size mismatched, got " << inner.size() << ", expect " << that.inner.size() << "\n";
            return false;
        }
        for (size_t i = 0; i < inner.size(); i++) {
            if (inner[i] != that.inner[i]) {
                std::cerr << "in function #" << funcid << ", argument #" << index << ": ";
                std::cerr << "element " << i << " mismatched, got " << inner[i] << ", expect " << that.inner[i] << "\n";
                std::cerr << "[ ";
                for (size_t j = i > 5 ? i - 5 : 0; j < std::min(inner.size(), i + 5); j++) {
                    std::cerr << inner[j] << ' ';
                }
                std::cerr << "]\n";
                std::cerr << "[ ";
                for (size_t j = i > 5 ? i - 5 : 0; j < std::min(inner.size(), i + 5); j++) {
                    std::cerr << that.inner[j] << ' ';
                }
                std::cerr << "]\n";
                return false;
            }
        }
        return true;
    }
};

template <class T>
struct AutoBenchArg<T const *> {
    std::vector<T> inner;

    using argument_type = std::tuple<size_t, T, T>;

    T const *get() {
        return inner.data();
    }

    void init(std::tuple<size_t, T, T> arg, size_t index) {
        auto [n, min, max] = arg;
        inner.resize(n);
        do_randomize(inner, min, max, (uint32_t)index);
    }

    bool test(AutoBenchArg const &, size_t, size_t) {
        return true;
    }
};

template <size_t N, class F>
struct AutoBench;

template <size_t N, class ...Ts>
struct AutoBench<N, void(Ts...)> {
    mutable std::array<void(*)(Ts...), N> fps;
    mutable std::tuple<AutoBenchArg<Ts>...> args;

    explicit AutoBench(std::array<void(*)(Ts...), N> fps) : fps(std::move(fps)) {}

    template <size_t ...Is>
    void _impl_add_arguments(std::index_sequence<Is...>,
                         std::conditional_t<false, Ts, typename AutoBenchArg<Ts>::argument_type> ...argvals) const {
        ((void)std::get<Is>(args).init(argvals, Is), ...);
    }

    template <size_t ...Is>
    bool _impl_test(std::index_sequence<Is...>, size_t i) const {
        std::tuple<AutoBenchArg<Ts>...> stdargs = args;
        fps[0](std::get<Is>(stdargs).get()...);
        fps[i](std::get<Is>(args).get()...);
        return (std::get<Is>(args).test(std::get<Is>(stdargs), i, Is) && ...);
    }

    template <size_t ...Is>
    void _impl_run(::benchmark::State &s, size_t i, std::index_sequence<Is...>) const {
        auto *const fp = fps[i];
        for (auto _: s) {
            fp(std::get<Is>(args).get()...);
        }
    }
};

template <class Lambda>
class LambdaBenchmark : public ::benchmark::internal::Benchmark {
 public:
  void Run(::benchmark::State& st) BENCHMARK_OVERRIDE { lambda_(st); }

  template <class OLambda>
  LambdaBenchmark(const std::string& name, OLambda&& lam)
      : Benchmark(name), lambda_(std::forward<OLambda>(lam)) {}

  LambdaBenchmark(LambdaBenchmark const&) = delete;

  Lambda lambda_;
};

template <class T>
struct unwrap_reference {
    using type = T;
};

template <class T>
struct unwrap_reference<std::reference_wrapper<T>> {
    using type = T &;
};

}

template <size_t N, class ...Ts>
int doAutoBench(std::string const &title,
    std::array<void(*)(Ts...), N> fps,
    std::conditional_t<false, Ts, typename
    _makeAutoBench_details::AutoBenchArg<Ts>::argument_type>
    ...argvals) {
    _makeAutoBench_details::AutoBench<N, void(
        typename _makeAutoBench_details::unwrap_reference<Ts>::type...)> ab{std::move(fps)};
    std::make_index_sequence<sizeof...(Ts)> iseq;
    ab._impl_add_arguments(iseq, argvals...);
    for (size_t i = 0; i < N; i++) {
        auto bmfunc = [=] (::benchmark::State &s) {
            /* bool testok = i == 0 || ( */
            /*     ab._impl_add_arguments(iseq, argvals...), */
            /*     ab._impl_test(iseq, i)); */
            /* if (!testok) s.SkipWithError("test failed"); */
            ab._impl_add_arguments(iseq, argvals...);
            ab._impl_run(s, i, iseq);
        };
        ::benchmark::internal::RegisterBenchmarkInternal(
            new _makeAutoBench_details::LambdaBenchmark<decltype(bmfunc)>(
                title + "_" + std::to_string(i), std::move(bmfunc)));
    }
    return 0;
}
