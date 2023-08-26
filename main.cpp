#include <sycl/sycl.hpp>
#include <benchmark/benchmark.h>

using namespace sycl;

template <class HistT, class ValT>
void sycl_histogram(HistT *hist_p, size_t hist_n, ValT const *val_p, size_t val_n, int times) {
    constexpr size_t k_block_size = 256;
    constexpr size_t k_num_bins = 256;
    constexpr size_t k_group_size = 128;
    static queue q{gpu_selector_v};
    buffer hist_buf{hist_p, range<1>{hist_n}};
    buffer val_buf{val_p, range<1>{val_n}};
    for (int i = 0; i < times; i++) {
        q.submit([&] (handler &cgh) {
            accessor hist_axr{hist_buf, cgh, read_write};
            accessor val_axr{val_buf, cgh, read_only};
            local_accessor<HistT> hist_loc{k_num_bins, cgh};
            cgh.parallel_for(nd_range<1>{((val_n + k_block_size - 1) / k_block_size + k_group_size - 1) / k_group_size * k_group_size, k_group_size}, [=] (nd_item<1> it) {
#if 1
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
#else
                size_t group = it.get_group()[0];
                size_t gSize = it.get_local_range()[0];
                auto sg = it.get_sub_group();
                size_t sgSize = sg.get_local_range()[0];
                size_t sgGroup = sg.get_group_id()[0];

                size_t factor = k_num_bins / gSize;
                size_t local_id = it.get_local_id()[0];
                if (factor <= 1 && local_id < k_num_bins) {
                    atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::local_space>{hist_loc[local_id]}.store(HistT{0});
                } else {
                    for (size_t k = 0; k < factor; k++) {
                        atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::local_space>{hist_loc[gSize * k + local_id]}.store(HistT{0});
                    }
                }
                it.barrier(access::fence_space::local_space);
                auto vbase = (group * gSize + sgGroup * sgSize) * k_block_size;
                for (size_t k = 0; k < k_block_size; k++) {
                    auto vid = vbase + sgSize * k;
                    if (vid < val_axr.size()) {
                        auto v = val_axr[vid] % k_num_bins;
                        atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::local_space>{hist_loc[v]}.fetch_add(HistT{1});
                    }
                }
                it.barrier(access::fence_space::local_space);
                if (factor <= 1 && local_id < k_num_bins) {
                    auto h = atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::local_space>{hist_loc[local_id]}.load();
                    atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::global_space>{hist_axr[local_id]}.fetch_add(h);
                } else {
                    for (size_t k = 0; k < factor; k++) {
                        auto v = val_axr[
                            (group * gSize + sgGroup * sgSize) * k_block_size
                            + sgSize * k] % k_num_bins;
                        auto h = atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::local_space>{hist_loc[gSize * k + local_id]}.load();
                        atomic_ref<HistT, memory_order_relaxed, memory_scope_device, access::address_space::global_space>{hist_axr[gSize * k + local_id]}.fetch_add(h);
                    }
                }
#endif
            });
        }).wait_and_throw();
    }
}

template <class HistT, class ValT>
void cpu_histogram(HistT *hist_p, size_t hist_n, ValT const *val_p, size_t val_n, int times) {
    memset(hist_p, 0, hist_n * sizeof(HistT));
    for (int i = 0; i < times; i++) {
        for (size_t i = 0; i < val_n; i++) {
            ++hist_p[val_p[i]];
        }
    }
}

namespace _makeAutoBench_details {

template <class T>
static std::enable_if_t<std::is_integral_v<T>> do_randomize(std::vector<T> &arr, T min, T max, uint32_t seed) {
    std::mt19937 rng{seed};
    for (auto &a: arr) {
        a = std::uniform_int_distribution<T>{min, max}(rng);
    }
}

template <class T>
static std::enable_if_t<std::is_floating_point_v<T>> do_randomize(std::vector<T> &arr, T min, T max, uint32_t seed) {
    std::mt19937 rng{seed};
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
        ((void)std::get<Is>(args).init(std::move(argvals), Is), ...);
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

}

template <size_t N, class ...Ts>
int doAutoBench(std::string const &title,
    std::array<void(*)(Ts...), N> fps,
    std::conditional_t<false, Ts, typename
    _makeAutoBench_details::AutoBenchArg<Ts>::argument_type>
    ...argvals) {
    _makeAutoBench_details::AutoBench<N, void(Ts...)> ab{std::move(fps)};
    ab._impl_add_arguments(std::make_index_sequence<sizeof...(Ts)>{}, argvals...);
    for (size_t i = 0; i < N; i++) {
        auto bmfunc = [=] (::benchmark::State &s) {
            bool testok = i == 0 || (
                ab._impl_add_arguments(std::make_index_sequence<sizeof...(Ts)>{}, argvals...),
                ab._impl_test(std::make_index_sequence<sizeof...(Ts)>{}, i));
            if (!testok) s.SkipWithError("test failed");
            ab._impl_add_arguments(std::make_index_sequence<sizeof...(Ts)>{}, argvals...);
            ab._impl_run(s, i, std::make_index_sequence<sizeof...(Ts)>{});
        };
        ::benchmark::internal::RegisterBenchmarkInternal(
            new _makeAutoBench_details::LambdaBenchmark<decltype(bmfunc)>(title + "_" + std::to_string(i), std::move(bmfunc)));
    }
    return 0;
}

constexpr size_t n = 65536 * 64;
constexpr size_t maxval = 256;

static int bench_histogram_uint32_t = doAutoBench("histogram", std::array{
    cpu_histogram<uint32_t, uint32_t>,
    sycl_histogram<uint32_t, uint32_t>,
}, maxval, maxval, {n, 0, maxval - 1}, n, 100);
