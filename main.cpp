#include <sycl/sycl.hpp>
#include <benchmark/benchmark.h>

using namespace sycl;

template <class HistT, class ValT>
void sycl_histogram(HistT *hist_p, size_t hist_n, ValT const *val_p, size_t val_n) {
    constexpr size_t k_block_size = 256;
    constexpr size_t k_num_bins = 256;
    static queue q{gpu_selector_v};
    buffer hist_buf{hist_p, range<1>{hist_n}};
    buffer val_buf{val_p, range<1>{val_n}};
    q.submit([&] (handler &cgh) {
        accessor hist_axr{hist_buf, cgh, read_write};
        accessor val_axr{val_buf, cgh, read_only};
        cgh.parallel_for(nd_range<1>{val_n / k_block_size, 64}, [=] (nd_item<1> it) {
            auto wg = it.get_group();
            auto wg_id = wg.get_group_id()[0];
            auto wg_size = wg.get_local_range()[0];
            auto sg = it.get_sub_group();
            auto sg_id = sg.get_group_id()[0];
            auto sg_size = sg.get_local_range()[0];
            HistT hist_priv[k_num_bins / 16];
            for (size_t j = 0; j < k_num_bins / 16; j++) {
                hist_priv[j] = HistT{};
            }
            for (size_t j = 0; j < k_block_size; j++) {
                ValT v0 = val_axr[wg_id * wg_size * k_block_size
                    + sg_id * sg_size * k_block_size
                    + sg_size * j];
                size_t v = v0 % k_num_bins;
                #pragma unroll
                for (size_t i1 = 0; i1 < sg_size; i1++) {
                    auto c = group_broadcast(sg, v, i1);
                    if (sg.get_local_id()[0] == (c & 0xf)) {
                        hist_priv[c >> 4] += HistT{1};
                    }
                }
            }
            for (size_t j = 0; j < k_num_bins / 16; j++) {
                sycl::atomic_ref<HistT, memory_order_relaxed, memory_scope_device>{hist_axr[16 * j + sg.get_local_id()[0]]}.fetch_add(hist_priv[j]);
            }
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

namespace _makeAutoBench_details {

template <class T>
static std::enable_if_t<std::is_integral_v<T>> do_randomize(std::vector<T> &arr, T min, T max) {
    static thread_local std::mt19937 rng;
    for (auto &a: arr) {
        a = std::uniform_int_distribution<T>{min, max}(rng);
    }
}

template <class T>
static std::enable_if_t<std::is_floating_point_v<T>> do_randomize(std::vector<T> &arr, T min, T max) {
    static thread_local std::mt19937 rng;
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

    void init(T v) {
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

    void init(size_t n) {
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

    void init(std::tuple<size_t, T, T> arg) {
        auto [n, min, max] = arg;
        inner.resize(n);
        do_randomize(inner, min, max);
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
        ((void)std::get<Is>(args).init(std::move(argvals)), ...);
    }

    template <size_t ...Is>
    bool _impl_test(std::index_sequence<Is...>) const {
        std::tuple<AutoBenchArg<Ts>...> stdargs = args;
        fps[0](std::get<Is>(stdargs).get()...);
        for (size_t i = 1; i < N; i++)
            if (((void)fps[i](std::get<Is>(args).get()...),
                !(std::get<Is>(args).test(std::get<Is>(stdargs), i, Is) && ...))
            ) return false;
        return true;
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
    if (!ab._impl_test(std::make_index_sequence<sizeof...(Ts)>{}))
        std::cerr << "test failed\n";
    for (size_t i = 0; i < N; i++) {
        auto bmfunc = [=] (::benchmark::State &s) {
            ab._impl_add_arguments(std::make_index_sequence<sizeof...(Ts)>{}, argvals...);
            ab._impl_run(s, i, std::make_index_sequence<sizeof...(Ts)>{});
        };
        ::benchmark::internal::RegisterBenchmarkInternal(
            new _makeAutoBench_details::LambdaBenchmark<decltype(bmfunc)>(title + "_" + std::to_string(i), std::move(bmfunc)));
    }
    return 0;
}

constexpr size_t n = 65536000;
constexpr size_t maxval = 256;

static int bench_histogram_uint32_t = doAutoBench("histogram", std::array{
    cpu_histogram<uint32_t, uint32_t>,
    sycl_histogram<uint32_t, uint32_t>,
}, maxval, maxval, {n, 0, maxval}, n);
