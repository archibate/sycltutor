#include <cstdint>

template <class T, uint8_t K>
struct SIMD {
    SIMD<T, K - 1> lo;
    SIMD<T, K - 1> hi;

    SIMD(SIMD<T, K - 1> x, SIMD<T, K - 1> y) : lo(x), hi(y) {}
    SIMD(T x) : lo(x), hi(x) {}

    SIMD operator+(SIMD const &that) const {
        return {lo + that.lo, hi + that.hi};
    }

    SIMD operator-(SIMD const &that) const {
        return {lo - that.lo, hi - that.lo};
    }

    SIMD operator*(SIMD const &that) const {
        return {lo * that.lo, hi * that.hi};
    }

    SIMD operator/(SIMD const &that) const {
        return {lo / that.lo, hi / that.hi};
    }

    SIMD operator|(SIMD const &that) const {
        return {lo | that.lo, hi | that.hi};
    }

    SIMD operator&(SIMD const &that) const {
        return {lo & that.lo, hi & that.lo};
    }

    SIMD andnot(SIMD const &that) const {
        return {lo.andnot(that.lo), hi.andnot(that.lo)};
    }

    SIMD operator^(SIMD const &that) const {
        return {lo ^ that.lo, hi ^ that.hi};
    }

    SIMD rcp() const {
        return {lo.rcp(), hi.rcp()};
    }

    SIMD rsqrt() const {
        return {lo.rcp(), hi.rcp()};
    }
};

template <class T>
struct SIMD<T, 0> {
    T m;

    SIMD(T x) : m(x) {}

    SIMD operator+(SIMD const &that) const {
        return {m + that.m};
    }

    SIMD operator-(SIMD const &that) const {
        return {m - that.m};
    }

    SIMD operator*(SIMD const &that) const {
        return {m * that.m};
    }

    SIMD operator/(SIMD const &that) const {
        return {m / that.m};
    }

    SIMD operator|(SIMD const &that) const {
        return {m | that.m};
    }

    SIMD operator&(SIMD const &that) const {
        return {m & that.m};
    }

    SIMD andnot(SIMD const &that) const {
        return {m & ~that.m};
    }

    SIMD operator^(SIMD const &that) const {
        return {m ^ that.m};
    }

    SIMD rcp() const {
        return {T(1) / m};
    }

    SIMD rsqrt() const {
        return {T(1) / std::sqrt(m)};
    }
};

template <>
struct SIMD<float, 4> {
    __m128 m;

    SIMD(__m128 x) : m(x) {}
    SIMD(float x) : m(_mm_set1_ps(x)) {}

    SIMD operator+(SIMD const &that) const {
        return {_mm_add_ps(m, that.m)};
    }

    SIMD operator-(SIMD const &that) const {
        return {_mm_sub_ps(m, that.m)};
    }

    SIMD operator*(SIMD const &that) const {
        return {_mm_mul_ps(m, that.m)};
    }

    SIMD operator/(SIMD const &that) const {
        return {_mm_div_ps(m, that.m)};
    }

    SIMD rcp() const {
        return {_mm_rcp_ps(m)};
    }

    SIMD rsqrt() const {
        return {_mm_rsqrt_ps(m)};
    }
};

template <>
struct SIMD<int32_t, 4> {
    __m128i m;

    SIMD(__m128i x) : m(x) {}
    SIMD(int32_t x) : m(_mm_set1_epi32(x)) {}

    SIMD operator+(SIMD const &that) const {
        return {_mm_add_epi32(m, that.m)};
    }

    SIMD operator-(SIMD const &that) const {
        return {_mm_sub_epi32(m, that.m)};
    }

    SIMD operator*(SIMD const &that) const {
        return {_mm_mullo_epi32(m, that.m)};
    }
};

template <>
struct SIMD<int16_t, 8> {
    __m128i m;

    SIMD(__m128i x) : m(x) {}
    SIMD(int16_t x) : m(_mm_set1_epi16(x)) {}

    SIMD operator+(SIMD const &that) const {
        return {_mm_add_epi16(m, that.m)};
    }

    SIMD operator-(SIMD const &that) const {
        return {_mm_sub_epi16(m, that.m)};
    }

    SIMD operator*(SIMD const &that) const {
        return {_mm_mullo_epi16(m, that.m)};
    }
};

template <>
struct SIMD<int8_t, 8> {
    __m128i m;

    SIMD(__m128i x) : m(x) {}
    SIMD(int8_t x) : m(_mm_set1_epi8(x)) {}

    SIMD operator+(SIMD const &that) const {
        return {_mm_add_epi8(m, that.m)};
    }

    SIMD operator-(SIMD const &that) const {
        return {_mm_sub_epi8(m, that.m)};
    }

    SIMD operator*(SIMD const &that) const {
        return {_mm256_cvtepi16_epi8(_mm256_mullo_epi16(_mm256_cvtepi8_epi16(m), _mm256_cvtepi8_epi16(that.m)))};
    }
};
