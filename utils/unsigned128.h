#pragma once

#include <algorithm>

struct unsigned128 {
    unsigned data[4];

    void clear() {
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
        data[3] = 0;
    }

    void set(int ii) {
        data[ii >> 5] |= 1u << (ii & 0x1f);
    }

    unsigned128 &operator&=(unsigned128 const &that) {
        data[0] &= that.data[0];
        data[1] &= that.data[1];
        data[2] &= that.data[2];
        data[3] &= that.data[3];
        return *this;
    }

    unsigned128 operator&(unsigned128 const &that) const {
        unsigned128 tmp = *this;
        tmp &= that;
        return tmp;
    }

    unsigned128 &operator|=(unsigned128 const &that) {
        data[0] |= that.data[0];
        data[1] |= that.data[1];
        data[2] |= that.data[2];
        data[3] |= that.data[3];
        return *this;
    }

    unsigned128 operator|(unsigned128 const &that) const {
        unsigned128 tmp = *this;
        tmp |= that;
        return tmp;
    }

    int popclo(int ii) {
        return
        __builtin_popcount(data[0] & (ii >= 32 ? 0xffffffffu : (1u << ii) - 1)) +
        __builtin_popcount(data[1] & (ii >= 64 ? 0xffffffffu : (1u << std::max(ii - 32, 0)) - 1)) +
        __builtin_popcount(data[2] & (ii >= 96 ? 0xffffffffu : (1u << std::max(ii - 64, 0)) - 1)) +
        __builtin_popcount(data[3] & ((1u << std::max(ii - 96, 0)) - 1));
    }

    int popc() {
        return
        __builtin_popcount(data[0]) +
        __builtin_popcount(data[1]) +
        __builtin_popcount(data[2]) +
        __builtin_popcount(data[3]);
    }

    int ctz() {
        return
        data[0] ? __builtin_ctz(data[0]) :
        data[1] ? __builtin_ctz(data[1]) + 32 :
        data[2] ? __builtin_ctz(data[2]) + 64 :
        __builtin_ctz(data[3]) + 96;
    }

    int ffs() {
        return
        data[0] ? __builtin_ffs(data[0]) :
        data[1] ? __builtin_ffs(data[1]) + 32 :
        data[2] ? __builtin_ffs(data[2]) + 64 :
        __builtin_ffs(data[3]) + 96;
    }
};
