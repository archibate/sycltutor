#pragma once

#include <chrono>
#include <iostream>

#define TICK(name) auto bench_##name = std::chrono::steady_clock::now();
#define TOCK(name) std::cerr<<#name": "<<std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-bench_##name).count()<<"秒\n";
#define TOCKS(name,times) auto bench_##name = std::chrono::steady_clock::now();for(int times_##name=0;times_##name<(times);times++){}std::cerr<<#name": "<<std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-bench_##name).count()<<"次/秒\n";
