cmake_minimum_required(VERSION 3.10)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SYCL_COMPILER IntelSYCL CACHE STRING "Choices are: IntelSYCL, OpenSYCL")
if (SYCL_COMPILER MATCHES "IntelSYCL")
    set(CMAKE_CXX_COMPILER "icpx" CACHE STRING "IntelSYCL must use icpx as C++ compiler" FORCE)
endif()

project(main LANGUAGES CXX)

message(STATUS "Compiling SyCL using ${SYCL_COMPILER}")
find_package(${SYCL_COMPILER} CONFIG REQUIRED)
if(SYCL_COMPILER MATCHES "OpenSYCL")
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(OPENSYCL_DEBUG_LEVEL 3 CACHE STRING "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)" FORCE)
    else()
        set(OPENSYCL_DEBUG_LEVEL 2 CACHE STRING "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)" FORCE)
    endif()
    add_compile_options(-DHIPSYCL_DEBUG_LEVEL=${OPENSYCL_DEBUG_LEVEL})
endif()

add_executable(main main.cpp)
add_sycl_to_target(TARGET main SOURCES main.cpp)

find_package(benchmark CONFIG)
if (benchmark_FOUND)
    target_link_libraries(main PUBLIC
        benchmark::benchmark benchmark::benchmark_main)
endif()

target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(TBB CONFIG REQUIRED)
target_link_libraries(main PUBLIC TBB::tbb)

target_compile_options(main PUBLIC
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-march=native>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-march=native>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/arch:AVX2>)

# find_package(OpenMP)
# if (TARGET OpenMP::OpenMP_CXX)
#     target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
# else()
#     message(WARNING "OpenMP for C++ not found")
# endif()
