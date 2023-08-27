cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

if (DEFINED ENV{DPCPP_ROOT} AND NOT DEFINED DPCPP_ROOT)
    set(DPCPP_ROOT $ENV{DPCPP_ROOT})
endif()
if (DEFINED DPCPP_ROOT)
    set(CMAKE_CXX_COMPILER ${DPCPP_ROOT}/bin/icpx)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
endif()

project(main LANGUAGES CXX)

if (DEFINED DPCPP_ROOT)
    message(STATUS "Compiling SyCL using Intel DPC++ (${DPCPP_ROOT}/bin/dpcpp)")
    macro(add_sycl_to_target)
    endmacro()
else()
    message(STATUS "Compiling SyCL using OpenSYCL")
    find_package(OpenSYCL CONFIG REQUIRED)
    if(NOT OPENSYCL_DEBUG_LEVEL)
      if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(OPENSYCL_DEBUG_LEVEL 3 CACHE STRING "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)" FORCE)
      else()
        set(OPENSYCL_DEBUG_LEVEL 2 CACHE STRING "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)" FORCE)
      endif()
    endif()
    add_compile_options(-DHIPSYCL_DEBUG_LEVEL=${OPENSYCL_DEBUG_LEVEL})
endif()

add_executable(main main.cpp)
add_sycl_to_target(TARGET main SOURCES main.cpp)

find_package(benchmark CONFIG REQUIRED)
target_link_libraries(main PUBLIC
    benchmark::benchmark benchmark::benchmark_main)

find_package(TBB CONFIG REQUIRED)
target_link_libraries(main PUBLIC TBB::tbb)

target_compile_options(main PUBLIC
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-mavx2$<SEMICOLON>-mfma>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/arch:AVX2>)

# find_package(OpenMP)
# if (TARGET OpenMP::OpenMP_CXX)
#     target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
# else()
#     message(WARNING "OpenMP for C++ not found")
# endif()
