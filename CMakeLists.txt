cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

# if (NOT DPCPP_ROOT)
    # set(DPCPP_ROOT /opt/intel/oneapi/compiler/2023.1.0/linux)
# endif()
# set(CMAKE_CXX_COMPILER ${DPCPP_ROOT}/bin/icpx)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")

project(main LANGUAGES CXX)

find_package(OpenSYCL CONFIG REQUIRED)
if(NOT OPENSYCL_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(OPENSYCL_DEBUG_LEVEL 3 CACHE STRING "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)" FORCE)
  else()
    set(OPENSYCL_DEBUG_LEVEL 2 CACHE STRING "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)" FORCE)
  endif()
endif()
add_compile_options(-DHIPSYCL_DEBUG_LEVEL=${OPENSYCL_DEBUG_LEVEL})

add_executable(main main.cpp)
target_include_directories(main PUBLIC ../SyclParallelSTL/include)
add_sycl_to_target(TARGET main SOURCES main.cpp)
