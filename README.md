# Building

First of all, could you uninstall Wendous? OpenSYCL's CUDA backend is now only available on Linux.

## Install OpenSYCL

Please read their official documents: https://github.com/OpenSYCL/OpenSYCL/blob/develop/doc/install.md and https://github.com/OpenSYCL/OpenSYCL/blob/develop/doc/install-cuda.md

TL;DR? Compile and install OpenSYCL to `/usr/local` (may specify `-DCMAKE_INSTALL_PREFIX=/opt/sycl-0.9.4` for custom install location):

```bash
git clone https://github.com/OpenSYCL/OpenSYCL --depth=1
cd OpenSYCL
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
sudo cmake --build build --target install
```

Check if OpenSYCL installed successfully:

```bash
which syclcc               # /usr/local/bin/syclcc
syclcc --opensycl-version  # 0.9.4
```

## Building my dear project

```bash
git clone https://github.com/archibate/sycltest
cd sycltest
export OPENSYCL_TARGETS='cuda:sm_75'  # must export this before -B, otherwise you have to rm -rf build to update this variable
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

> If you installed OpenSYCL to `/opt/sycl-0.9.4` in the previous step, you may need to `export OpenSYCL_DIR=/opt/sycl-0.9.4/lib/cmake/OpenSYCL` here too.

## Running the project

Running on CPU backend:

```
Data: 9 7 4 7 8 1 9 10 10 5 9 0 7 4 5 4 
running on device: hipSYCL OpenMP host device
18446744073709551615 bytes of local memory  # so this is (size_t)-1
Sum: 99
```

Running on CUDA backend:

```
Data: 5 1 4 10 3 2 3 7 5 7 2 6 6 2 0 3 
running on device: NVIDIA GeForce RTX 2080 with Max-Q Design
49152 bytes of local memory
Sum: 66  # yeah he used random seed per launch
```

Running on AMD ROCm:

```
I don't have an Y.E.S. device, I have N.O. devices, but you may try export OPENSYCL_TARGETS="hip:gfx906"
```

# Troubleshooting

## `noinline` error

If you got `__attribute__((noinline))`` errors like:

```
In file included from /usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/13.1.1/../../../../include/c++/13.1.1/string:55:
/usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/13.1.1/../../../../include/c++/13.1.1/bits/basic_string.tcc:473:20: error: expected expression
/opt/cuda/include/crt/host_defines.h:83:33: note: expanded from macro '__noinline__'
        __attribute__((noinline))
                                ^
```

You should modify the file `/usr/lib/clang/*.*.*/include/__clang_cuda_runtime_wrapper.h`:

```diff
#if CUDA_VERSION < 9000
#include "crt/device_runtime.h"
#endif
#include "crt/host_runtime.h"
+#undef __noinline__
// device_runtime.h defines __cxa_* macros that will conflict with
// cxxabi.h.
```

> So you are 'Okay' with this 'diff' view, right?

## `texture` error

If you got `texture` compile errors (this occurs when using CUDA 12):

```
In file included from /usr/lib/clang/15.0.7/include/__clang_cuda_runtime_wrapper.h:365:
/usr/lib/clang/15.0.7/include/__clang_cuda_texture_intrinsics.h:696:13: error: no template named 'texture'
            texture<__DataT, __TexT, cudaReadModeNormalizedFloat> __handle,
            ^
/usr/lib/clang/15.0.7/include/__clang_cuda_texture_intrinsics.h:709:13: error: no template named 'texture'
            texture<__DataT, __TexT, cudaReadModeElementType> __handle,
            ^
```

You should modify the file `/usr/lib/clang/*.*.*/include/__clang_cuda_runtime_wrapper.h` (again):

```diff
-#if __cplusplus >= 201103L && CUDA_VERSION >= 9000
+#if __cplusplus >= 201103L && CUDA_VERSION >= 9000 && CUDA_VERSION < 12000
```

Source: https://discourse.llvm.org/t/compiling-cuda-programs-with-clang-14-15/70013/2

## `format` error

```
/usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/13.1.1/../../../../include/c++/13.1.1/format:3538:30: note: hidden overloaded virtual function 'std::__format::_Scanner<wchar_t>::_M_format_arg' declared here
      constexpr virtual void _M_format_arg(size_t __id) = 0;
                             ^
```

The stupid CUDA compiler still doesn't support C++20, just keep `set(CMAKE_CXX_STANDARD 17)`.

## program stucks

So your program starts and stucks forever? Without printing anything anymore?

This seems to be an Intel bug (thanks man, DickAPI). Try removing this file:

```bash
/etc/OpenCL/vendors/intel-oneapi-compiler-shared-opencl-cpu.icd
```

For Arch / Manjaro:

```bash
pacman -R intel-oneapi-compiler-shared-opencl-cpu
```

Source: https://forum.manjaro.org/t/clinfo-and-codes-using-opencl-hangs-forever-after-last-update/138541

# References

- SYCL official guide (cheetsheet): https://www.khronos.org/files/sycl/sycl-2020-reference-guide.pdf
- SYCL official specification: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
- Parallel STL based on SYCL: https://github.com/KhronosGroup/SyclParallelSTL
