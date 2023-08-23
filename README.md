# References

SYCL official guide: https://www.khronos.org/files/sycl/sycl-2020-reference-guide.pdf

# Troubleshooting

## `texture` error

If you got `texture` compile errors (this occurs when using CUDA 12):

```cpp
In file included from /usr/lib/clang/15.0.7/include/__clang_cuda_runtime_wrapper.h:365:
/usr/lib/clang/15.0.7/include/__clang_cuda_texture_intrinsics.h:696:13: error: no template named 'texture'
            texture<__DataT, __TexT, cudaReadModeNormalizedFloat> __handle,
            ^
/usr/lib/clang/15.0.7/include/__clang_cuda_texture_intrinsics.h:709:13: error: no template named 'texture'
            texture<__DataT, __TexT, cudaReadModeElementType> __handle,
            ^
```

According to you should modify the file `/usr/lib/clang/*.*.*/include/__clang_cuda_runtime_wrapper.h`:

```diff
-#if __cplusplus >= 201103L && CUDA_VERSION >= 9000
+#if __cplusplus >= 201103L && CUDA_VERSION >= 9000 && CUDA_VERSION < 12000
```

Source: https://discourse.llvm.org/t/compiling-cuda-programs-with-clang-14-15/70013/2

## `noinline` error

If you got `__attribute__((noinline))`` errors like:

```cpp
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
// FIXME: redefine these as __device__ functions.
```
