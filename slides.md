# 打倒 CUDA 暴政，世界属于 SyCL

# 概念辨析

```cpp
```

# 与 CUDA 的转换关系

```cpp
item.get_local_id(); // threadIdx
item.get_local_range(); // blockDim
item.get_group().get_group_id(); // blockIdx
item.get_group_range(); // gridDim
item.get_global_id(); // blockDim * blockIdx + threadIdx
item.get_global_range(); // blockDim * gridDim
```

# 与 CUDA 的转换关系

| CUDA | SyCL |
|-|-|
| .cu | .cpp |
| kernel | kernel function |
| kernel<<<dim3, dim3>>> | parallel_for(nd_range, lambda) |
| thread | work-item |
| threadIdx | (work-item) local id |
| blockIdx | work-group id |
| blockDim * blockIdx + threadIdx | (work-item) global id |
| __shared__ | (work-group) local memory |
| __syncthread | work-group barrier |
| thread-local storage | (work-item) private memory |
| block | work-group |
| blockDim | work-group range |
| grid | command |
| gridDim | work-group range |
| int3 | id |
| dim3 | range |
| <<<dim3, dim3>>> | nd_range |
| blockDim * blockIdx + threadIdx | item |
| blockIdx, threadIdx | nd_item |
| cudaTexture_t | image |
| cudaStream_t | queue |
| Unified Memory | USM |
