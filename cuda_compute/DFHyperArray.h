// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#include "DFCudaCodes.h"

#define HYPER_ARRAY_MAX_DIMS 4
namespace dexsim {
namespace cudamgr {
using HyperArrayHook = void*;

// TODO: implement Reshape
struct ArrayShape {
    size_t dims[HYPER_ARRAY_MAX_DIMS];

    ArrayShape() : dims() {}

    inline size_t operator[](size_t i) const { return dims[i]; }
    inline size_t& operator[](size_t i) { return dims[i]; }
};

template <typename T>
struct SharedDataCPU {
    T* value_ = nullptr;
    int semaphore_ = 1;
    bool is_allocated_ = false;
};

struct SharedDataGPU {
    CUdeviceptr value_ = (CUdeviceptr) nullptr;
    int semaphore_ = 1;
    bool is_allocated_ = false;
};

template <typename T>
struct HyperArray {
    HyperArray(size_t dim0) {
        shape_[0] = dim0;
        ndim_ = 1;
        strides_[0] = sizeof(T);
        strides_[1] = 0;
        strides_[2] = 0;
        strides_[3] = 0;
        size_ = dim0;
    }
    HyperArray(size_t dim0, size_t dim1) {
        shape_[0] = dim0;
        shape_[1] = dim1;
        ndim_ = 2;
        strides_[0] = sizeof(T) * dim1;
        strides_[1] = sizeof(T);
        strides_[2] = 0;
        strides_[3] = 0;
        size_ = dim0 * dim1;
    }
    HyperArray(size_t dim0, size_t dim1, size_t dim2) {
        shape_[0] = dim0;
        shape_[1] = dim1;
        shape_[2] = dim2;
        ndim_ = 3;
        strides_[0] = sizeof(T) * dim1 * dim2;
        strides_[1] = sizeof(T) * dim2;
        strides_[2] = sizeof(T);
        strides_[3] = 0;
        size_ = dim0 * dim1 * dim2;
    }
    HyperArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
        shape_[0] = dim0;
        shape_[1] = dim1;
        shape_[2] = dim2;
        shape_[3] = dim3;
        ndim_ = 4;
        strides_[0] = sizeof(T) * dim1 * dim2 * dim3;
        strides_[1] = sizeof(T) * dim2 * dim3;
        strides_[2] = sizeof(T) * dim3;
        strides_[3] = sizeof(T);
        size_ = dim0 * dim1 * dim2 * dim3;
    }

    SharedDataCPU<T>* cpu_data_ = nullptr;
    SharedDataGPU* gpu_data_ = nullptr;
    ArrayShape shape_;
    size_t strides_[HYPER_ARRAY_MAX_DIMS];
    size_t ndim_;
    size_t size_;
};

}  // namespace cudamgr
}  // namespace dexsim