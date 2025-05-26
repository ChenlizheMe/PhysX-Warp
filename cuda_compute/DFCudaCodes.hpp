// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------

#pragma once
#include "DFCudaCodes.h"

#define CUDA_API_FUNC(name, DECL, ARGS) \
    CUDA_CODES(*PFN_##name) DECL;       \
    CUDA_CODES name DECL { return PFN_##name ARGS; }

namespace dexsim {
namespace cudamgr {

class CudaFunctionManager : public ICudaFunctionManager {
public:
    static CudaFunctionManager& instance() {
        static CudaFunctionManager instance;
        return instance;
    }

private:
    CudaFunctionManager();
    ~CudaFunctionManager();
    CudaFunctionManager(const CudaFunctionManager&) = delete;
    CudaFunctionManager& operator=(const CudaFunctionManager&) = delete;

    // Driver Management
    CUDA_API_FUNC(cuDriverGetVersion, (int* version), (version))
    CUDA_API_FUNC(cuInit, (unsigned int flags), (flags))
    CUDA_API_FUNC(
            cuGetProcAddress,
            (const char* symbol, void** pfn, int cudaVersion, uint64_t flags),
            (symbol, pfn, cudaVersion, flags))

    // Device Management
    CUDA_API_FUNC(cuDeviceGetCount, (int* count), (count))
    CUDA_API_FUNC(cuDeviceGet,
                  (CUdevice * device, int ordinal),
                  (device, ordinal))
    CUDA_API_FUNC(cuDeviceGetName,
                  (char* name, int len, CUdevice dev),
                  (name, len, dev))
    CUDA_API_FUNC(cuDeviceGetAttribute,
                  (int* pi, int attr, CUdevice dev),
                  (pi, attr, dev))

    // Context Management
    CUDA_API_FUNC(cuCtxCreate,
                  (CUcontext * pctx, unsigned int flags, CUdevice dev),
                  (pctx, flags, dev))
    CUDA_API_FUNC(cuCtxGetCurrent, (CUcontext * pctx), (pctx))
    CUDA_API_FUNC(cuCtxSynchronize, (), ())

    // Memory Management
    CUDA_API_FUNC(cuMemAlloc,
                  (CUdeviceptr * dptr, size_t bytesize),
                  (dptr, bytesize))
    CUDA_API_FUNC(cuMemFree, (CUdeviceptr dptr), (dptr))
    CUDA_API_FUNC(cuMemcpyHtoD,
                  (CUdeviceptr dstDevice,
                   const void* srcHost,
                   size_t ByteCount),
                  (dstDevice, srcHost, ByteCount))
    CUDA_API_FUNC(cuMemcpyDtoH,
                  (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount),
                  (dstHost, srcDevice, ByteCount))

    // Module and Kernel Control
    CUDA_API_FUNC(cuModuleLoadData,
                  (CUmodule * module, const void* image),
                  (module, image))
    CUDA_API_FUNC(cuModuleLoadDataEx,
                  (CUmodule * module,
                   const void* image,
                   unsigned int numOptions,
                   CUjit_option* options,
                   void** optionValues),
                  (module, image, numOptions, options, optionValues))
    CUDA_API_FUNC(cuModuleGetFunction,
                  (CUfunction * hfunc, CUmodule hmod, const char* name),
                  (hfunc, hmod, name))
    CUDA_API_FUNC(cuLaunchKernel,
                  (CUfunction f,
                   unsigned int gridDimX,
                   unsigned int gridDimY,
                   unsigned int gridDimZ,
                   unsigned int blockDimX,
                   unsigned int blockDimY,
                   unsigned int blockDimZ,
                   unsigned int sharedMemBytes,
                   CUstream hStream,
                   void** kernelParams,
                   void** extra),
                  (f,
                   gridDimX,
                   gridDimY,
                   gridDimZ,
                   blockDimX,
                   blockDimY,
                   blockDimZ,
                   sharedMemBytes,
                   hStream,
                   kernelParams,
                   extra))

    // Stream Management
    CUDA_API_FUNC(cuStreamCreate,
                  (CUstream * stream, unsigned int flags),
                  (stream, flags))
    CUDA_API_FUNC(cuStreamDestroy, (CUstream stream), (stream))
    CUDA_API_FUNC(cuStreamSynchronize, (CUstream stream), (stream))
    CUDA_API_FUNC(cuEventCreate,
                  (CUevent * event, unsigned int flags),
                  (event, flags))
    CUDA_API_FUNC(cuEventRecord,
                  (CUevent event, CUstream stream),
                  (event, stream))
    CUDA_API_FUNC(cuStreamWaitEvent,
                  (CUstream stream, CUevent event, unsigned int flags),
                  (stream, event, flags))
    CUDA_API_FUNC(cuCtxSetCurrent, (CUcontext ctx), (ctx))
    CUDA_API_FUNC(cuDevicePrimaryCtxRetain,
                  (CUcontext * pctx, CUdevice dev),
                  (pctx, dev))

    // Event Management
    CUDA_API_FUNC(cuEventDestroy, (CUevent event), (event))
    CUDA_API_FUNC(cuEventSynchronize, (CUevent event), (event))

    CUDA_API_FUNC(cuPointerGetAttribute,
                  (int* data, int attribute, CUdeviceptr ptr),
                  (data, attribute, ptr))

    // Error Handling
    CUDA_API_FUNC(cuGetErrorString,
                  (CUDA_CODES error, const char** pStr),
                  (error, pStr))
};

}  // namespace cudamgr
}  // namespace dexsim

#undef CUDA_API_FUNC