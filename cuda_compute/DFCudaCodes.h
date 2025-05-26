// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace dexsim {
namespace cudamgr {
#if defined(_WIN64) || defined(__LP64__)
using CUdeviceptr_v2 = unsigned long long;
#else
using CUdeviceptr_v2 = unsigned int;
#endif
using CUdeviceptr = CUdeviceptr_v2;

#define ICUDA_API(name, DECL, ARGS) \
    CUDA_CODES(*PFN_##name) DECL;   \
    virtual CUDA_CODES name DECL = 0;

enum CUDA_CODES {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CU_GET_PROC_ADDRESS_DEFAULT = 0,
    CU_ENABLE_DEFAULT = 0,
};
enum CUjit_option {
    CU_JIT_MAX_REGISTERS,
    CU_JIT_THREADS_PER_BLOCK,
    CU_JIT_WALL_TIME,
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_OPTIMIZATION_LEVEL,
    CU_JIT_TARGET_FROM_CUCONTEXT,
    CU_JIT_TARGET,
    CU_JIT_FALLBACK_STRATEGY,
    CU_JIT_GENERATE_DEBUG_INFO,
    CU_JIT_LOG_VERBOSE,
    CU_JIT_GENERATE_LINE_INFO,
    CU_JIT_CACHE_MODE
};

// CUDA types
using CUGraphicsResource_t = struct cudaGraphicsResource*;
using CUstream = struct CUstream_st*;
using CUdaArray_t = struct cudaArray_st*;
using CUcontext = struct CUctx_st*;

using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUdevice_v1 = int;
using CUevent = struct CUevent_st*;
using CUdevice = CUdevice_v1;

class ICudaFunctionManager {
public:
    ICUDA_API(cuDriverGetVersion, (int* version), (version))
    ICUDA_API(cuInit, (unsigned int flags), (flags))
    ICUDA_API(cuGetProcAddress,
              (const char* symbol, void** pfn, int cudaVersion, uint64_t flags),
              (symbol, pfn, cudaVersion, flags))

    // Device Management
    ICUDA_API(cuDeviceGetCount, (int* count), (count))
    ICUDA_API(cuDeviceGet, (CUdevice * device, int ordinal), (device, ordinal))
    ICUDA_API(cuDeviceGetName,
              (char* name, int len, CUdevice dev),
              (name, len, dev))
    ICUDA_API(cuDeviceGetAttribute,
              (int* pi, int attr, CUdevice dev),
              (pi, attr, dev))

    // Context Management
    ICUDA_API(cuCtxCreate,
              (CUcontext * pctx, unsigned int flags, CUdevice dev),
              (pctx, flags, dev))
    ICUDA_API(cuCtxGetCurrent, (CUcontext * pctx), (pctx))
    ICUDA_API(cuCtxSynchronize, (), ())

    // Memory Management
    ICUDA_API(cuMemAlloc,
              (CUdeviceptr * dptr, size_t bytesize),
              (dptr, bytesize))
    ICUDA_API(cuMemFree, (CUdeviceptr dptr), (dptr))
    ICUDA_API(cuMemcpyHtoD,
              (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount),
              (dstDevice, srcHost, ByteCount))
    ICUDA_API(cuMemcpyDtoH,
              (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount),
              (dstHost, srcDevice, ByteCount))

    // Module and Kernel Control
    ICUDA_API(cuModuleLoadData,
              (CUmodule * module, const void* image),
              (module, image))
    ICUDA_API(cuModuleLoadDataEx,
              (CUmodule * module, const void* image, unsigned int numOptions,
               CUjit_option* options, void** optionValues),
              (module, image, numOptions, options, optionValues))
    ICUDA_API(cuModuleGetFunction,
              (CUfunction * hfunc, CUmodule hmod, const char* name),
              (hfunc, hmod, name))
    ICUDA_API(cuLaunchKernel,
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

    // Stream and Event Management
    ICUDA_API(cuStreamCreate,
              (CUstream * stream, unsigned int flags),
              (stream, flags))
    ICUDA_API(cuStreamDestroy, (CUstream stream), (stream))
    ICUDA_API(cuStreamSynchronize, (CUstream stream), (stream))
    ICUDA_API(cuEventCreate,
              (CUevent * event, unsigned int flags),
              (event, flags))
    ICUDA_API(cuStreamWaitEvent,
              (CUstream stream, CUevent event, unsigned int flags),
              (stream, event, flags))
    ICUDA_API(cuCtxSetCurrent, (CUcontext ctx), (ctx))
    ICUDA_API(cuDevicePrimaryCtxRetain,
              (CUcontext * pctx, CUdevice dev),
              (pctx, dev))

    // Event
    ICUDA_API(cuEventRecord, (CUevent event, CUstream stream), (event, stream))
    ICUDA_API(cuEventDestroy, (CUevent event), (event))
    ICUDA_API(cuEventSynchronize, (CUevent event), (event))

    // Pointer Attributes
    ICUDA_API(cuPointerGetAttribute,
              (int* data, int attribute, CUdeviceptr ptr),
              (data, attribute, ptr))

    // Error Handling
    ICUDA_API(cuGetErrorString,
              (CUDA_CODES error, const char** pStr),
              (error, pStr))
};
extern "C" void cudaCodesMgr(ICudaFunctionManager** mgr);
}  // namespace cudamgr
}  // namespace dexsim