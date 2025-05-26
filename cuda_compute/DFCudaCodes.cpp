// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#include "DFCudaCodes.hpp"

#if defined(__unix__) || defined(__unix)
#include <dlfcn.h>
#include <link.h>
#include <cstdio>
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define dlsym GetProcAddress
#endif

#define LOAD_CUDA_FUNCTION(name, version)                    \
    PFN_##name = reinterpret_cast<decltype(PFN_##name)>(     \
            dlsym(cuda_lib, #name version));                 \
    if (!PFN_##name) {                                       \
        std::cerr << "Error loading CUDA function " << #name \
                  << " (symbol not found)\n";                \
    }

namespace dexsim {
namespace cudamgr {

CudaFunctionManager::CudaFunctionManager() {
#ifdef _WIN32
    HMODULE cuda_lib = (HMODULE)LoadLibraryA("nvcuda.dll");
#else
    void* cuda_lib = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                            RTLD_NOW | RTLD_GLOBAL);
#endif
    if (!cuda_lib) {
        std::cerr << "Unable to load library.\n";
        return;
    }

    // Driver and Initialization
    LOAD_CUDA_FUNCTION(cuDriverGetVersion, "");
    LOAD_CUDA_FUNCTION(cuInit, "");
    LOAD_CUDA_FUNCTION(cuGetProcAddress, "");

    // Device Management
    LOAD_CUDA_FUNCTION(cuDeviceGetCount, "");
    LOAD_CUDA_FUNCTION(cuDeviceGet, "");
    LOAD_CUDA_FUNCTION(cuDeviceGetName, "");
    LOAD_CUDA_FUNCTION(cuDeviceGetAttribute, "");

    // Context Management
    LOAD_CUDA_FUNCTION(cuCtxCreate, "");
    LOAD_CUDA_FUNCTION(cuCtxGetCurrent, "");
    LOAD_CUDA_FUNCTION(cuCtxSynchronize, "");
    LOAD_CUDA_FUNCTION(cuCtxSetCurrent, "");
    LOAD_CUDA_FUNCTION(cuDevicePrimaryCtxRetain, "");

    // Memory Management
    LOAD_CUDA_FUNCTION(cuMemAlloc, "");
    LOAD_CUDA_FUNCTION(cuMemFree, "");
    LOAD_CUDA_FUNCTION(cuMemcpyHtoD, "");
    LOAD_CUDA_FUNCTION(cuMemcpyDtoH, "");

    // Module and Kernel Execution
    LOAD_CUDA_FUNCTION(cuModuleLoadData, "");
    LOAD_CUDA_FUNCTION(cuModuleLoadDataEx, "");
    LOAD_CUDA_FUNCTION(cuModuleGetFunction, "");
    LOAD_CUDA_FUNCTION(cuLaunchKernel, "");

    // Stream and Event Management
    LOAD_CUDA_FUNCTION(cuStreamCreate, "");
    LOAD_CUDA_FUNCTION(cuStreamDestroy, "");
    LOAD_CUDA_FUNCTION(cuStreamSynchronize, "");
    LOAD_CUDA_FUNCTION(cuEventCreate, "");
    LOAD_CUDA_FUNCTION(cuEventRecord, "");
    LOAD_CUDA_FUNCTION(cuStreamWaitEvent, "");

    // Event Management
    LOAD_CUDA_FUNCTION(cuEventDestroy, "");
    LOAD_CUDA_FUNCTION(cuEventSynchronize, "");

    // Pointer Attributes
    LOAD_CUDA_FUNCTION(cuPointerGetAttribute, "");

    // Error Handling
    LOAD_CUDA_FUNCTION(cuGetErrorString, "");

    std::cout << "CUDA library loaded successfully." << std::endl;
}

CudaFunctionManager::~CudaFunctionManager() {}

extern "C" void cudaCodesMgr(ICudaFunctionManager** mgr) {
    *mgr = &CudaFunctionManager::instance();
}
}  // namespace cudamgr
}  // namespace dexsim