// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#pragma once
#include <iostream>
#include <mutex>
#include "cuda/DFCudaMgr.h"
#include "cuda/DFCudaCodes.h"

#if defined(__unix__) || defined(__unix)
#include <dlfcn.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define dlsym GetProcAddress
#endif

#define LOAD_WARP_FUNCTION(name, version) do {                        \
    dlerror();                                                        \
    name = reinterpret_cast<decltype(name)>(                          \
      dlsym(warp_lib, #name version));                                \
    const char* err = dlerror();                                      \
    if (err) {                                                        \
      std::cerr << "Cannot load symbol `" #name version "`: "         \
                << err << '\n';                                       \
      std::terminate();                                               \
    }                                                                 \
  } while (0)

namespace dexsim {
namespace compute {

// WARP types
using CudaInitFunc = void (*)(cudamgr::ICudaManager**);

class DFComputeCore {
public:
    static void Initialize(bool useCuda) {
        std::cout<<"init compute core"<<std::endl;
        _instance.reset(new DFComputeCore(useCuda));
    }

    static DFComputeCore& Instance() {
        if (!_instance)
            throw std::runtime_error(
                    "DFComputeCore has not been initialize!, please use "
                    "Initialize() first");
        return *_instance;
    }

    /// \brief Get cu_mgr_
    ///
    /// \return the ICudaManager in CudaMgr
    cudamgr::ICudaManager* GetCudaMgr() { return cu_mgr_; }

    /// \brief Get the CUDA driver
    ///
    /// \return the CUdevice in CudaMgr
    cudamgr::ICudaFunctionManager* GetCudaDriver() {
        return cu_mgr_->GetCuda();
    }

private:
    /// \brief Initializes the CUDA manager and loads the warp library.
    /// \param useCuda Flag indicating whether to use CUDA.
    DFComputeCore(bool useCuda);

    /// \brief Initializes the CUDA manager and loads the warp library.
    /// \param useCuda Flag indicating whether to use CUDA.
    /// \warning While cuda is initialized, the warp library is also loaded.
    void CudaInit();

    DFComputeCore(const DFComputeCore&) = delete;
    DFComputeCore& operator=(const DFComputeCore&) = delete;
    DFComputeCore(DFComputeCore&&) = delete;
    DFComputeCore& operator=(DFComputeCore&&) = delete;

    inline static std::unique_ptr<DFComputeCore> _instance;
    inline static std::once_flag _initFlag;
    cudamgr::ICudaManager* cu_mgr_;
    CudaInitFunc cudaInit;
};
}  // namespace compute
}  // namespace dexsim