// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#include "DFCudaMgr.hpp"

namespace dexsim {
namespace cudamgr {

CudaManager::CudaManager() {
    InitCUDA();
}

void CudaManager::InitCUDA() {
    cudaCodesMgr(&cuda_);

    auto result = cuda_->cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "Failed to initialize CUDA driver API." << std::endl;
        return;
    }

    result = cuda_->cuDriverGetVersion(&cudaDriverVersion_);
    std::cout << "CUDA driver version: " << cudaDriverVersion_ << std::endl;
    result = cuda_->cuDeviceGetCount(&deviceCount_);
    std::cout << "CUDA Device count: " << deviceCount_ << std::endl;

    result = cuda_->cuDeviceGet(&cu_device_, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "Failed to get CUDA device" << std::endl;
        return;
    }

    // get device name
    char deviceName[256];
    result = cuda_->cuDeviceGetName(deviceName, sizeof(deviceName), cu_device_);
    if (result != CUDA_SUCCESS) {
        std::cout << "Failed to get device name" << std::endl;
        return;
    }
    std::cout << "Using CUDA device: " << deviceName << std::endl;

    result = cuda_->cuCtxCreate(&cu_context_, CU_CTX_LMEM_RESIZE_TO_MAX | CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, cu_device_);
    if (result != CUDA_SUCCESS) {
        std::cout << "cuCtxCreate failed, code " << result << std::endl;
        return;
    }

    result = cuda_->cuCtxSetCurrent(cu_context_);
    if (result != CUDA_SUCCESS) {
        std::cout << "cuCtxSetCurrent failed, code " << result << std::endl;
        return;
    }

    std::cout << "Using CUcontext: " << cu_context_ << " with address: " << cu_context_ << std::endl;

    std::cout << "Cuda manager init successful." << std::endl;
}

ICudaFunctionManager* CudaManager::GetCuda() const { return cuda_; }

extern "C" void cudaInit(cudamgr::ICudaManager** mgr) {
    std::cout << "Initializing CudaManager..." << std::endl;
    *mgr = new CudaManager();
}
}  // namespace cudamgr
}  // namespace dexsim