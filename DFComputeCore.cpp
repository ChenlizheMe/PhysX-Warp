// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------

#include "DFComputeCore.h"

namespace dexsim {
namespace compute {
DFComputeCore::DFComputeCore(bool useCuda) {
    std::cout<<"compute core constructor"<<std::endl;
    if (useCuda) { CudaInit(); }
}

void DFComputeCore::CudaInit() {
    std::cout<<"get cuda init"<<std::endl;
#ifdef _WIN32
    HMODULE warp_lib = (HMODULE)LoadLibraryA("cuda_compute.dll");
#else
    void* warp_lib = dlopen("libcuda_compute.so", RTLD_NOW | RTLD_GLOBAL);
#endif

    LOAD_WARP_FUNCTION(cudaInit, "");
    cudaInit(&cu_mgr_);
}
}  // namespace compute
}  // namespace dexsim