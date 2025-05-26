// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------

#include "DFComputeCore.h"
#include "cuda_compute/DFCudaMgr.hpp"
#include <dlfcn.h>
namespace dexsim {
namespace compute {
DFComputeCore::DFComputeCore(bool useCuda) {
    if (useCuda) { CudaInit(); }
}

void DFComputeCore::CudaInit() {
    cu_mgr_ = new cudamgr::CudaManager();
}

int DFComputeCore::CreateStream(int stream_type) {
    int stream_id = cu_mgr_->CreateStreamInFamily(stream_type);
    if (stream_id == -1) {
        std::cerr << "Failed to create stream in family " << stream_type
                  << std::endl;
    }
    return stream_id;
}

void DFComputeCore::DeleteStream(int stream_type, int stream_id) {
    cu_mgr_->DeleteStreamFromFamily(stream_type, stream_id);
}

void* DFComputeCore::GetCudaContext() { return cu_mgr_->GetCudaContext(); }
}  // namespace compute
}  // namespace dexsim