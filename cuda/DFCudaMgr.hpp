// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#include <warp/native/builtin.h>

#include "DFCudaMgr.h"

namespace dexsim {
namespace cudamgr {

struct CudaBounds {
    int shape[4];
    int ndim;
    size_t size;
};

struct CudaThread {
    int numBlocks[3];
    int numThreadsPerBlock[3];
};

class CudaManager : public ICudaManager {
public:
    CudaManager();

    ICudaFunctionManager* GetCuda() const override;

private:
    void InitCUDA();

    ICudaFunctionManager* cuda_;
    CUdevice cu_device_ = 0;
    CUcontext cu_context_ = nullptr;

    int cudaDriverVersion_;
    int deviceCount_ = 0;
};
}  // namespace cudamgr
}  // namespace dexsim