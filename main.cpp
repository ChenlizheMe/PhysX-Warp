#include "DFComputeCore.h"
#include <PxPhysicsAPI.h>

using namespace dexsim;
using namespace dexsim::compute;
using namespace physx;

int main()
{
    compute::DFComputeCore::Initialize(true);

    auto cuDriver = dexsim::compute::DFComputeCore::Instance().GetCudaMgr()->GetCuda();
    CUcontext cuContext;
    CUdevice device;
    cuDriver->cuDeviceGet(&device, 0);
    cuDriver->cuCtxGetCurrent(&cuContext);

    int major = 0;
    int minor = 0;
    cuDriver->cuDeviceGetAttribute(&major, 75, device);
    cuDriver->cuDeviceGetAttribute(&minor, 76, device);
    std::cout << "CUDA Device: " << major << "." << minor << std::endl;
    std::cout << "CUDA Context: " << cuContext << " with address " << &cuContext << std::endl;

    static PxDefaultErrorCallback gDefaultErrorCallback;
    static PxDefaultAllocator gDefaultAllocatorCallback;

    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, gDefaultAllocatorCallback, gDefaultErrorCallback);
    if (!foundation) {
        printf("Failed to create PxFoundation\n");
        return -1;
    }

    PxCudaContextManagerDesc cudaContextManagerDesc;
    cudaContextManagerDesc.ctx = &cuContext;
    
    cuDriver->cuCtxSetCurrent(*cudaContextManagerDesc.ctx);
    
    std::cout << "CUDA Context get in cuda context manager: " << *cudaContextManagerDesc.ctx << " address: " << cudaContextManagerDesc.ctx << std::endl;

    auto cudaMgr = PxCreateCudaContextManager(*foundation, cudaContextManagerDesc, PxGetProfilerCallback());
    if (!cudaMgr || !cudaMgr->contextIsValid()) {
        if (cudaMgr) cudaMgr->release();
        foundation->error(physx::PxErrorCode::eINVALID_OPERATION, __FILE__,
        __LINE__,
        "Failed to initialize PhysX CUDA Context Manager");
    }
    return 0;
}