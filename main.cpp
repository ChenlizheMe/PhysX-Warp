#include "DFComputeCore.h"
#include <PxPhysicsAPI.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

using namespace dexsim;
using namespace dexsim::compute;
using namespace physx;

class MyErrorCallback : public PxErrorCallback {
public:
    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) override {
        std::cerr << "PhysX Error [" << code << "] " << message << " at " << file << ":" << line << std::endl;
    }
};

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open failed: " + path);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!f.read(buffer.data(), size)) throw std::runtime_error("read failed: " + path);
    return buffer;
}

int main() {
    // 初始化你的 CUDA 接口（DFComputeCore 封装）
    compute::DFComputeCore::Initialize(true);
    auto cuMgr = compute::DFComputeCore::Instance().GetCudaMgr()->GetCuda();

    // 获取 Primary Context
    dexsim::cudamgr::CUdevice device;
    dexsim::cudamgr::CUcontext ctx;
    cuMgr->cuDeviceGet(&device, 0);
    auto res = cuMgr->cuCtxGetCurrent(&ctx);
    if (res != 0) {
        std::cerr << "cuDevicePrimaryCtxRetain failed: " << res << "\n";
        return -1;
    }

    cuMgr->cuCtxSetCurrent(ctx);
    std::cout << "CUDA context retained and set successfully.\n";

    // 可选：加载 CUBIN/PTX 模块
    try {
        auto ptx = readFile("hello.cubin");
        dexsim::cudamgr::CUmodule mod;
        res = cuMgr->cuModuleLoadData(&mod, ptx.data());
        if (res == 0)
            std::cout << "CUDA module loaded successfully\n";
        else
            std::cerr << "Module load failed: " << res << "\n";
    } catch (const std::exception& e) {
        std::cerr << "PTX load error: " << e.what() << "\n";
    }

    // 创建 PhysX 基础对象
    static PxDefaultAllocator allocator;
    static MyErrorCallback errorCallback;

    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errorCallback);
    if (!foundation) {
        std::cerr << "Failed to create PxFoundation\n";
        return -1;
    }

    // 创建共享 CUDA 上下文管理器
    PxCudaContextManagerDesc desc;
    desc.ctx = reinterpret_cast<CUcontext*>(&ctx);
    desc.appGUID = "DEX_PHYSX_APP_GUID";

    auto cudaMgr = PxCreateCudaContextManager(*foundation, desc, nullptr);
    if (!cudaMgr || !cudaMgr->contextIsValid()) {
        std::cerr << "Failed to initialize PxCudaContextManager\n";
        if (cudaMgr) cudaMgr->release();
        foundation->release();
        return -1;
    }

    std::cout << "PhysX CudaContextManager initialized with shared context.\n";

    auto &compute_core = dexsim::compute::DFComputeCore::Instance();
    dexsim::cudamgr::HyperArrayHook a, b, dest;

    // test 1d array
    int shape1d[1] = {6};
    std::vector<float> value_1d_a = {1, 2, 3, 4, 5, 6};
    std::vector<float> value_1d_b = {10, 9, 8, 7, 6, 5};
    std::vector<float> value_1d_dest(6, 0);
    compute_core.CreateArray<float>(&a, 1, shape1d, value_1d_a.data(), false);
    compute_core.CreateArray<float>(&b, 1, shape1d, value_1d_b.data(), true);
    compute_core.CreateArray<float>(&dest, 1, shape1d, value_1d_dest.data(),
                                    true);
    dexsim::cudamgr::HyperArrayHook args[] = {a, b, dest};

    compute_core.AllocateDevice<float>(a);
    compute_core.SyncToDevice<float>(a);

    compute_core.Launch<float>("array1d_addf32_0", 3, args);
    std::vector<float> value = {0, 0, 0, 0, 0, 0};
    compute_core.AllocateHost<float>(dest);
    compute_core.SyncToHost<float>(dest);
    compute_core.GetArrayDataHost<float>(dest, value.data());
    std::cout << "1D Array Result: ";
    for (const auto& v : value) {
        std::cout << v << " ";
    }
    std::cout << "\n";
    return 0;
}
