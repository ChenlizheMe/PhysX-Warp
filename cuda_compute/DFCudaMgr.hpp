// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#include <native/builtin.h>

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

    int CreateStreamInFamily(int stream_type) override;
    void DeleteStreamFromFamily(int stream_type, int stream_id) override;
    void AllocateDeviceMemoryImpl(CUdeviceptr* arr, size_t size) override;
    void SyncToHostImpl(CUdeviceptr src, void* dst, size_t size) override;
    void SyncToDeviceImpl(void* src, CUdeviceptr dst, size_t size) override;

    CUdevice* GetCudaDevice()  override;
    CUcontext* GetCudaContext()  override;
    ICudaFunctionManager* GetCuda() const override;
    void UnInit() override;

private:
    void InitCUDA();

    void ProcessFile(const std::filesystem::path& filePath);
    void LoadPTXFile(const std::string& type);

    std::vector<CUstream>* GetStreamFamily(int stream_type);
    CUstream GetStream(int stream_type, int stream_id);

    void ReleaseArrayDataDeviceImpl(SharedDataGPU* gpuData) override;

    void LaunchImpl(const char* func,
                    int num_arrays,
                    HyperArray<float>** arrays,
                    int stream_type,
                    int stream_id) override;

    // Generates warp arguments for Warp kernel launch automatically
    template <typename T>
    void** GetWarpArgs(int num_arrays,
                       int ndim,
                       ArrayShape shape,
                       size_t strides[4],
                       const std::vector<CUdeviceptr>& gpu_data) {
        CudaBounds* bounds = new CudaBounds;
        std::vector<wp::array_t<T>>* arrays =
                new std::vector<wp::array_t<T>>(num_arrays);
        void** args = new void*[num_arrays + 1];  // bounds + arrays

        bounds->ndim = ndim;
        bounds->size = 1;
        for (int i = 0; i < ndim; ++i) {
            bounds->shape[i] = shape[i];
            bounds->size *= shape[i];
        }

        for (int arr_idx = 0; arr_idx < num_arrays; ++arr_idx) {
            for (int dim = 0; dim < ndim; ++dim) {
                (*arrays)[arr_idx].shape[dim] = shape[dim];
                (*arrays)[arr_idx].strides[dim] = strides[dim];
            }
            (*arrays)[arr_idx].data = reinterpret_cast<T*>(gpu_data[arr_idx]);
        }

        args[0] = bounds;
        for (int i = 0; i < num_arrays; ++i) { args[i + 1] = &(*arrays)[i]; }

        return args;
    }

    CudaThread GetCudaThread(int dim, ArrayShape shape) {
        CudaThread thread;
        if (dim == 1) {
            thread.numThreadsPerBlock[0] = shape[0] < 256 ? shape[0] : 256;
            thread.numThreadsPerBlock[1] = 1;
            thread.numThreadsPerBlock[2] = 1;
            thread.numBlocks[0] =
                    (shape[0] + thread.numThreadsPerBlock[0] - 1) /
                    thread.numThreadsPerBlock[0];
            thread.numBlocks[1] = 1;
            thread.numBlocks[2] = 1;
        } else if (dim == 2) {
            thread.numThreadsPerBlock[0] = shape[0] < 16 ? shape[0] : 16;
            thread.numThreadsPerBlock[1] = shape[1] < 16 ? shape[1] : 16;
            thread.numThreadsPerBlock[2] = 1;
            thread.numBlocks[0] =
                    (shape[0] + thread.numThreadsPerBlock[0] - 1) /
                    thread.numThreadsPerBlock[0];
            thread.numBlocks[1] =
                    (shape[1] + thread.numThreadsPerBlock[1] - 1) /
                    thread.numThreadsPerBlock[1];
            thread.numBlocks[2] = 1;
        } else {
            thread.numThreadsPerBlock[0] = shape[0] < 8 ? shape[0] : 8;
            thread.numThreadsPerBlock[1] = shape[1] < 8 ? shape[1] : 8;
            thread.numThreadsPerBlock[2] = shape[2] < 8 ? shape[2] : 8;
            thread.numBlocks[0] =
                    (shape[0] + thread.numThreadsPerBlock[0] - 1) /
                    thread.numThreadsPerBlock[0];
            thread.numBlocks[1] =
                    (shape[1] + thread.numThreadsPerBlock[1] - 1) /
                    thread.numThreadsPerBlock[1];
            thread.numBlocks[2] =
                    (shape[2] + thread.numThreadsPerBlock[2] - 1) /
                    thread.numThreadsPerBlock[2];
        }
        return thread;
    }

    template <typename T>
    void LaunchImplT(const char* kernel_name,
                     int num_arrays,
                     HyperArray<T>** arrays,
                     int stream_type,
                     int stream_id) {
        int threadNum = 1;
        ArrayShape threadShape;
        std::vector<CUdeviceptr> gpu_data(num_arrays);

        for (int i = 0; i < 4; ++i) { threadShape[i] = 1; }
        for (int i = 0; i < num_arrays; ++i) {
            gpu_data[i] = arrays[i]->gpu_data_->value_;
            threadNum =
                    arrays[i]->ndim_ > threadNum ? arrays[i]->ndim_ : threadNum;
            auto currentShape = arrays[i]->shape_;
            for (int j = 0; j < arrays[i]->ndim_; ++j) {
                threadShape[j] = currentShape[j] > threadShape[j]
                                         ? currentShape[j]
                                         : threadShape[j];
            }
        }

        auto warpArgs =
                GetWarpArgs<T>(num_arrays, arrays[0]->ndim_, arrays[0]->shape_,
                               arrays[0]->strides_, gpu_data);
        auto thread = GetCudaThread(threadNum, threadShape);
        CUstream stream =
                stream_type == -1 ? nullptr : GetStream(stream_type, stream_id);
        // 4. Launch the kernel
        auto res = cuda_->cuLaunchKernel(
                functions_[kernel_name], thread.numBlocks[0],
                thread.numBlocks[1], thread.numBlocks[2],  // grid dim
                thread.numThreadsPerBlock[0], thread.numThreadsPerBlock[1],
                thread.numThreadsPerBlock[2],  // block dim
                0,                             // shared mem
                stream,                        // stream
                warpArgs,                      // kernel args
                nullptr);

        // 5. Release resources
        ReleaseWarpArgs(warpArgs);

        // 6. Error checking
        if (res != CUDA_SUCCESS) {
            const char* errorStr;
            cuda_->cuGetErrorString(res, &errorStr);
            std::cerr << "Kernel launch failed (" << kernel_name
                      << "): " << errorStr << std::endl;
        }
    }

    // Releases warp arguments after kernel launch
    void ReleaseWarpArgs(void** args);

    ICudaFunctionManager* cuda_;
    CUdevice cu_device_ = 0;
    CUcontext cu_context_ = nullptr;

    // only use default stream now, stream family will be used in the next PR
    std::vector<CUstream> rendering_stream_;
    std::vector<CUstream> calculate_stream_;
    std::vector<CUstream> geometry_stream_;
    std::vector<CUstream> physics_stream_;
    std::vector<CUstream> custom_stream_;

    std::map<std::string, CUmodule> modules_;
    std::map<std::string, CUfunction> functions_;
    std::filesystem::path basePath_;

    int cudaDriverVersion_;
    int deviceCount_ = 0;
};
}  // namespace cudamgr
}  // namespace dexsim