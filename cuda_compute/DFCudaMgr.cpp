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

    std::ios::sync_with_stdio(false);
    const char* homePath = std::getenv("HOME");
    if (homePath) {
        std::filesystem::path targetPath =
                std::filesystem::path(homePath) / "dexsim_data" / "kernels";
        if (std::filesystem::exists(targetPath)) {
            for (const auto& entry :
                 std::filesystem::recursive_directory_iterator(targetPath)) {
                if (entry.path().filename() == "CoreLUT.txt") {
                    ProcessFile(entry.path());
                }
            }
        }
    }
}

void CudaManager::UnInit() {
    auto result = cuda_->cuMemFree(cu_device_);
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuda_->cuGetErrorString(result, &errorStr);
        std::cerr << "Failed to free device memory! Error: " << errorStr
                  << ", Result Code: " << result << std::endl;
    }
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

    result = cuda_->cuCtxCreate(&cu_context_, 0, cu_device_);
    // or result = cuda_->cuCtxCreate(&cu_context_, 0, cu_device_);
    if (result != CUDA_SUCCESS) {
        std::cout << "cuDevicePrimaryCtxRetain failed, code " << result << std::endl;
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

void CudaManager::LoadPTXFile(const std::string& type) {
    std::filesystem::path ptxPath = basePath_ / (type + ".ptx");

    if (!std::filesystem::exists(ptxPath)) {
        std::cerr << "Warning: PTX file not found for type " << type << ": "
                  << ptxPath << std::endl;
        return;
    }

    std::ifstream ptxFile(ptxPath, std::ios::binary);
    if (!ptxFile.is_open()) {
        std::cerr << "Warning: Failed to open PTX file: " << ptxPath
                  << std::endl;
        return;
    }

    std::string content((std::istreambuf_iterator<char>(ptxFile)),
                        std::istreambuf_iterator<char>());

    modules_[type] = nullptr;
    // auto result = cuda_->cuModuleLoadData(&modules_[type], content.c_str());
    auto result = cuda_->cuModuleLoadDataEx(
            &modules_[type], content.c_str(), 0, nullptr, nullptr);
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuda_->cuGetErrorString(result, &errorStr);
        std::cerr << "Failed to load PTX file: " << ptxPath
                  << ", Error: " << errorStr << ", Result Code: " << result
                  << std::endl;
        return;
    }
    std::cout << "Loaded PTX for type: " << type << " (" << content.size()
              << " bytes)" << std::endl;
}

void CudaManager::ProcessFile(const std::filesystem::path& filePath) {
    basePath_ = filePath.parent_path();
    std::ifstream file(filePath);
    if (!file.is_open()) return;

    std::string line;
    std::string currentType;
    int expectedCount = 0;
    int processedCount = 0;

    while (std::getline(file, line)) {
        line.erase(line.begin(),
                   std::find_if(line.begin(), line.end(),
                                [](int ch) { return !std::isspace(ch); }));
        line.erase(std::find_if(line.rbegin(), line.rend(),
                                [](int ch) { return !std::isspace(ch); })
                           .base(),
                   line.end());

        if (line.empty()) continue;

        size_t spacePos = line.find(' ');
        if (spacePos != std::string::npos) {
            std::string type = line.substr(0, spacePos);
            std::string countStr = line.substr(spacePos + 1);

            int count = std::stoi(countStr);
            currentType = type;
            expectedCount = count;
            processedCount = 0;

            LoadPTXFile(currentType);
        }

        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            if (currentType.empty()) continue;

            std::string originalName = line.substr(0, colonPos);
            std::string implementation = line.substr(colonPos + 1);

            CUfunction tempFunction;
            auto result = cuda_->cuModuleGetFunction(&tempFunction,
                                                     modules_[currentType],
                                                     implementation.c_str());
            functions_[originalName] = tempFunction;

            if (result != CUDA_SUCCESS) {
                const char* errorStr;
                cuda_->cuGetErrorString(result, &errorStr);
                std::cerr << "Failed to get function: " << originalName
                          << ", Error: " << errorStr
                          << ", Result Code: " << result << std::endl;
                continue;
            }
            std::cout << "Loaded function: " << originalName
                      << " from module: " << currentType << std::endl;
            processedCount++;
        }
    }

    file.close();
}

std::vector<CUstream>* CudaManager::GetStreamFamily(int stream_type) {
    switch (stream_type) {
        case RENDERING_STREAM:
            return &rendering_stream_;
        case CALCULATE_STREAM:
            return &calculate_stream_;
        case GEOMETRY_STREAM:
            return &geometry_stream_;
        case PHYSICS_STREAM:
            return &physics_stream_;
        case CUSTOM_STREAM:
            return &custom_stream_;
        default:
            std::cerr << "Invalid stream type: " << stream_type << std::endl;
            return nullptr;
    }
}

int CudaManager::CreateStreamInFamily(int stream_type) {
    // get target stream family
    std::vector<CUstream>* targetStreamFamily = GetStreamFamily(stream_type);

    // check if a null ptr is present, if so create a stream at that position
    for (size_t i = 0; i < targetStreamFamily->size(); ++i) {
        if ((*targetStreamFamily)[i] == nullptr) {
            CUDA_CODES result =
                    cuda_->cuStreamCreate(&(*targetStreamFamily)[i], 0);
            if (result != CUDA_SUCCESS) {
                std::cerr << "Failed to create stream at index " << i
                          << std::endl;
                return -1;
            }
            return i;  // return the index of the new stream
        }
    }

    // if no null ptr is found, create a new stream and add it to the family
    CUstream newStream;
    CUDA_CODES result = cuda_->cuStreamCreate(&newStream, 0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to create new stream." << std::endl;
        return -1;
    }

    targetStreamFamily->push_back(newStream);
    return targetStreamFamily->size() - 1;
}

CUstream CudaManager::GetStream(int stream_type, int stream_id) {
    if (stream_id < GetStreamFamily(stream_type)->size()) {
        return GetStreamFamily(stream_type)->at(stream_id);
    } else {
        std::cout << "Warning: stream type" << stream_type << " with id "
                  << stream_id
                  << " is not exist, use default stream instead.\n";
        return nullptr;
    }
}

void CudaManager::DeleteStreamFromFamily(int stream_type, int stream_id) {
    // get target stream family
    std::vector<CUstream>* targetStreamFamily = GetStreamFamily(stream_type);

    // check if the stream_id is valid
    if (stream_id < 0 || stream_id >= targetStreamFamily->size()) {
        std::cerr << "Invalid stream ID: " << stream_id << std::endl;
        return;
    }

    // get the stream to delete
    CUstream streamToDelete = (*targetStreamFamily)[stream_id];

    // if the stream is not null, destroy it
    if (streamToDelete != nullptr) {
        CUDA_CODES result = cuda_->cuStreamDestroy(streamToDelete);
        if (result != CUDA_SUCCESS) {
            std::cerr << "Failed to destroy stream with ID " << stream_id
                      << " in stream family " << stream_type << std::endl;
            return;
        }

        // set the stream pointer to null
        (*targetStreamFamily)[stream_id] = nullptr;
    } else {
        std::cerr << "Stream with ID " << stream_id << " in stream family "
                  << stream_type << " is already null, cannot destroy."
                  << std::endl;
    }
}

void CudaManager::AllocateDeviceMemoryImpl(CUdeviceptr* arr, size_t size) {
    // get primary context
    cuda_->cuDevicePrimaryCtxRetain(&cu_context_, cu_device_);
    auto result = cuda_->cuMemAlloc(arr, size);
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuda_->cuGetErrorString(result, &errorStr);
        std::cerr << "Failed to allocate device memory. Error: " << errorStr
                  << ", Result Code: " << result << std::endl;
        return;  // Exit if there is an error
    }
}

void CudaManager::ReleaseArrayDataDeviceImpl(SharedDataGPU* gpuData) {
    gpuData->semaphore_ -= 1;
    if (gpuData->semaphore_ == 0) {
        gpuData->is_allocated_ = false;
        auto result = cuda_->cuMemFree(gpuData->value_);
        if (result != CUDA_SUCCESS) {
            const char* errorStr;
            cuda_->cuGetErrorString(result, &errorStr);
            std::cerr << "Failed to free device memory. Error: " << errorStr
                      << ", Result Code: " << result << std::endl;
            return;
        }
        delete gpuData;
    }
}

void CudaManager::SyncToHostImpl(CUdeviceptr src, void* dst, size_t size) {
    auto result = cuda_->cuMemcpyDtoH(dst, src, size);
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuda_->cuGetErrorString(result, &errorStr);
        std::cerr << "Failed to copy data from device to host. Error: "
                  << errorStr << ", Result Code: " << result << std::endl;
        return;  // Exit if there is an error
    }
}

void CudaManager::SyncToDeviceImpl(void* src, CUdeviceptr dst, size_t size) {
    auto result = cuda_->cuMemcpyHtoD(dst, src, size);
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuda_->cuGetErrorString(result, &errorStr);
        std::cerr << "Failed to copy data from host to device. Error: "
                  << errorStr << ", Result Code: " << result << std::endl;
        return;  // Exit if there is an error
    }
}

void CudaManager::ReleaseWarpArgs(void** args) {
    if (!args) return;
    delete static_cast<CudaBounds*>(args[0]); 
    delete[] args;
}

void CudaManager::LaunchImpl(const char* func,
                             int num_arrays,
                             HyperArray<float>** arrays,
                             int stream_type,
                             int stream_id) {
    LaunchImplT<float>(func, num_arrays, arrays, stream_type, stream_id);
}

CUcontext* CudaManager::GetCudaContext() { return &cu_context_; }
CUdevice* CudaManager::GetCudaDevice() { return &cu_device_; }
ICudaFunctionManager* CudaManager::GetCuda() const { return cuda_; }

extern "C" void cudaInit(cudamgr::ICudaManager** mgr) {
    std::cout << "Initializing CudaManager..." << std::endl;
    *mgr = new CudaManager();
}
}  // namespace cudamgr
}  // namespace dexsim