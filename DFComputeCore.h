// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#pragma once
#include <iostream>
#include "DFComputeType.h"
#include "cuda_compute/DFCudaMgr.h"
#include <mutex>
#include "cuda_compute/DFCudaCodes.h"

#include <dlfcn.h>

namespace dexsim {
namespace compute {

// WARP types
using CudaInitFunc = void (*)(cudamgr::ICudaManager**);

class DFComputeCore {
public:
    static void Initialize(bool useCuda) {
        std::call_once(_initFlag,
                       [&]() { _instance.reset(new DFComputeCore(useCuda)); });
    }

    static void Uninitialize() {
        if (_instance) {
            _instance->cu_mgr_->UnInit();
            _instance.reset();
        }
    }

    static DFComputeCore& Instance() {
        if (!_instance)
            throw std::runtime_error(
                    "DFComputeCore has not been initialize!, please use "
                    "Initialize() first");
        return *_instance;
    }

    /// \brief Creates a stream in the specified stream family.
    ///
    /// \param stream_type The type of the stream family
    /// \return The ID of the created stream
    int CreateStream(int stream_type);

    /// \brief Deletes a stream from the specified stream family.
    ///
    /// \param stream_type The type of the stream family
    /// \param stream_id The ID of the stream to delete
    /// \warning This function does not check if the stream is currently in use.
    void DeleteStream(int stream_type, int stream_id);

    /// \brief Create a HyperArray with the specified dimensions and shape.
    ///
    /// \param arr Pointer to CUdeviceptr that will store the allocated memory
    /// \param dims Number of dimensions
    /// \param shape Array of dimensions
    /// \param data Pointer to the data to be allocated
    /// \param use_gpu Flag indicating whether to use GPU. If dose, the data
    /// will be copy to device, but not copy to host. \warning After a
    /// HyperArray is created, the data will be allocated on the device, and the
    /// shape will be freeze.
    template <typename T>
    void CreateArray(HyperArrayHook* arr,
                     int ndim,
                     int* shape,
                     T* data,
                     bool use_gpu = false) {
        cu_mgr_->CreateArray<T>(arr, ndim, shape, data, use_gpu);
    }

    /// \brief Allocates device memory for a HyperArray
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \tparam T Type of data stored in the array
    template <typename T>
    void AllocateDevice(HyperArrayHook arr) {
        cu_mgr_->AllocateDevice<T>(arr);
    }

    /// \brief Allocates host memory for a HyperArray
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \tparam T Type of data stored in the array
    template <typename T>
    void AllocateHost(HyperArrayHook arr) {
        cu_mgr_->AllocateHost<T>(arr);
    }

    /// \brief Synchronizes data from the host to the device
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \tparam T Type of data stored in the array
    template <typename T>
    void SyncToDevice(HyperArrayHook arr) {
        cu_mgr_->SyncToDevice<T>(arr);
    }

    /// \brief Synchronizes data from the device to the host
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \tparam T Type of data stored in the array
    template <typename T>
    void SyncToHost(HyperArrayHook arr) {
        cu_mgr_->SyncToHost<T>(arr);
    }

    /// \brief Synchronizes data from the device to the host
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \tparam T Type of data stored in the array
    template <typename T>
    void GetArrayDataDevice(HyperArrayHook arr, T* data) {
        cu_mgr_->GetArrayDataDevice<T>(arr, data);
    }

    /// \brief Writes data to a HyperArray on the host
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \param data Pointer to host data to write to the array
    /// \tparam T Type of data stored in the array
    template <typename T>
    void GetArrayDataHost(HyperArrayHook arr, T* data) {
        cu_mgr_->GetArrayDataHost<T>(arr, data);
    }

    /// \brief Writes data to a HyperArray on the host
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \param data Pointer to host data to write to the array
    /// \tparam T Type of data stored in the array
    template <typename T>
    void WriteArrayDataHost(HyperArrayHook arr, T* data) {
        cu_mgr_->WriteArrayDataHost<T>(arr, data);
    }

    /// \brief Writes data to a HyperArray on the device
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \param data Pointer to device data to write to the array
    /// \tparam T Type of data stored in the array
    template <typename T>
    void WriteArrayDataDevice(HyperArrayHook arr, T* data) {
        cu_mgr_->WriteArrayDataDevice(arr, data);
    }

    /// \brief Releases GPU data for a HyperArray
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    template <typename T>
    void ReleaseArrayDataDevice(HyperArrayHook arr) {
        cu_mgr_->ReleaseArrayDataDevice<T>(arr);
    }

    /// \brief Releases CPU data for a HyperArray
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    template <typename T>
    void ReleaseArrayDataHost(HyperArrayHook arr) {
        cu_mgr_->ReleaseArrayDataHost<T>(arr);
    }

    /// \brief Shares CPU data between two HyperArrays
    ///
    /// \param src Source HyperArray handle
    /// \param dst Destination HyperArray handle
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataHost(HyperArrayHook src, HyperArrayHook dst) {
        cu_mgr_->ShareFromArrayDataHost<T>(src, dst);
    }

    /// \brief Shares CPU data from a HyperArray to a pointer
    ///
    /// \param src Source HyperArray handle
    /// \param dst Destination pointer
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataHost(HyperArrayHook src, T* dst) {
        cu_mgr_->ShareFromArrayDataHost<T>(src, dst);
    }

    /// \brief Shares GPU data between two HyperArrays
    ///
    /// \param src Source HyperArray handle
    /// \param dst Destination HyperArray handle
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataDevice(HyperArrayHook src, HyperArrayHook dst) {
        cu_mgr_->ShareFromArrayDataDevice<T>(src, dst);
    }

    /// \brief Shares GPU data from a HyperArray to a pointer
    ///
    /// \param src Source HyperArray handle
    /// \param dst Destination pointer
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataDevice(HyperArrayHook src, T* dst) {
        cu_mgr_->ShareFromArrayDataDevice<T>(src, dst);
    }

    /// \brief Shares GPU data from a pointer to a HyperArray
    ///
    /// \param src Source pointer
    /// \param dst Destination HyperArray handle
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataDevice(T* src, HyperArrayHook dst) {
        cu_mgr_->ShareFromArrayDataDevice<T>(src, dst);
    }

    /// \brief Launches a CUDA kernel with the specified function name and
    ///
    /// arrays. \param func Name of the kernel function to launch \param
    /// num_arrays Number of arrays to pass to the kernel \param arrays Array of
    /// HyperArray handles \tparam T Type of data stored in the arrays
    /// \param stream_type The type of the stream family
    /// \param stream_id The ID of the stream to use
    template <typename T>
    void Launch(const char* func,
                int num_arrays,
                HyperArrayHook* arrays,
                int stream_type = -1,
                int stream_id = -1) {
        cu_mgr_->Launch<T>(func, num_arrays, arrays, stream_type, stream_id);
    }

    /// \brief Synchronizes the specified stream
    ///
    /// \return the CUcontext in CudaMgr
    void* GetCudaContext();

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
};
}  // namespace compute
}  // namespace dexsim