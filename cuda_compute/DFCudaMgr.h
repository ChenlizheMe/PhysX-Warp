// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <algorithm>

#include "DFCudaCodes.h"
#include "DFHyperArray.h"

#define RENDERING_STREAM 0
#define CALCULATE_STREAM 1
#define GEOMETRY_STREAM 2
#define PHYSICS_STREAM 3
#define CUSTOM_STREAM 4

namespace dexsim {
namespace cudamgr {
class ICudaManager {
public:
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
    void CreateArray(
            HyperArrayHook* arr, int dims, int* shape, T* data, bool use_gpu) {
        switch (dims) {
            case 1:
                *arr = new HyperArray<T>(shape[0]);
                break;
            case 2:
                *arr = new HyperArray<T>(shape[0], shape[1]);
                break;
            case 3:
                *arr = new HyperArray<T>(shape[0], shape[1], shape[2]);
                break;
            case 4:
                *arr = new HyperArray<T>(shape[0], shape[1], shape[2],
                                         shape[3]);
                break;
            default:
                // Error handling for unsupported dimensions
                std::cerr << "Error: Unsupported number of dimensions: " << dims
                          << std::endl;
                break;
        }
        if (use_gpu) {
            AllocateDevice<T>(*arr);
            WriteArrayDataDevice(*arr, data);
        } else {
            AllocateHost<T>(*arr);
            WriteArrayDataHost<T>(*arr, data);
        }
    }

    /// \brief Allocates device memory for a HyperArray
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    /// \@warning This function should be called only after the HyperArray has
    /// been created
    template <typename T>
    void AllocateDevice(HyperArrayHook arr) {
        auto* array = reinterpret_cast<HyperArray<T>*>(arr);
        if (array->gpu_data_ != nullptr && array->gpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to allocate device memory for the "
                         "gpu data has been allocated.\n";
            return;
        }

        array->gpu_data_ = new SharedDataGPU;
        AllocateDeviceMemoryImpl(&(array->gpu_data_->value_),
                                 array->strides_[0] * array->shape_[0]);
        array->gpu_data_->is_allocated_ = true;
    }

    /// \brief Allocates host memory for a HyperArray
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    /// \warning This function should be called only after the HyperArray has
    /// been created
    template <typename T>
    void AllocateHost(HyperArrayHook arr) {
        auto* array = static_cast<HyperArray<T>*>(arr);
        if (array->cpu_data_ != nullptr && array->cpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to allocate host memory for the cpu "
                         "data has been allocated.\n";
            return;
        }
        array->cpu_data_ = new SharedDataCPU<T>;
        array->cpu_data_->value_ = new T[array->size_];
        array->cpu_data_->is_allocated_ = true;
    }

    /// \brief Synchronizes data from the host to the device
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    template <typename T>
    void SyncToDevice(HyperArrayHook arr) {
        auto* array = static_cast<HyperArray<T>*>(arr);

        if (array->gpu_data_ == nullptr ||
            array->gpu_data_->is_allocated_ == false) {
            std::cerr << "Warning: Failed to sync device memory, GPU memory "
                         "has not been allocated.\n";
            return;
        }
        if (array->cpu_data_ == nullptr || !array->cpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to sync device memory, CPU memory "
                         "has not been allocated.\n";
            return;
        }
        // Transfer data from host to device
        SyncToDeviceImpl(array->cpu_data_->value_, array->gpu_data_->value_,
                         array->strides_[0] * array->shape_[0]);
    }

    /// \brief Synchronizes data from the device to the host
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    /// \warning This function should be called only after the HyperArray has
    /// been created
    template <typename T>
    void SyncToHost(HyperArrayHook arr) {
        auto* array = reinterpret_cast<HyperArray<T>*>(arr);
        if (array->gpu_data_->is_allocated_ == false) {
            std::cerr << "Warning: Failed to sync host memory, GPU memory "
                         "has not been allocated.\n";
            return;
        }
        if (array->cpu_data_ == nullptr || !array->cpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to sync host memory, CPU memory "
                         "has not been allocated.\n";
            return;
        }
        // Transfer data from device to host
        SyncToHostImpl(array->gpu_data_->value_, array->cpu_data_->value_,
                       array->strides_[0] * array->shape_[0]);
    }

    /// \brief Retrieves host data from a HyperArray
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    /// \param data Output buffer that will receive the host data
    template <typename T>
    void GetArrayDataHost(HyperArrayHook arr, T* data) {
        auto* array = reinterpret_cast<HyperArray<T>*>(arr);
        // Copy data from HyperArray's host storage to output buffer
        std::copy(array->cpu_data_->value_,
                  array->cpu_data_->value_ + array->size_, data);
    }

    /// \brief Retrieves device data from a HyperArray
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    /// \param data Output buffer that will receive the device data
    template <typename T>
    void GetArrayDataDevice(HyperArrayHook arr, T* data) {
        auto* array = reinterpret_cast<HyperArray<T>*>(arr);
        // Transfer data from device directly to output buffer
        SyncToHostImpl(array->gpu_data_->value_, data,
                       array->strides_[0] * array->shape_[0]);
    }

    /// \brief Writes data to a HyperArray on the host
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \param data Pointer to host data to write to the array
    /// \tparam T Type of data stored in the array
    template <typename T>
    void WriteArrayDataHost(HyperArrayHook arr, T* data) {
        auto* array = reinterpret_cast<HyperArray<T>*>(arr);
        if (array->cpu_data_ == nullptr || !array->cpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to write host memory, CPU memory "
                         "has not been allocated.\n";
            return;
        }
        std::copy(data, data + array->size_, array->cpu_data_->value_);
    }

    /// \brief Writes data to a HyperArray on the device
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \param data Pointer to device data to write to the array
    /// \tparam T Type of data stored in the array
    template <typename T>
    void WriteArrayDataDevice(HyperArrayHook arr, T* data) {
        auto* array = reinterpret_cast<HyperArray<T>*>(arr);
        if (array->gpu_data_ == nullptr || !array->gpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to write device memory, GPU memory "
                         "has not been allocated.\n";
            return;
        }
        SyncToDeviceImpl(data, array->gpu_data_->value_,
                         array->strides_[0] * array->shape_[0]);
    }

    /// \brief Releases GPU data for a HyperArray
    ///
    /// \tparam T Type of data stored in the array
    /// \param arr HyperArray handle created with CreateArray
    template <typename T>
    void ReleaseArrayDataDevice(HyperArrayHook arr) {
        auto* array = reinterpret_cast<HyperArray<T>*>(arr);
        if (array->gpu_data_ == nullptr || !array->gpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to release device memory, GPU memory "
                         "has not been allocated.\n";
            return;
        }
        ReleaseArrayDataDeviceImpl(array->gpu_data_);
        array->gpu_data_ = nullptr;
    }

    /// \brief Releases CPU data for a HyperArray
    ///
    /// \param arr HyperArray handle created with CreateArray
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ReleaseArrayDataHost(HyperArrayHook arr) {
        auto* array = reinterpret_cast<HyperArray<T>*>(arr);
        if (array->cpu_data_ == nullptr || !array->cpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to release host memory, CPU memory "
                         "has not been allocated.\n";
            return;
        }
        array->cpu_data_->semaphore_ -= 1;
        if (array->cpu_data_->semaphore_ == 0) { delete array->cpu_data_; }
        array->cpu_data_ = nullptr;
    }

    /// \brief Shares CPU data between two HyperArrays
    ///
    /// \param src Source HyperArray handle
    /// \param dst Destination HyperArray handle
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataHost(HyperArrayHook src, HyperArrayHook dst) {
        auto* dstArray = reinterpret_cast<HyperArray<T>*>(dst);
        auto* srcArray = reinterpret_cast<HyperArray<T>*>(src);
        if (dstArray->cpu_data_ == nullptr ||
            !dstArray->cpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to share from host memory, dst CPU "
                         "host "
                         "memory has not been allocated.\n";
            return;
        }
        if (srcArray->cpu_data_ == nullptr ||
            !srcArray->cpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to share from host memory, src CPU "
                         "host "
                         "memory has not been allocated.\n";
            return;
        }
        dstArray->cpu_data_ = srcArray->cpu_data_;
        dstArray->cpu_data_->semaphore_ += 1;
    }

    /// \brief Shares CPU data from a HyperArray to a pointer
    ///
    /// \param src Source HyperArray handle
    /// \param dst Destination pointer
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataHost(HyperArrayHook src, T* dst) {
        auto* srcArray = reinterpret_cast<HyperArray<T>*>(src);
        if (srcArray->cpu_data_ == nullptr ||
            !srcArray->cpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to share from host memory, src CPU "
                         "host "
                         "memory has not been allocated.\n";
            return;
        }
        dst = srcArray->cpu_data_->value_;
    }

    /// \brief Shares GPU data between two HyperArrays
    ///
    /// \param src Source HyperArray handle
    /// \param dst Destination HyperArray handle
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataDevice(HyperArrayHook src, HyperArrayHook dst) {
        auto* dstArray = reinterpret_cast<HyperArray<T>*>(dst);
        auto* srcArray = reinterpret_cast<HyperArray<T>*>(src);
        if (dstArray->gpu_data_ == nullptr ||
            !dstArray->gpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to share from device memory, dst "
                         "device "
                         "memory has not been allocated.\n";
            return;
        }
        if (srcArray->gpu_data_ == nullptr ||
            !srcArray->gpu_data_->is_allocated_) {
            std::cerr << "Warning: Failed to share from device memory, src "
                         "device "
                         "memory has not been allocated.\n";
            return;
        }
        ReleaseArrayDataDevice<T>(dstArray);
        dstArray->gpu_data_ = srcArray->gpu_data_;
        dstArray->gpu_data_->semaphore_ += 1;
    }

    /// \brief Shares GPU data from a pointer to a HyperArray
    ///
    /// \param src Source pointer
    /// \param dst Destination HyperArray handle
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataDevice(T* src, HyperArrayHook dst) {
        auto* dstArray = reinterpret_cast<HyperArray<T>*>(dst);
        if (src == nullptr) {
            std::cerr << "Warning: Failed to share from device memory, src "
                         "device "
                         "memory has not been allocated.\n";
            return;
        }
        ReleaseArrayDataDevice<T>(dstArray);
        dstArray->gpu_data_ = new SharedDataGPU;
        dstArray->gpu_data_->value_ = (CUdeviceptr)src;
        dstArray->gpu_data_->is_allocated_ = true;
        dstArray->gpu_data_->semaphore_ = 1;
    }

    /// \brief Shares GPU data from a HyperArray to a pointer
    ///
    /// \param src Source HyperArray handle
    /// \param dst Destination pointer
    /// \tparam T Type of data stored in the array
    template <typename T>
    void ShareFromArrayDataDevice(HyperArrayHook src, T* dst) {
        auto* srcArray = reinterpret_cast<HyperArray<T>*>(src);
        if (srcArray->gpu_data_ == nullptr ||
            srcArray->gpu_data_->is_allocated_ == false) {
            std::cerr << "Warning: Failed to share from device memory, src "
                         "device "
                         "memory has not been allocated.\n";
            return;
        }
        dst = (T*)srcArray->gpu_data_->value_;
    }

    /// \brief Launches a custom Warp kernel
    ///
    /// \param func Name of the kernel function to launch
    /// \param num_arrays Number of arrays to pass to the kernel
    /// \param arrays Array of HyperArrayHook pointers representing the arrays
    /// \param stream_type Type of the stream to use for the kernel launch
    /// \param stream_id ID of the stream to use for the kernel launch
    /// \tparam T Type of data stored in the arrays
    template <typename T>
    void Launch(const char* func,
                int num_arrays,
                HyperArrayHook* arrays,
                int stream_type,
                int stream_id) {
        std::vector<HyperArray<T>*> converted_arrays(num_arrays);
        for (int i = 0; i < num_arrays; ++i) {
            converted_arrays[i] = reinterpret_cast<HyperArray<T>*>(arrays[i]);
        }

        LaunchImpl(func, num_arrays, converted_arrays.data(), stream_type,
                   stream_id);
    }

    /// \brief Creates a stream in a specific stream family
    ///
    /// \param stream_type The type/category of the stream family
    /// \return The ID of the created stream
    virtual int CreateStreamInFamily(int stream_type) = 0;

    /// \brief Deletes a stream from a specific stream family
    ///
    /// \param stream_type The type/category of the stream family
    /// \param stream_id The ID of the stream to delete
    virtual void DeleteStreamFromFamily(int stream_type, int stream_id) = 0;

    virtual CUdevice* GetCudaDevice()  = 0;
    virtual CUcontext* GetCudaContext()  = 0;
    virtual ICudaFunctionManager* GetCuda() const = 0;
    virtual void UnInit() = 0;

protected:
    virtual void ReleaseArrayDataDeviceImpl(SharedDataGPU* gpuData) = 0;
    virtual void AllocateDeviceMemoryImpl(CUdeviceptr* arr, size_t size) = 0;
    virtual void SyncToHostImpl(CUdeviceptr src, void* dst, size_t size) = 0;
    virtual void SyncToDeviceImpl(void* src, CUdeviceptr dst, size_t size) = 0;

    // virtual function do not support template, so we need to use
    // non-template function to call the template function.
    // pipeline: Launch->LaunchImpl->LaunchImplT
    virtual void LaunchImpl(const char* func,
                            int num_arrays,
                            HyperArray<float>** arrays,
                            int stream_type,
                            int stream_id) = 0;
};

}  // namespace cudamgr
}  // namespace dexsim