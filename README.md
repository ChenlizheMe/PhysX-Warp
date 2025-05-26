first, compile `hello.cu`:
```bash
nvcc -cubin -arch=sm_you hello.cu -o hello.cubin
```
then, modify `PHYSX_ROOT` in CMakeLists.txt:
```bash
# ---------- 3. PhysX 路径 ----------
set(PHYSX_ROOT "/home/dex/project/PhysX-106.4-physx-5.5.0/physx")
set(PHYSX_INCLUDE_PATH "${PHYSX_ROOT}/include")
set(PHYSX_LIB_DIR "${PHYSX_ROOT}/bin/linux.x86_64/checked")
file(GLOB PHYSX_LIBS "${PHYSX_LIB_DIR}/*.so" "${PHYSX_LIB_DIR}/*.a")
```

Finally:
```bash
mkdir build
cd build
cmake ..
make
```

You can use `CuCtxCreate` or `CuDevicePrimaryCtxRetain` to create a cuContext, in DFCudaMgr.cpp:71:
```C++
    result = cuda_->cuDevicePrimaryCtxRetain(&cu_context_, cu_device_);
    // or result = cuda_->cuCtxCreate(&cu_context_, 0, cu_device_);
```

While using `CuCtxCreate`, physX can not load module successfully:
```bash
Loaded function: array3d_addf32_0 from module: linear
CUDA context retained and set successfully.
CUDA module loaded successfully
PhysX Error [32] Failed to load CUDA module data. Cuda error code 200.
 at /builds/omniverse/physics/physx/source/cudamanager/src/Cu
```

and while using `CuDevicePrimaryCtxRetain`, CUDA self can not load module successfully:
```bash
Loaded function: array2d_addf32_0 from module: linear
Loaded function: array3d_addf32_0 from module: linear
CUDA context retained and set successfully.
CUDA module loaded successfully
PhysX CudaContextManager initialized with shared context.
Failed to allocate device memory. Error: invalid device context, Result Code: 201
Failed to copy data from host to device. Error: invalid device context, Result Code: 201
```