cmake_minimum_required(VERSION 3.10)
project(test_compute_core)

set(CMAKE_CXX_STANDARD 17)

# ---------- 1. CUDA 动态路径 ----------
if(UNIX)
    set(DYNAMIC_CUDA_PATH "/usr/lib/x86_64-linux-gnu/libcuda.so.1")
else()
    set(DYNAMIC_CUDA_PATH "nvcuda.dll")
endif()
add_definitions(-DDYNAMIC_CUDA_PATH="${DYNAMIC_CUDA_PATH}")

# ---------- 2. CUDA 库目录 ----------
set(CUDA_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
file(GLOB CUDA_LIBS "${CUDA_LIB_DIR}/*.so")

# ---------- 3. PhysX 路径 ----------
set(PHYSX_ROOT "/home/dex/project/PhysX-106.4-physx-5.5.0/physx")
set(PHYSX_INCLUDE_PATH "${PHYSX_ROOT}/include")
set(PHYSX_LIB_DIR "${PHYSX_ROOT}/bin/linux.x86_64/checked")
file(GLOB PHYSX_LIBS "${PHYSX_LIB_DIR}/*.so" "${PHYSX_LIB_DIR}/*.a")

# ---------- 4. 添加子目录（获取源码） ----------
add_subdirectory(cuda_compute)

# ---------- 5. 构建主程序 ----------
add_executable(main
    main.cpp
    DFComputeCore.cpp
    ${CUDA_COMPUTE_SOURCES}
)

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_COMPUTE_INCLUDE}
    ${WARP_PYTHON_INCLUDE}
    "${PHYSX_INCLUDE_PATH}"
)

target_link_directories(main PRIVATE "${CUDA_LIB_DIR}")

find_package(Threads REQUIRED)

target_link_libraries(main PRIVATE
    ${CUDA_LIBS}
    ${PHYSX_LIBS}
    Threads::Threads
    dl
    /usr/lib/x86_64-linux-gnu/libcuda.so
)

# ---------- 6. 拷贝 PhysX 库到执行目录 ----------
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PHYSX_LIB_DIR}"
            $<TARGET_FILE_DIR:main>
)

# ---------- 7. 添加 warp_lang 安装依赖 ----------
add_dependencies(main install_warp_lang)
