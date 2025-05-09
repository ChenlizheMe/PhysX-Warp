cmake_minimum_required(VERSION 3.10)
project(test_compute_core)

set(CMAKE_CXX_STANDARD 17)

# --------- 1. 路径/变量阶段（随便放，顺序无所谓） -----------
# 动态 CUDA
if(UNIX)
    set(DYNAMIC_CUDA_PATH "/usr/lib/x86_64-linux-gnu/libcuda.so.1")
else()
    set(DYNAMIC_CUDA_PATH "nvcuda.dll")
endif()
add_definitions(-DDYNAMIC_CUDA_PATH="${DYNAMIC_CUDA_PATH}")

# CUDA libs
set(CUDA_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
file(GLOB CUDA_LIBS "${CUDA_LIB_DIR}/*.so")

# PhysX
set(PHYSX_ROOT "/home/dex/project/dexsim/3rdparty_downloads/vcpkg_installed/x64-linux-20250116/x64-linux")
find_path(PHYSX_INCLUDE_PATH "PxPhysicsAPI.h" PATHS "${PHYSX_ROOT}/include")
set(PHYSX_LIB_DIR "${PHYSX_ROOT}/bin/linux.clang/release")
file(GLOB PHYSX_LIBS "${PHYSX_LIB_DIR}/*.so" "${PHYSX_LIB_DIR}/*.a")

set(PHYSX_IMPORTED_TARGETS "")
foreach(lib_path IN LISTS PHYSX_LIBS)
    get_filename_component(lib_name "${lib_path}" NAME_WE)
    string(REPLACE "." "_" lib_name "${lib_name}")
    add_library(${lib_name} SHARED IMPORTED GLOBAL)
    set_target_properties(${lib_name} PROPERTIES IMPORTED_LOCATION "${lib_path}")
    list(APPEND PHYSX_IMPORTED_TARGETS ${lib_name})
endforeach()

add_executable(main
    main.cpp
    DFComputeCore.cpp
)

# 头文件
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${PHYSX_INCLUDE_PATH}"
)

# 链接目录（让 -L 自动带上）
target_link_directories(main PRIVATE "${CUDA_LIB_DIR}")

# 链接库
find_package(Threads REQUIRED)
target_link_libraries(main PRIVATE
    ${CUDA_LIBS}            # 你自己的 cuda/*.so
    ${PHYSX_IMPORTED_TARGETS}
    Threads::Threads
    dl
    ${DYNAMIC_CUDA_PATH}
)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PHYSX_LIB_DIR}"
            $<TARGET_FILE_DIR:main>
)
