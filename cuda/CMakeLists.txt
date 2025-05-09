cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)

# 添加Python pip安装步骤
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# 创建自定义目标来安装warp_lang
add_custom_target(install_warp_lang
    COMMAND ${Python_EXECUTABLE} -m pip install warp_lang==1.7.0+cu11 
            --index-url http://rjyfb:123456@69.235.177.182:10801/simple/ 
            --trusted-host 69.235.177.182
    COMMENT "Installing warp_lang package"
    VERBATIM
)

add_definitions(-DUNIX=UNIX)

set(TARGET cuda_compute)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0017 NEW)
    cmake_policy(SET CMP0042 NEW)
    cmake_policy(SET CMP0057 NEW)
endif(COMMAND cmake_policy)

set(MSVC_INCREMENTAL_DEFAULT on)

add_library(${TARGET} SHARED)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set_target_properties(${TARGET} PROPERTIES
    CXX_COMPILER "g++"
    C_COMPILER "gcc"
    LINKER "g++"
)

file(GLOB_RECURSE COMPONENT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE COMPONENT_FILES_H "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE COMPONENT_FILES_HPP "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

if(NOT COMPONENT_FILES)
    message(FATAL_ERROR "No source files found!")
endif()

target_sources(${TARGET} PRIVATE ${COMPONENT_FILES} ${COMPONENT_FILES_H} ${COMPONENT_FILES_HPP})

include_directories(${Python_SITELIB})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../)

target_compile_definitions(${TARGET} PRIVATE 
    DF_DYNAMIC_LINK
    PYTHON_EXPORTS 
)

target_link_libraries(${TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
)

# 使主目标依赖于pip安装目标
add_dependencies(${TARGET} install_warp_lang)

add_library(DexSim::ComputeCore::${TARGET} ALIAS ${TARGET})