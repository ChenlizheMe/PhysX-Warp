cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)

# 添加 Python pip 安装 warp_lang 步骤
find_package(Python REQUIRED COMPONENTS Interpreter Development)

add_custom_target(install_warp_lang
    COMMAND ${Python_EXECUTABLE} -m pip install warp_lang==1.7.0+cu11 
            --index-url http://rjyfb:123456@69.235.177.182:10801/simple/ 
            --trusted-host 69.235.177.182
    COMMENT "Installing warp_lang package"
    VERBATIM
)

# 自动检测 warp 的头文件路径（假设已安装）
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import warp, os; print(os.path.dirname(warp.__file__))"
    OUTPUT_VARIABLE WARP_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Found warp at: ${WARP_PYTHON_PATH}")

# 暴露路径供外部使用
set(WARP_PYTHON_INCLUDE "${WARP_PYTHON_PATH}" PARENT_SCOPE)

# 递归收集源文件和头文件
file(GLOB_RECURSE CUDA_COMPUTE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

set(CUDA_COMPUTE_SOURCES ${CUDA_COMPUTE_SOURCES} PARENT_SCOPE)
set(CUDA_COMPUTE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
