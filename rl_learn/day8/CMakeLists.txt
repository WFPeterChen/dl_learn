# day8 / LibTorch C++ inference (GPU 版, 用 standard find_package(Torch))
#
# 用法 (从 day8/build/):
#   cmake ..
#   make -j
#   ./dqn_inference ../trained_dqn.script.pt

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dqn_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefix paths — CMake 用来找:
#   1. /usr/local/cuda-12.8     CUDA toolkit (nvcc, headers, libcudart, ...)
#   2. ~/dl_env/.../torch       LibTorch (PyTorch wheel 自带的)
list(APPEND CMAKE_PREFIX_PATH
    "/usr/local/cuda-12.8"
    "$ENV{HOME}/dl_env/lib/python3.12/site-packages/torch"
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(dqn_inference main.cpp)
target_link_libraries(dqn_inference "${TORCH_LIBRARIES}")
