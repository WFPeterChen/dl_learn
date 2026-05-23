# day9 / Full C++ deploy: LibTorch + MuJoCo
#
# 用法 (从 day9/build/):
#   cmake ..
#   make -j
#   ./dqn_pendulum ../../day8/trained_dqn.script.pt

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dqn_pendulum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. LibTorch + CUDA ---
list(APPEND CMAKE_PREFIX_PATH
    "/usr/local/cuda-12.8"
    "$ENV{HOME}/dl_env/lib/python3.12/site-packages/torch"
)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- 2. MuJoCo (Python venv 里的, 没有 CMake config, 手动设 IMPORTED target) ---
set(MUJOCO_ROOT "$ENV{HOME}/dl_env/lib/python3.12/site-packages/mujoco")
add_library(mujoco_lib SHARED IMPORTED)
set_target_properties(mujoco_lib PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_ROOT}/libmujoco.so.3.8.1"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
)

# --- 3. 主程序 ---
add_executable(dqn_pendulum main.cpp)
target_link_libraries(dqn_pendulum
    "${TORCH_LIBRARIES}"
    mujoco_lib
)

# Runtime rpath: 让 binary 跑时自动找 mujoco lib, 不依赖 LD_LIBRARY_PATH
set_target_properties(dqn_pendulum PROPERTIES
    BUILD_RPATH "${MUJOCO_ROOT}"
)
