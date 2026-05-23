# day8 / LibTorch CPU-only inference (绕过 GPU wheel 的 CUDA toolkit 要求)
#
# 为什么不用 find_package(Torch):
#   PyTorch GPU wheel (cu128) 的 CMake config 强制要求 CUDA toolkit (nvcc, cuda.h),
#   而我们只装了 nvidia driver, 没装 CUDA toolkit. find_package 会失败.
#   绕开方法: 直接手动指 include / link CPU 版的 libtorch 共享库.
#   (GPU wheel 同时带 libtorch_cpu.so 和 libtorch_cuda.so, 我们只 link 前者)
#
# 用法 (从 day8/build/):
#   cmake ..
#   make -j
#   ./dqn_inference ../trained_dqn.script.pt
#
# 性能: CartPole forward 在 CPU 几微秒就够, 不需要 GPU.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dqn_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PyTorch 在 Python venv 里的位置
set(TORCH_ROOT "$ENV{HOME}/dl_env/lib/python3.12/site-packages/torch")

# 1. include headers
include_directories(SYSTEM
    ${TORCH_ROOT}/include
    ${TORCH_ROOT}/include/torch/csrc/api/include
)

# 2. link 共享库目录
link_directories(${TORCH_ROOT}/lib)

# 3. PyTorch wheel 用 cxx11 ABI (新版默认), 跟 gcc 13 一致
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# 4. 主程序
add_executable(dqn_inference main.cpp)

# link CPU 部分 (跳过 libtorch_cuda 即可避免 CUDA 依赖)
target_link_libraries(dqn_inference
    torch
    torch_cpu
    c10
)

# 设 rpath, 这样 binary 能在 runtime 找到 .so 文件 (不必每次 export LD_LIBRARY_PATH)
set_target_properties(dqn_inference PROPERTIES
    BUILD_RPATH "${TORCH_ROOT}/lib"
    INSTALL_RPATH "${TORCH_ROOT}/lib"
)
