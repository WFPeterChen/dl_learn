# day10 / 31-action DQN viewer (LibTorch + MuJoCo + GLFW)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dqn31 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH
    "/usr/local/cuda-12.8"
    "$ENV{HOME}/dl_env/lib/python3.12/site-packages/torch"
)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(MUJOCO_ROOT "$ENV{HOME}/dl_env/lib/python3.12/site-packages/mujoco")
add_library(mujoco_lib SHARED IMPORTED)
set_target_properties(mujoco_lib PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_ROOT}/libmujoco.so.3.8.1"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
)

find_package(glfw3 REQUIRED)

add_executable(dqn31_viewer main_viewer.cpp)
target_link_libraries(dqn31_viewer "${TORCH_LIBRARIES}" mujoco_lib glfw)
set_target_properties(dqn31_viewer PROPERTIES BUILD_RPATH "${MUJOCO_ROOT}")
