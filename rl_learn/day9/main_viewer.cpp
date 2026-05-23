// day9 / Full C++ deploy + GLFW viewer.
//
// 跟 main.cpp 区别:
//   main.cpp        headless 跑完, 打印 statistics
//   main_viewer.cpp 开 GLFW 窗口实时显示, 鼠标可旋转视角
//
// 基于 mujoco 官方 simulate.cc 精简而来.
//
// 用法 (从 day9/build/):
//   ./dqn_pendulum_viewer ../../day8/trained_dqn.script.pt

#include <torch/script.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <iomanip>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

// MJCF + actions
const char* MJCF =
    "/home/peter/dl_env/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/inverted_pendulum.xml";
constexpr float ACTION_FORCES[3] = {-3.0f, 0.0f, 3.0f};

// ===== Globals (GLFW callbacks 是 C 风格 function pointer, 需要 global 状态) =====
mjModel* m  = nullptr;
mjData*  d  = nullptr;
mjvCamera  cam;
mjvOption  opt;
mjvScene   scn;
mjrContext con;

// 鼠标交互
bool   button_left   = false;
bool   button_middle = false;
bool   button_right  = false;
double last_x        = 0;
double last_y        = 0;

// 控制 flags
bool paused = false;

// ===== GLFW callbacks =====
void keyboard(GLFWwindow* /*window*/, int key, int /*scancode*/, int act, int /*mods*/) {
    if (act != GLFW_PRESS) return;
    if (key == GLFW_KEY_BACKSPACE) {           // 复位
        mj_resetData(m, d);
        d->qpos[1] = 0.05;
        mj_forward(m, d);
        std::cout << "  [BACKSPACE] reset\n";
    } else if (key == GLFW_KEY_SPACE) {        // 暂停 / 继续
        paused = !paused;
        std::cout << "  [SPACE] " << (paused ? "paused" : "resume") << "\n";
    }
}

void mouse_button(GLFWwindow* window, int /*button*/, int /*act*/, int /*mods*/) {
    button_left   = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
    button_middle = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    button_right  = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
    glfwGetCursorPos(window, &last_x, &last_y);
}

void mouse_move(GLFWwindow* window, double x, double y) {
    if (!button_left && !button_middle && !button_right) return;

    double dx = x - last_x;
    double dy = y - last_y;
    last_x = x; last_y = y;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (button_right)       action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)   action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else                    action = mjMOUSE_ZOOM;

    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

void scroll(GLFWwindow* /*window*/, double /*xoff*/, double yoff) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoff, &scn, &cam);
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <path-to-script.pt>\n";
        return 1;
    }

    // ---- 1. 加载 mujoco ----
    char error[1024] = "";
    m = mj_loadXML(MJCF, nullptr, error, sizeof(error));
    if (!m) { std::cerr << "mj_loadXML failed: " << error << "\n"; return 1; }
    d = mj_makeData(m);

    // ---- 2. 加载 DQN ----
    torch::jit::script::Module net = torch::jit::load(argv[1]);
    net.to(torch::kCPU);
    net.eval();

    std::cout << "=== day9 viewer ===\n"
              << "  MJCF:   " << MJCF << "\n"
              << "  DQN pt: " << argv[1] << "\n"
              << "\n窗口操作:\n"
              << "  鼠标左键拖     旋转视角\n"
              << "  鼠标右键拖     平移视角\n"
              << "  滚轮          缩放\n"
              << "  Space         暂停 / 继续\n"
              << "  Backspace     复位\n"
              << "  Esc / 关窗    退出\n\n";

    // ---- 3. GLFW + OpenGL context setup ----
    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }

    GLFWwindow* window = glfwCreateWindow(1200, 900, "day9: DQN @ MuJoCo (C++)", nullptr, nullptr);
    if (!window) { std::cerr << "glfwCreateWindow failed\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);   // vsync

    // ---- 4. MuJoCo visualization setup ----
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                      // max 2000 geoms
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // GLFW callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // ---- 5. Episode init ----
    mj_resetData(m, d);
    d->qpos[1] = 0.05;
    mj_forward(m, d);

    // ---- 6. Main loop ----
    while (!glfwWindowShouldClose(window)) {
        // 每帧推进 sim 大约 1/60s (匹配 60Hz 显示, real-time)
        mjtNum sim_start = d->time;
        const mjtNum frame_dt = 1.0 / 60.0;

        while (!paused && d->time - sim_start < frame_dt) {
            // DQN forward
            std::array<float, 4> obs = {
                static_cast<float>(d->qpos[0]),
                static_cast<float>(d->qpos[1]),
                static_cast<float>(d->qvel[0]),
                static_cast<float>(d->qvel[1])
            };
            auto state_t = torch::from_blob(obs.data(), {1, 4}, torch::kFloat32);
            int64_t action = net.forward({state_t}).toTensor().argmax(1).item<int64_t>();
            d->ctrl[0] = ACTION_FORCES[action];

            mj_step(m, d);

            // auto reset on terminate
            if (std::abs(d->qpos[1]) > 0.2) {
                std::cout << "  terminate at t=" << d->time << "s, reset\n";
                mj_resetData(m, d);
                d->qpos[1] = 0.05;
                mj_forward(m, d);
                break;   // skip this frame's remaining sub-steps
            }
        }

        // Render
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ---- Cleanup ----
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    glfwDestroyWindow(window);
    glfwTerminate();
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
