// day10 / 31-action DQN viewer (基于 day9/main_viewer.cpp)
#include <torch/script.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <array>
#include <cmath>

const char* MJCF =
    "/home/peter/dl_env/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/inverted_pendulum.xml";

constexpr int N_ACTIONS = 31;
float ACTION_FORCES[N_ACTIONS];   // runtime init: linspace(-3, +3, 31)

mjModel* m = nullptr;
mjData*  d = nullptr;
mjvCamera  cam;
mjvOption  opt;
mjvScene   scn;
mjrContext con;

bool button_left = false, button_middle = false, button_right = false;
double last_x = 0, last_y = 0;
bool paused = false;

void keyboard(GLFWwindow*, int key, int, int act, int) {
    if (act != GLFW_PRESS) return;
    if (key == GLFW_KEY_BACKSPACE) { mj_resetData(m, d); d->qpos[1] = 0.05; mj_forward(m, d); }
    else if (key == GLFW_KEY_SPACE) paused = !paused;
}

void mouse_button(GLFWwindow* w, int, int, int) {
    button_left   = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
    button_middle = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    button_right  = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
    glfwGetCursorPos(w, &last_x, &last_y);
}

void mouse_move(GLFWwindow* w, double x, double y) {
    if (!button_left && !button_middle && !button_right) return;
    double dx = x - last_x, dy = y - last_y;
    last_x = x; last_y = y;
    int width, height; glfwGetWindowSize(w, &width, &height);
    bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                 glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    mjtMouse a = button_right ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
              : button_left  ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
              : mjMOUSE_ZOOM;
    mjv_moveCamera(m, a, dx/height, dy/height, &scn, &cam);
}

void scroll(GLFWwindow*, double, double yoff) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoff, &scn, &cam);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <script.pt>\n"; return 1; }

    // 31 个离散 action: -3.0, -2.8, ..., 0.0, ..., +2.8, +3.0
    for (int i = 0; i < N_ACTIONS; ++i)
        ACTION_FORCES[i] = -3.0f + 0.2f * static_cast<float>(i);

    char error[1024] = "";
    m = mj_loadXML(MJCF, nullptr, error, sizeof(error));
    if (!m) { std::cerr << "mj_loadXML: " << error << "\n"; return 1; }
    d = mj_makeData(m);

    torch::jit::script::Module net = torch::jit::load(argv[1]);
    net.to(torch::kCPU);
    net.eval();

    std::cout << "=== day10: 31-action DQN viewer ===\n"
              << "  actions: " << N_ACTIONS << " forces ∈ [-3, +3] step 0.2\n"
              << "  左键拖=旋转  右键拖=平移  滚轮=缩放\n"
              << "  Space=暂停  Backspace=复位  Esc=退出\n\n";

    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(1200, 900, "day10: 31-action DQN", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam); mjv_defaultOption(&opt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    mj_resetData(m, d);
    d->qpos[1] = 0.05;
    mj_forward(m, d);

    while (!glfwWindowShouldClose(window)) {
        mjtNum sim_start = d->time;
        while (!paused && d->time - sim_start < 1.0/60.0) {
            std::array<float, 4> obs = {
                (float)d->qpos[0], (float)d->qpos[1],
                (float)d->qvel[0], (float)d->qvel[1]
            };
            auto t = torch::from_blob(obs.data(), {1, 4}, torch::kFloat32);
            int64_t a = net.forward({t}).toTensor().argmax(1).item<int64_t>();
            d->ctrl[0] = ACTION_FORCES[a];
            mj_step(m, d);
            if (std::abs(d->qpos[1]) > 0.2) {
                std::cout << "  terminate t=" << d->time << "s, reset\n";
                mj_resetData(m, d); d->qpos[1] = 0.05; mj_forward(m, d);
                break;
            }
        }
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    mjv_freeScene(&scn); mjr_freeContext(&con);
    glfwDestroyWindow(window); glfwTerminate();
    mj_deleteData(d); mj_deleteModel(m);
    return 0;
}
