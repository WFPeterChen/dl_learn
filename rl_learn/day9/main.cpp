// day9 / Full C++ deploy: LibTorch + MuJoCo C API
//
// 把 day7/02_mujoco_dqn.py 完整翻译成 C++.
// Python → C++ 1:1 对应:
//
//   mujoco.MjModel.from_xml_path(...)  →  mj_loadXML(...)
//   mujoco.MjData(model)               →  mj_makeData(m)
//   mujoco.mj_resetData(m, d)          →  mj_resetData(m, d)
//   mujoco.mj_forward(m, d)            →  mj_forward(m, d)
//   mujoco.mj_step(m, d)               →  mj_step(m, d)
//   torch.load(...)                    →  torch::jit::load(...)
//   net(state)                         →  module.forward({tensor})
//
// 用法 (从 day9/build/):
//   ./dqn_pendulum ../../day8/trained_dqn.script.pt

#include <torch/script.h>
#include <mujoco/mujoco.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>

// MJCF 路径 (复用 gymnasium 自带的 inverted_pendulum.xml)
const char* MJCF =
    "/home/peter/dl_env/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/inverted_pendulum.xml";

// 离散 action index → 连续 force (跟 day6 训练时一致)
constexpr float ACTION_FORCES[3] = {-3.0f, 0.0f, 3.0f};


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <path-to-script.pt>\n";
        return 1;
    }

    // -------- 1. 加载 MuJoCo model --------
    char error[1024] = "";
    mjModel* m = mj_loadXML(MJCF, nullptr, error, sizeof(error));
    if (!m) {
        std::cerr << "mj_loadXML failed: " << error << "\n";
        return 1;
    }
    mjData* d = mj_makeData(m);

    std::cout << "=== MuJoCo model loaded ===\n"
              << "  xml:   " << MJCF << "\n"
              << "  nq=" << m->nq << ", nv=" << m->nv
              << ", nu=" << m->nu << ", dt=" << m->opt.timestep << "s\n\n";

    // -------- 2. 加载 DQN (TorchScript) --------
    torch::jit::script::Module net;
    try {
        net = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "torch::jit::load failed: " << e.what() << "\n";
        mj_deleteData(d);
        mj_deleteModel(m);
        return 1;
    }

    // day8 结论: small-batch (B=1) + small NN 时 CPU 比 GPU 快 ~2x.
    // 控制 loop 单 step 推理用 CPU 最合适.
    torch::Device device(torch::kCPU);
    net.to(device);
    net.eval();

    std::cout << "=== DQN model loaded ===\n"
              << "  pt:     " << argv[1] << "\n"
              << "  device: cpu (single-step inference, GPU overhead too big)\n\n";

    // -------- 3. Episode 初始化 --------
    mj_resetData(m, d);
    d->qpos[1] = 0.05;   // 小初始扰动 (跟 day7/02 一致)
    mj_forward(m, d);

    // -------- 4. Control loop --------
    constexpr int MAX_STEPS = 1000;        // 20s @ 50Hz
    std::vector<double> hist_t, hist_x, hist_theta;
    std::vector<float> hist_u;

    std::cout << "=== Control loop (DQN policy in pure C++) ===\n";

    auto t_wall_start = std::chrono::high_resolution_clock::now();
    int step;
    for (step = 0; step < MAX_STEPS; ++step) {
        // a. 提 state 4 维: [x, theta, x_dot, theta_dot]
        std::array<float, 4> obs = {
            static_cast<float>(d->qpos[0]),
            static_cast<float>(d->qpos[1]),
            static_cast<float>(d->qvel[0]),
            static_cast<float>(d->qvel[1])
        };

        // b. 喂 NN
        // torch::from_blob 借用现有内存 (不 copy), shape (1, 4)
        auto state_t = torch::from_blob(obs.data(), {1, 4}, torch::kFloat32);
        auto q = net.forward({state_t}).toTensor();
        int64_t action = q.argmax(1).item<int64_t>();

        // c. ctrl
        d->ctrl[0] = ACTION_FORCES[action];

        // d. 物理推进 一步
        mj_step(m, d);

        // e. 记录
        hist_t.push_back(d->time);
        hist_x.push_back(d->qpos[0]);
        hist_theta.push_back(d->qpos[1]);
        hist_u.push_back(ACTION_FORCES[action]);

        // f. terminate check (跟 gym InvertedPendulum-v5 一致)
        if (std::abs(d->qpos[1]) > 0.2) {
            std::cout << "  terminate at step=" << step
                      << ", t=" << d->time << "s, |θ|="
                      << std::abs(d->qpos[1]) << "\n";
            break;
        }
    }
    auto t_wall_end = std::chrono::high_resolution_clock::now();
    auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_wall_end - t_wall_start).count();

    // -------- 5. Summary --------
    std::cout << "\n=== Result ===\n"
              << std::fixed << std::setprecision(3)
              << "  steps survived: " << step << " / " << MAX_STEPS << "\n"
              << "  sim time:       " << d->time << " s\n"
              << "  wall time:      " << wall_ms << " ms\n"
              << "  speedup:        " << (d->time * 1000.0 / wall_ms)
              << "x real-time\n";

    if (!hist_t.empty()) {
        std::cout << "  final x:        " << hist_x.back() << " m\n"
                  << "  final θ:        " << hist_theta.back() << " rad\n";
    }

    // -------- Cleanup --------
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
