// day8 / C++ LibTorch inference demo
//
// 加载 day8 导出的 TorchScript 模型 (trained_dqn.script.pt),
// 在 C++ 里跑 forward, 输出 Q 值 + argmax action.
// 用同样的 test states 跟 Python (export_dqn.py 输出) 1:1 比对.
//
// 编译: 见 CMakeLists.txt
// 运行: ./dqn_inference ../trained_dqn.script.pt

#include <torch/script.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <path-to-script.pt>\n";
        return 1;
    }

    // -------- 1. 加载 TorchScript 模型 --------
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    // -------- 2. CPU-only inference --------
    // 注: CartPole forward 微秒级, CPU 完全够; 跳过 CUDA 避免依赖 toolkit
    torch::Device device(torch::kCPU);
    module.to(device);
    module.eval();

    std::cout << "Loaded model from " << argv[1] << "\n"
              << "  device: cpu (CartPole forward 不需要 GPU)\n\n";

    // -------- 3. Test states (跟 Python 一致) --------
    std::vector<std::vector<float>> test_states = {
        {0.0f,  0.00f, 0.0f, 0.0f},
        {0.0f,  0.05f, 0.0f, 0.0f},
        {0.0f, -0.05f, 0.0f, 0.0f},
        {0.0f,  0.15f, 0.0f, 0.0f},
        {0.0f, -0.15f, 0.0f, 0.0f},
    };

    std::cout << "=== C++ Q values ===\n"
              << "  state -> [Q(a=-3), Q(a=0), Q(a=+3)]  argmax\n"
              << std::fixed << std::setprecision(4);

    const int action_forces[] = {-3, 0, 3};

    for (const auto& s : test_states) {
        // 构造 input tensor (1, 4) on device
        auto state = torch::tensor(s, torch::dtype(torch::kFloat32).device(device)).unsqueeze(0);

        // forward
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(state);
        auto q = module.forward(inputs).toTensor();   // (1, 3)

        int64_t a = q.argmax(1).item<int64_t>();

        // 打印
        std::cout << "  [";
        for (size_t i = 0; i < s.size(); ++i) {
            std::cout << std::showpos << s[i] << (i < s.size() - 1 ? ", " : "");
        }
        std::cout << std::noshowpos << "] -> ["
                  << std::showpos
                  << q[0][0].item<float>() << ", "
                  << q[0][1].item<float>() << ", "
                  << q[0][2].item<float>() << "]"
                  << std::noshowpos
                  << "  argmax=" << a
                  << " (force=" << std::showpos << action_forces[a] << std::noshowpos << ")\n";
    }

    return 0;
}
