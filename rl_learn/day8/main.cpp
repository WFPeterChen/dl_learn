// day8 / C++ LibTorch inference (GPU 版)
//
// 1. 加载 TorchScript model
// 2. GPU if available, 否则 CPU
// 3. 5 个 test states 跑 forward, 跟 Python 输出对比
// 4. Latency benchmark: 10000 次 forward, 算每次平均耗时

#include <torch/script.h>
#include <torch/cuda.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

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

    // -------- 2. GPU if available --------
    bool cuda_ok = torch::cuda::is_available();
    torch::Device device(cuda_ok ? torch::kCUDA : torch::kCPU);
    module.to(device);
    module.eval();

    std::cout << "Loaded model from " << argv[1] << "\n"
              << "  CUDA available: " << (cuda_ok ? "yes" : "no") << "\n"
              << "  device:         " << (device.is_cuda() ? "cuda" : "cpu") << "\n\n";

    // -------- 3. Test states (跟 Python 对比) --------
    std::vector<std::vector<float>> test_states = {
        {0.0f,  0.00f, 0.0f, 0.0f},
        {0.0f,  0.05f, 0.0f, 0.0f},
        {0.0f, -0.05f, 0.0f, 0.0f},
        {0.0f,  0.15f, 0.0f, 0.0f},
        {0.0f, -0.15f, 0.0f, 0.0f},
    };

    std::cout << "=== Q values (跟 Python 1:1 比对) ===\n"
              << "  state -> [Q(a=-3), Q(a=0), Q(a=+3)]  argmax\n"
              << std::fixed << std::setprecision(4);

    const int action_forces[] = {-3, 0, 3};

    for (const auto& s : test_states) {
        auto state = torch::tensor(s, torch::dtype(torch::kFloat32).device(device)).unsqueeze(0);
        auto q_gpu = module.forward({state}).toTensor();   // (1, 3) on GPU
        auto q     = q_gpu.to(torch::kCPU);                 // 拉回 CPU 打印 (强制 sync)

        int64_t a = q.argmax(1).item<int64_t>();

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

    // -------- 4. CPU vs GPU latency benchmark --------
    std::cout << "\n=== Latency benchmark (10000 forwards, batch=1) ===\n";

    auto bench_on_device = [&module](torch::Device dev) -> double {
        module.to(dev);
        auto s = torch::tensor({0.0f, 0.05f, 0.0f, 0.0f},
                               torch::dtype(torch::kFloat32).device(dev)).unsqueeze(0);
        std::vector<torch::jit::IValue> inp;
        inp.push_back(s);

        // Warmup (GPU 首次 lazy compile, CPU 也热个 cache)
        for (int i = 0; i < 100; ++i) module.forward(inp).toTensor();
        module.forward(inp).toTensor().to(torch::kCPU);   // 强制 sync

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; ++i) module.forward(inp).toTensor();
        module.forward(inp).toTensor().to(torch::kCPU);   // sync
        auto t1 = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 10000.0;
    };

    double cpu_us = bench_on_device(torch::kCPU);
    std::cout << "  CPU: " << std::fixed << std::setprecision(2)
              << cpu_us << " us/forward  ("
              << static_cast<int>(1e6 / cpu_us) << " inferences/sec)\n";

    if (cuda_ok) {
        double gpu_us = bench_on_device(torch::kCUDA);
        std::cout << "  GPU: " << gpu_us << " us/forward  ("
                  << static_cast<int>(1e6 / gpu_us) << " inferences/sec)\n";
        std::cout << "  GPU/CPU ratio: " << std::setprecision(2)
                  << (gpu_us / cpu_us) << "x\n";
        if (gpu_us > cpu_us) {
            std::cout << "  → CPU 反而更快! 这种 small-batch 推理 GPU launch overhead 占主导.\n"
                      << "    GPU 在大 batch / 大网络才能 amortize (摊销) overhead.\n";
        }
    }

    return 0;
}
