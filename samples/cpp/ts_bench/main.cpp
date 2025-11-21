#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

namespace {
constexpr int kWarmupIterations = 3;
constexpr int kBenchmarkIterations = 10000;

void set_env(const std::string& name, const std::string& value) {
#ifdef _WIN32
    if (_putenv_s(name.c_str(), value.c_str()) != 0) {
        throw std::runtime_error("failed to set environment variable " + name);
    }
#else
    if (setenv(name.c_str(), value.c_str(), 1) != 0) {
        std::ostringstream oss;
        oss << "failed to set environment variable " << name << ": " << std::strerror(errno);
        throw std::runtime_error(oss.str());
    }
#endif
}

std::string build_binding_string(int total_threads) {
    int core_id = 1;
    std::ostringstream oss;
    for (int s = 0; s < total_threads; ++s) {
        if (s > 0) {
            oss << '_';
        }
        oss << core_id++;
    }
    return oss.str();
}

void zero_inputs(ov::InferRequest& request) {
    for (const auto& input : request.get_compiled_model().inputs()) {
        ov::Tensor tensor = request.get_tensor(input);
        std::memset(tensor.data(), 0, tensor.get_byte_size());
    }
}

void print_usage(const char* app) {
    std::cout << "Usage: " << app << " <path_to_model> <streams> <threads_per_stream> <total_threads>\n";
    std::cout << "  path_to_model      - required OpenVINO IR (.xml/.onnx)\n";
    std::cout << "  streams            - positive integer number of CPU streams\n";
    std::cout << "  threads_per_stream - positive integer threads allocated per stream\n";
    std::cout << "  total_threads      - positive integer total threads to bind\n";
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 5) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];
    const int streams = std::stoi(argv[2]);
    const int threads_per_stream = std::stoi(argv[3]);
    const int total_threads = std::stoi(argv[4]);

    if (streams <= 0 || threads_per_stream <= 0) {
        std::cerr << "streams and threads_per_stream must be positive integers\n";
        return EXIT_FAILURE;
    }

    try {
        const std::string binding = build_binding_string(total_threads);
        set_env("TS_OV_BIND_CORES", binding);
        std::cout << "[ts_bench] TS_OV_BIND_CORES=" << binding << std::endl;

        ov::Core core;
        ov::AnyMap properties;
        properties["INFERENCE_NUM_THREADS"] = std::to_string(streams);
        properties["NUM_STREAMS"] = std::to_string(threads_per_stream * streams);
        properties.emplace(ov::hint::enable_cpu_pinning(true));

        auto model = core.read_model(model_path);
        auto compiled_model = core.compile_model(model, "CPU", properties);

        std::vector<ov::InferRequest> infer_requests;
        infer_requests.reserve(streams);
        for (int i = 0; i < streams; ++i) {
            auto request = compiled_model.create_infer_request();
            zero_inputs(request);
            infer_requests.emplace_back(std::move(request));
        }

        for (int i = 0; i < kWarmupIterations; ++i) {
            for (auto& request : infer_requests) {
                request.start_async();
            }
            for (auto& request : infer_requests) {
                request.wait();
            }
        }

        double total_ms = 0.0;
        double max_ms = 0.0;

        for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
            const auto start = std::chrono::high_resolution_clock::now();
            for (auto& request : infer_requests) {
                request.start_async();
            }
            for (auto& request : infer_requests) {
                request.wait();
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - start).count();
            total_ms += ms;
            if (ms > max_ms) {
                max_ms = ms;
            }
        }

        const double avg_ms = total_ms / static_cast<double>(kBenchmarkIterations);
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[ts_bench] iterations=" << kBenchmarkIterations << ", warmup=" << kWarmupIterations << '\n';
        std::cout << "[ts_bench] avg latency (ms): " << avg_ms << '\n';
        std::cout << "[ts_bench] max latency (ms): " << max_ms << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "ts_bench failed: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

