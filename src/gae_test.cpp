#include "ppo_kernels.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_floats(std::ifstream& in, size_t count) {
    std::vector<float> data(count);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        throw std::runtime_error("failed to read float buffer");
    }
    return data;
}

std::vector<int> read_ints(std::ifstream& in, size_t count) {
    std::vector<int> data(count);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(int)));
    if (!in) {
        throw std::runtime_error("failed to read int buffer");
    }
    return data;
}

void write_floats(std::ofstream& out, const std::vector<float>& data) {
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("failed to write float buffer");
    }
}

int find_iters(int argc, char** argv, int default_iters) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--iters") {
            return std::max(1, std::stoi(argv[i + 1]));
        }
    }
    return default_iters;
}

bool is_flag(const char* arg) {
    return std::string(arg).rfind("--", 0) == 0;
}

void make_synthetic(
    int N_envs,
    int T,
    int seed,
    std::vector<float>& rewards,
    std::vector<float>& values,
    std::vector<float>& dones) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::bernoulli_distribution done_dist(0.05);
    rewards.resize(static_cast<size_t>(N_envs) * T);
    values.resize(static_cast<size_t>(N_envs) * (T + 1));
    dones.resize(static_cast<size_t>(N_envs) * T);
    for (float& v : rewards) {
        v = normal(rng);
    }
    for (float& v : values) {
        v = normal(rng);
    }
    for (float& v : dones) {
        v = done_dist(rng) ? 1.0f : 0.0f;
    }
}

int run_gae(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error("usage: gae_test N_envs T gamma lam [input.bin output.bin] [--iters K]");
    }
    int N_envs = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    float gamma = std::stof(argv[3]);
    float lam = std::stof(argv[4]);
    int iterations = find_iters(argc, argv, 1);

    std::vector<float> rewards;
    std::vector<float> values;
    std::vector<float> dones;
    std::string output_path;
    if (argc >= 7 && !is_flag(argv[5])) {
        std::ifstream in(argv[5], std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open GAE input file");
        }
        rewards = read_floats(in, static_cast<size_t>(N_envs) * T);
        values = read_floats(in, static_cast<size_t>(N_envs) * (T + 1));
        dones = read_floats(in, static_cast<size_t>(N_envs) * T);
        output_path = argv[6];
    } else {
        make_synthetic(N_envs, T, 0, rewards, values, dones);
    }

    GaeResult result = run_gae_host(rewards, values, dones, N_envs, T, gamma, lam, iterations);

    if (!output_path.empty()) {
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("failed to open GAE output file");
        }
        write_floats(out, result.advantages);
        write_floats(out, result.returns);
    }

    std::cout << std::fixed << std::setprecision(6)
              << "{\"mode\":\"gae\",\"N_envs\":" << N_envs
              << ",\"T\":" << T
              << ",\"gpu_ms\":" << result.milliseconds
              << ",\"peak_mem_gbps\":" << get_device_peak_memory_bandwidth_gbps()
              << "}\n";
    return 0;
}

int run_normalize(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error("usage: gae_test normalize M input.bin output.bin [eps] [--iters K]");
    }
    int M = std::stoi(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    float eps = (argc >= 6 && !is_flag(argv[5])) ? std::stof(argv[5]) : 1.0e-8f;
    int iterations = find_iters(argc, argv, 1);

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open normalize input file");
    }
    std::vector<float> x = read_floats(in, M);
    NormalizeResult result = run_normalize_host(x, eps, iterations);

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open normalize output file");
    }
    write_floats(out, result.advantages);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"mode\":\"normalize\",\"M\":" << M
              << ",\"gpu_ms\":" << result.milliseconds << "}\n";
    return 0;
}

int run_gather(int argc, char** argv) {
    if (argc < 7) {
        throw std::runtime_error("usage: gae_test gather M D batch_size input.bin output.bin [--iters K]");
    }
    int M = std::stoi(argv[2]);
    int D = std::stoi(argv[3]);
    int batch_size = std::stoi(argv[4]);
    std::string input_path = argv[5];
    std::string output_path = argv[6];
    int iterations = find_iters(argc, argv, 1);

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open gather input file");
    }
    std::vector<float> flat = read_floats(in, static_cast<size_t>(M) * D);
    std::vector<int> indices = read_ints(in, batch_size);
    GatherResult result = run_gather_host(flat, indices, M, D, iterations);

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open gather output file");
    }
    write_floats(out, result.out);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"mode\":\"gather\",\"M\":" << M
              << ",\"D\":" << D
              << ",\"batch_size\":" << batch_size
              << ",\"gpu_ms\":" << result.milliseconds << "}\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "normalize") {
            return run_normalize(argc, argv);
        }
        if (argc >= 2 && std::string(argv[1]) == "gather") {
            return run_gather(argc, argv);
        }
        return run_gae(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

