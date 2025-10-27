/*
 * This script has been kept independent of the F3 package
 * to make it easier to test the exported .pt2 files on Orin.
 * This way, there is no need to install the F3 package on Orin,
 * which has many dependencies and can be cumbersome to install.
 *
 * Author: Richeek Das
 */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

struct Config {
  std::string f3_pt2_path;
  std::string dav2_pt2_path;
  std::string flowhead_pt2_path;
  int32_t warmup_runs = 20;
  int32_t runs = 200;
  int32_t n_events = 200000;
  int32_t dav2_height = 238;
  int32_t dav2_width = 308;
  int32_t flow_height = 238;
  int32_t flow_width = 308;
};

void print_usage(const char *program_name) {
  std::cout
      << "Usage: " << program_name << " [OPTIONS]\n"
      << "\nRequired options:\n"
      << "  --f3_pt2_path PATH          Path to the F3 pt2 file\n"
      << "\nOptional options:\n"
      << "  --dav2_pt2_path PATH        Path to the DAV2 pt2 file\n"
      << "  --flowhead_pt2_path PATH    Path to the FlowHead pt2 file\n"
      << "  --runs N                    Number of timed inference runs "
         "(default: 200)\n"
      << "  --warmup_runs N             Number of warmup runs before timing "
         "(default: 20)\n"
      << "  --dav2_height H             Target height for DAV2 input (default: "
         "238)\n"
      << "  --dav2_width W              Target width for DAV2 input (default: "
         "308)\n"
      << "  --flow_height H             Target height for FlowHead input "
         "(default: 238)\n"
      << "  --flow_width W              Target width for FlowHead input "
         "(default: 308)\n"
      << "  --help                      Display this help message\n";
}

bool parse_args(int argc, char *argv[], Config &config) {
  bool f3_path_provided = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--f3_pt2_path" && i + 1 < argc) {
      config.f3_pt2_path = argv[++i];
      f3_path_provided = true;
    } else if (arg == "--dav2_pt2_path" && i + 1 < argc) {
      config.dav2_pt2_path = argv[++i];
    } else if (arg == "--flowhead_pt2_path" && i + 1 < argc) {
      config.flowhead_pt2_path = argv[++i];
    } else if (arg == "--runs" && i + 1 < argc) {
      config.runs = std::stoi(argv[++i]);
    } else if (arg == "--warmup_runs" && i + 1 < argc) {
      config.warmup_runs = std::stoi(argv[++i]);
    } else if (arg == "--dav2_height" && i + 1 < argc) {
      config.dav2_height = std::stoi(argv[++i]);
    } else if (arg == "--dav2_width" && i + 1 < argc) {
      config.dav2_width = std::stoi(argv[++i]);
    } else if (arg == "--flow_height" && i + 1 < argc) {
      config.flow_height = std::stoi(argv[++i]);
    } else if (arg == "--flow_width" && i + 1 < argc) {
      config.flow_width = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
      print_usage(argv[0]);
      return false;
    }
  }

  if (!f3_path_provided) {
    std::cerr << "Error: --f3_pt2_path is required\n" << std::endl;
    print_usage(argv[0]);
    return false;
  }

  return true;
}

void print_config(const Config &config) {
  std::cout << "⚙️ Configuration:" << std::endl;
  std::cout << "    F3 PT2 path: " << config.f3_pt2_path << std::endl;
  std::cout << "    DAV2 PT2 path: "
            << (config.dav2_pt2_path.empty() ? "None" : config.dav2_pt2_path)
            << std::endl;
  std::cout << "    FlowHead PT2 path: "
            << (config.flowhead_pt2_path.empty() ? "None"
                                                 : config.flowhead_pt2_path)
            << std::endl;
  std::cout << "    DAV2 input shape: " << config.dav2_height << "x"
            << config.dav2_width << std::endl;
  std::cout << "    FlowHead input shape: " << config.flow_height << "x"
            << config.flow_width << std::endl;
  std::cout << "    Warmup runs: " << config.warmup_runs << std::endl;
  std::cout << "    Timed runs: " << config.runs << std::endl;
  std::cout << std::endl;
}

torch::Tensor generate_random_events(int32_t n_events) {
  // Generate random events (N, 4) with values in [0, 1) range
  // Columns: [x, y, t, p]
  return torch::rand({n_events, 4},
                     at::TensorOptions().dtype(at::kFloat).device(at::kCUDA)) *
         0.9f;
}

int main(int argc, char *argv[]) {
  c10::InferenceMode mode;

  Config config;

  if (!parse_args(argc, argv, config)) {
    return 1;
  }

  print_config(config);

  std::cout << "Loading F3 model..." << std::endl;
  auto start_load = std::chrono::high_resolution_clock::now();
  torch::inductor::AOTIModelPackageLoader f3_loader(config.f3_pt2_path);
  auto end_load = std::chrono::high_resolution_clock::now();

  auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_load - start_load);
  std::cout << "F3 model loading time: " << load_duration.count() << " ms"
            << std::endl;

  // Load depth and flow models if paths are provided
  std::optional<torch::inductor::AOTIModelPackageLoader> dav2_loader;
  std::optional<torch::inductor::AOTIModelPackageLoader> flowhead_loader;

  if (!config.dav2_pt2_path.empty()) {
    std::cout << "Loading DAV2 model..." << std::endl;
    start_load = std::chrono::high_resolution_clock::now();
    dav2_loader.emplace(config.dav2_pt2_path);
    end_load = std::chrono::high_resolution_clock::now();
    load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_load - start_load);
    std::cout << "DAV2 model loading time: " << load_duration.count() << " ms"
              << std::endl;
  }

  if (!config.flowhead_pt2_path.empty()) {
    std::cout << "Loading FlowHead model..." << std::endl;
    start_load = std::chrono::high_resolution_clock::now();
    flowhead_loader.emplace(config.flowhead_pt2_path);
    end_load = std::chrono::high_resolution_clock::now();
    load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_load - start_load);
    std::cout << "FlowHead model loading time: " << load_duration.count()
              << " ms" << std::endl;
  }

  // For now, using random events
  std::vector<torch::Tensor> inputs = {generate_random_events(config.n_events)};

  std::cout << "\nInput events shape: [" << inputs[0].size(0) << ", "
            << inputs[0].size(1) << "]" << std::endl;

  // Warmup runs
  std::cout << "\nRunning warmup iterations..." << std::endl;
  for (int i = 0; i < config.warmup_runs; i++) {
    std::vector<torch::Tensor> f3_outputs = f3_loader.run(inputs);
    torch::Tensor f3_feat = f3_outputs[0].permute({0, 3, 2, 1});

    if (i == 0)
      std::cout << "F3 output shape: [" << f3_feat.size(0) << ", "
                << f3_feat.size(1) << ", " << f3_feat.size(2) << ", "
                << f3_feat.size(3) << "]" << std::endl;

    if (dav2_loader) {
      torch::Tensor f3_feat_ds = torch::nn::functional::interpolate(
          f3_feat,
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{config.dav2_height, config.dav2_width})
              .mode(torch::kBilinear)
              .align_corners(true));

      std::vector<torch::Tensor> dav2_inputs = {f3_feat_ds.to(at::kFloat)};

      std::vector<torch::Tensor> depth_outputs =
          dav2_loader.value().run(dav2_inputs);

      if (i == 0)
        std::cout << "Depth output shape: [" << depth_outputs[0].size(0) << ", "
                  << depth_outputs[0].size(1) << ", "
                  << depth_outputs[0].size(2) << "]" << std::endl;
    }

    if (flowhead_loader) {
      torch::Tensor f3_feat_flow = torch::nn::functional::interpolate(
          f3_feat,
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{config.flow_height, config.flow_width})
              .mode(torch::kBilinear)
              .align_corners(true));
      std::vector<torch::Tensor> flow_inputs = {f3_feat_flow.to(at::kFloat)};
      std::vector<torch::Tensor> flow_outputs =
          flowhead_loader.value().run(flow_inputs);

      if (i == 0)
        std::cout << "Flow output shape: [" << flow_outputs[0].size(0) << ", "
                  << flow_outputs[0].size(1) << ", " << flow_outputs[0].size(2)
                  << ", " << flow_outputs[0].size(3) << "]" << std::endl;
    }
  }

  torch::cuda::synchronize();

  // Timed runs
  std::cout << "\nRunning timed iterations..." << std::endl;
  auto start_inference = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < config.runs; i++) {
    std::vector<torch::Tensor> f3_outputs = f3_loader.run(inputs);
    torch::Tensor f3_feat = f3_outputs[0].permute({0, 3, 2, 1});

    if (dav2_loader) {
      torch::Tensor f3_feat_ds = torch::nn::functional::interpolate(
          f3_feat,
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{config.dav2_height, config.dav2_width})
              .mode(torch::kBilinear)
              .align_corners(true));
      std::vector<torch::Tensor> dav2_inputs = {f3_feat_ds.to(at::kFloat)};
      std::vector<torch::Tensor> depth_outputs =
          dav2_loader.value().run(dav2_inputs);
    }

    if (flowhead_loader) {
      torch::Tensor f3_feat_flow = torch::nn::functional::interpolate(
          f3_feat,
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{config.flow_height, config.flow_width})
              .mode(torch::kBilinear)
              .align_corners(true));
      std::vector<torch::Tensor> flow_inputs = {f3_feat_flow.to(at::kFloat)};
      std::vector<torch::Tensor> flow_outputs =
          flowhead_loader.value().run(flow_inputs);
    }
  }
  torch::cuda::synchronize();
  auto end_inference = std::chrono::high_resolution_clock::now();

  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_inference - start_inference);
  double avg_time_ms = (total_duration.count() / 1000.0) / config.runs;

  std::cout << "\nAverage inference time over " << config.runs
            << " runs: " << std::fixed << std::setprecision(2) << avg_time_ms
            << " ms" << std::endl;

  return 0;
}