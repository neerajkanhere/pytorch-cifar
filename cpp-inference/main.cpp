#define OPENVINO

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <core/session/onnxruntime_cxx_api.h>
#ifdef OPENVINO
#include <core/providers/openvino/openvino_provider_factory.h>
#endif

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ratio>
#include <string>
#include <vector>

using namespace std::chrono;

template <typename T>
static void softmax(T& input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

cv::Mat Bgr24MatToBlob(cv::Mat in, cv::Size output_size) {
  cv::Mat resized_bgr;
  cv::resize(in, resized_bgr, output_size, cv::InterpolationFlags::INTER_CUBIC);
  cv::Mat resized_rgb;
  cv::cvtColor(resized_bgr, resized_rgb, cv::ColorConversionCodes::COLOR_BGR2RGB);
  cv::Mat normalized;
  resized_rgb.convertTo(normalized, CV_32F, 1.0 / 255);
  cv::Mat channels[3];
  cv::split(normalized, channels);
  channels[0] = (channels[0] - 0.4914) / 0.2023;
  channels[1] = (channels[1] - 0.4822) / 0.1994;
  channels[2] = (channels[2] - 0.4465) / 0.2010;
  cv::merge(channels, 3, normalized);
  cv::Mat blob;
  cv::dnn::blobFromImage(normalized, blob);
  return blob;
}

int main(int argc, char* argv[]) {
  Ort::SessionOptions opts{};

  opts.SetInterOpNumThreads(1);
  opts.SetIntraOpNumThreads(1);
  opts.SetExecutionMode(ORT_SEQUENTIAL);
  
  #ifdef OPENVINO
  std::unordered_map<std::string, std::string> openvino_options;
    openvino_options["device_type"] = "CPU";
    openvino_options["precision"] = "FP32";
    openvino_options["num_of_threads"] = "1";
    openvino_options["num_of_streams"] = "1";
    openvino_options["cache_dir"] = "./";
    opts.AppendExecutionProvider("OpenVINO", openvino_options);
    opts.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    std::cout << "[INFO] Using OpenVINO execution provider" << std::endl;
  #endif

  cv::setNumThreads(1);

  Ort::Env env;
  Ort::Session session{env, "./model.onnx", opts};

  cv::dnn::Net net = cv::dnn::readNet("./model.onnx");

  static constexpr const int width = 32;
  static constexpr const int height = 32;
  static constexpr const int output_size = 10;

  std::array<int64_t, 4> input_shape{1, 3, width, height};
  cv::Mat input_blob;
  cv::Mat img_input(width, height, CV_8UC3, cv::Scalar{0, 0, 0});
  cv::dnn::blobFromImage(img_input, input_blob);  // Test with all-zero tensor to compare with python output

  std::array<int64_t, 2> output_shape_{1, 10};
  std::array<float, output_size> output_onnx{};
  std::array<float, output_size> output_opencv{};

  Ort::RunOptions run_options;
  int N = 1000;
  std::cout << "======== ONNX ========\n";
  const char* input_names[] = {"input.1"};
  const char* output_names[] = {"255"};  // mobilenetv1{"255"} mobilenetv2{"518"};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_onnx.data(), output_onnx.size(),
                                                       output_shape_.data(), output_shape_.size());

  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_blob.ptr<float>(), input_blob.total(),
                                                      input_shape.data(), input_shape.size());
  session.Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_blob.ptr<float>(), input_blob.total(),
                                                        input_shape.data(), input_shape.size());
    session.Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
    // softmax(output_onnx);
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  int msec = time_span.count() * 1000;
  std::cout << msec << " msec for " << N << " iterations\n";

  std::cout << "======== OpenCV ========\n";
  t1 = high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    net.setInput(input_blob);
    cv::Mat prob = net.forward();
    for (auto p = 0; p < prob.total(); p++) {
      output_opencv[p] = prob.at<float>(p);
    }
    // softmax(output_opencv);
  }
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  msec = time_span.count() * 1000;
  std::cout << msec << " msec for " << N << " iterations\n";

  std::cout << "======== Output: onnx, opencv ========\n";
  for (auto i = 0; i < output_onnx.size(); ++i) {
    std::cout << output_onnx[i] << ", " << output_opencv[i] << "\n";
  }
  std::cout << "\n";
}
