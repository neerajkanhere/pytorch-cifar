#include <chrono>
#include <ratio>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


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

int main(int argc, char* argv[]) {
  int inpWidth = 32;
  int inpHeight = 32;

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  static constexpr const int width_ = 32;
  static constexpr const int height_ = 32;
  static constexpr const int output_size_ = 10;

  const char* input_names[] = {"input.1"};
  const char* output_names[] = {"255"};
  std::array<int64_t, 4> input_shape_{1, 3, width_, height_};
  std::array<float, 3 * width_ * height_> input_image_{};
  std::array<int64_t, 2> output_shape_{1, 10};
  std::array<float, output_size_> results_{};
  
  Ort::Env env;
  Ort::SessionOptions opts{}; 
  opts.SetInterOpNumThreads(1);
  opts.SetIntraOpNumThreads(1);
  opts.SetExecutionMode(ORT_SEQUENTIAL);
  Ort::Session session_{env, "./model.onnx", opts};
  
  
  Ort::RunOptions run_options;
  int N = 1000;
  std::cout << "======== ONNX ========\n";
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  auto output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                      output_shape_.data(), output_shape_.size());    
  for(int i=0; i < N; i++) {
    for(auto p = input_image_.begin(); p != input_image_.end(); ++p) { 
      *p = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    auto input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                      input_shape_.data(), input_shape_.size());
    session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    softmax(results_);
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  int msec = time_span.count() * 1000;
  std::cout << msec << " msec for " << N << " iterations\n";
  
  std::cout << "======== OpenCV ========\n";
  cv::setNumThreads(1);
  cv::dnn::Net net = cv::dnn::readNet("./model.onnx");
  cv::Scalar mean{0.4151, 0.3771, 0.4568};
  cv::Mat blob;
  t1 = high_resolution_clock::now();
  for(int i=0; i < N; i++) {
    cv::Mat img(width_, height_, CV_8UC3);
    //cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::dnn::blobFromImage(img, blob, 1.0, cv::Size(width_, height_), mean, false, false);
    net.setInput(blob);
    cv::Mat prob = net.forward();
  }  
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  msec = time_span.count() * 1000;
  std::cout << msec << " msec for " << N << " iterations\n";
} 
