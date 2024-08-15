#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::vector<std::string> readLabels(std::string& labelFilepath) {
  std::vector<std::string> labels;
  std::string line;
  std::ifstream fp(labelFilepath);
  while (std::getline(fp, line)) {
    labels.push_back(line);
  }
  return labels;
}

std::vector<float> sigmoid(const std::vector<float>& m1) {
  const unsigned long vectorSize = m1.size();
  std::vector<float> output(vectorSize);
  for (unsigned i = 0; i != vectorSize; ++i) {
    output[i] = 1 / (1 + exp(-m1[i]));
  }
  return output;
}

int main(int argc, char* argv[]) {
  int inpWidth = 32;
  int inpHeight = 32;
  std::string modelFilepath{
      "./"
      "model.onnx"};
  std::string labelFilepath{
      "./"
      "classes.txt"};
  std::string imageFilepath{argv[1]};

  std::vector<std::string> labels{readLabels(labelFilepath)};
  cv::Mat image = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat blob;
  cv::Scalar mean{0.4151, 0.3771, 0.4568};
  cv::Scalar std{0.2011, 0.2108, 0.1896};
  bool swapRB = false;
  bool crop = false;
  cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(inpWidth, inpHeight), mean,
                         swapRB, crop);
  if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0) {
    cv::divide(blob, std, blob);
  }

  cv::dnn::Net net = cv::dnn::readNet(modelFilepath);
  net.setInput(blob);
  cv::Mat prob = net.forward();
  std::cout << prob << std::endl;

  // Apply sigmoid
  cv::Mat probReshaped = prob.reshape(1, prob.total() * prob.channels());
  std::vector<float> probVec =
      probReshaped.isContinuous() ? probReshaped : probReshaped.clone();
  std::vector<float> probNormalized = sigmoid(probVec);

  cv::Point classIdPoint;
  double confidence;
  minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
  int classId = classIdPoint.x;
  std::cout << " ID " << classId << " - " << labels[classId] << " confidence "
            << confidence << std::endl;
}