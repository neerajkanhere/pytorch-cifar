g++ -O3 -ggdb -I/opt/z/tmc3/libs/include/onnxruntime/core/session/ `pkg-config --cflags opencv4` main.cpp `pkg-config --libs opencv4` -lonnxruntime -o test
