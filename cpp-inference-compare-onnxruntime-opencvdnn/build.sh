g++ -O3 -ggdb `pkg-config --cflags opencv4` main.cpp `pkg-config --libs opencv4` -lonnxruntime -o test
