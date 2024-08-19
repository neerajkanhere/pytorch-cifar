source ./environment
g++ -O0 -ggdb `pkg-config --cflags opencv4` main.cpp -L/opt/z/onnxruntime/build/Linux/RelWithDebInfo -lonnxruntime `pkg-config --libs opencv4` -o test
