source ./environment
g++ `pkg-config --cflags opencv4` main.cpp `pkg-config --libs opencv4` -o test
