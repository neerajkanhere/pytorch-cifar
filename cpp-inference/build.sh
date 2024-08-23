g++ -O3 -ggdb \
-I/opt/z/tmc3/libs/include/onnxruntime/core/session/ \
-I/opt/z/tmc3/libs/include/onnxruntime \
`pkg-config --cflags opencv4` \
main.cpp \
`pkg-config --libs opencv4` \
-lonnxruntime -lonnxruntime_providers_shared \
-o test
