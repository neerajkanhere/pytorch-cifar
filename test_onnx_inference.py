from onnxruntime import InferenceSession
import numpy as np
import time

#sess = InferenceSession("./model.onnx", providers=['CUDAExecutionProvider'])
sess = InferenceSession("./model.onnx")

t0 = time.time()
num_iter = 1000
for i in range(num_iter):
    output = sess.run(None, {"input.1": np.random.rand(1, 3, 32, 32).astype(np.float32)})

dt = time.time() - t0
print(f"total: {int(dt)} seconds, per frame: {dt * 1000 / num_iter} ms")
