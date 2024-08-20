import onnxruntime as ort
import numpy as np
import cv2 as cv

#c=cv.imread("test.jpg")
c = np.zeros([32, 32, 3], np.uint8)
c=np.transpose(c,(2,1,0))
x=np.expand_dims(c, axis=0).astype(np.float32)

ort_sess = ort.InferenceSession('model.onnx')
output_onnx = ort_sess.run(None, {'input.1': x})
print(output_onnx)

net = cv.dnn.readNet("model.onnx")
net.setInput(x)
output_cvdnn = net.forward()
print(output_cvdnn)
