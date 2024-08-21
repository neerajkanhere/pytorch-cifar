'''Train CIFAR10 with PyTorch.'''
import torch

import os

from models import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = TvNet()
net = net.to(device)

if device == 'cuda':
  net = torch.nn.DataParallel(net)

print('==> loading checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True) #random input tensor

dev = torch.device('cpu')
net.to(dev)
dummy_input.to(dev)
torch_out = torch.onnx.export(net.module, dummy_input, "model.onnx", export_params=True)
