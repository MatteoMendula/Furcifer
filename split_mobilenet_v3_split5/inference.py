from mobilenetv3 import mobilenetv3
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import os

split_position=5
bottleneck_channels=12

model_head = mobilenetv3.SplitMobileNetV3(num_classes=10, pretrained='True', split_position=split_position, bottleneck_channels=bottleneck_channels,split_name='head')
model_tail = mobilenetv3.SplitMobileNetV3(num_classes=10, pretrained='True', split_position=split_position, bottleneck_channels=bottleneck_channels,split_name='tail')

model_head.cuda()
model_tail.cuda()

model_head = torch.nn.DataParallel(model_head).cuda()
model_tail = torch.nn.DataParallel(model_tail).cuda()

model_head.eval()
model_tail.eval()

transform = transforms.Compose([
            transforms.ToTensor()
        ])

def get_image_size_MB(image_path):
    file_size = os.path.getsize(image_path)
    size_in_mb = file_size / (1024 * 1024)
    return size_in_mb

def get_tensor_size_MB(tensor):
    bytes_tensor = tensor.tobytes()
    stream = BytesIO(bytes_tensor)
    size_in_bytes = len(stream.getbuffer())
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb

# image size in MB
print("----- image size in MB -----")
image_path = '../client/000000001675.jpg'
print("Size of the image in MB:", get_image_size_MB(image_path))

# ---------------- no quantization
print("----- no quantization -----")

raw_image = Image.open(image_path)
input_image=transform(raw_image).unsqueeze(0)

output_head=model_head(input_image)
output_head_numpy = output_head.cpu().detach().numpy()

print("Size of the output_head_numpy in MB:", get_tensor_size_MB(output_head_numpy))

outputs=model_tail(output_head)

classes_probabilies = {}

for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    prob_string = '{label:<75} ({p:.2f}%)'.format(label=idx, p=prob*100)
    classes_probabilies[idx] = prob*100

print(classes_probabilies)

#  ---------------- quantization
print("----- quantization -----")
from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

# quantize output_head
output_head_quantized = quantize_tensor(output_head, num_bits=8)
output_head_numpy = output_head_quantized.tensor.cpu().detach().numpy()

print("Size of the output_head_numpy in MB:", get_tensor_size_MB(output_head_numpy))

# dequantize output_head
output_head_dequantized = dequantize_tensor(output_head_quantized)
output_head_dequantized_numpy = output_head_dequantized.cpu().detach().numpy()
print("Size of the output_head_dequantized in MB:", get_tensor_size_MB(output_head_dequantized_numpy))

outputs=model_tail(output_head_dequantized)

classes_probabilies = {}

for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    prob_string = '{label:<75} ({p:.2f}%)'.format(label=idx, p=prob*100)
    classes_probabilies[idx] = prob*100

print(classes_probabilies)

# calculate min square error between x and dq_x
print("----- calculate min square error between x and dq_x -----")
mse = nn.MSELoss()
loss = mse(output_head, output_head_dequantized)
print("loss", loss)