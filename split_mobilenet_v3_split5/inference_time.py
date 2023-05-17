from mobilenetv3 import mobilenetv3
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import os
import time

split_position=1
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


start_time=time.time()
output_head=model_head(input_image)
torch.cuda.synchronize()
print("heading inference: ",time.time()-start_time)

start_time=time.time()
outputs=model_tail(output_head)
torch.cuda.synchronize()
print("tail inference: ",time.time()-start_time)

classes_probabilies = {}

for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    prob_string = '{label:<75} ({p:.2f}%)'.format(label=idx, p=prob*100)
    classes_probabilies[idx] = prob*100

print(classes_probabilies)