from mobilenetv3 import mobilenetv3
import torch
import torchvision.transforms as transforms
from PIL import Image

split_position=5
bottleneck_channels=12

model_head = mobilenetv3.SplitMobileNetV3(num_classes=10, pretrained='True', split_position=split_position, bottleneck_channels=bottleneck_channels,split_name='head')
model_tail = mobilenetv3.SplitMobileNetV3(num_classes=10, pretrained='True', split_position=split_position, bottleneck_channels=bottleneck_channels,split_name='tail')
# print(model_head)
# print(model_tail)

model_head.cuda()
model_tail.cuda()

model_head = torch.nn.DataParallel(model_head).cuda()
model_tail = torch.nn.DataParallel(model_tail).cuda()

transform = transforms.Compose([
            transforms.ToTensor()
        ])

input=transform(Image.open(('../client/000000001675.jpg'))).unsqueeze(0)

output_head=model_head(input)
output=model_tail(output_head)
print(output)
print(output.shape)