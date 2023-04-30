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

model_head.eval()
model_tail.eval()

transform = transforms.Compose([
            transforms.ToTensor()
        ])

input=transform(Image.open(('../client/000000001675.jpg'))).unsqueeze(0)

output_head=model_head(input)
outputs=model_tail(output_head)

classes_probabilies = {}

for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    prob_string = '{label:<75} ({p:.2f}%)'.format(label=idx, p=prob*100)
    classes_probabilies[idx] = prob*100

print(outputs)
print(outputs.shape)
print(classes_probabilies)