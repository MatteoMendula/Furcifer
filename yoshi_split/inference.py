import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms

head_model_loaded = torch.jit.load('head_model.pt')
tail_model_loaded = torch.jit.load('tail_model.pt')

image = Image.open("car.jpg")
transform = transforms.Compose([
    transforms.ToTensor()
])

image = transform(image)
my_image = torch.unsqueeze(torch.Tensor(image), 0)
input = torch.tensor(my_image, dtype=torch.float32)
input = input.cuda()

out_head, other_info = head_model_loaded(input)
out_tail_loaded = tail_model_loaded(*(*out_head, *other_info))

print(out_tail_loaded)