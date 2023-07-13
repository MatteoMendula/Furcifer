import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms

def parse_to_onnx(model):
    input = [torch.randn((1,3,300,300)).to("cuda")]
    model = model.to("cuda")
    traced_model = torch.jit.trace(model, input)    
    torch.onnx.export(traced_model,  # model being run
                        input,  # model input (or a tuple for multiple inputs)
                        "./models/head.onnx",  # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=13,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=['input'],  # the model's input names
                        output_names=['output0', 'output1', 'output2'],  # the model's output names]
                    )
    
def parse_to_trt(model, precision = 'fp32'):
    import subprocess
    print("[{}] parse_to_trt".format(precision))
    cmd = '/home/matteo/TensorRT-8.6.1.6/bin/trtexec --onnx=./models/head.onnx --saveEngine=./models/head_fp32.trt'
    if precision == 'fp16':
        cmd = '/home/matteo/TensorRT-8.6.1.6/bin/trtexec --onnx=./models/head.onnx --saveEngine=./models/head_fp16.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16'
    output = subprocess.check_call(cmd.split(' '))
    print(output)

head_model_loaded = torch.jit.load('./models/head_model.pt')
tail_model_loaded = torch.jit.load('./models/tail_model.pt')

image = Image.open("car.jpg")
transform = transforms.Compose([
    transforms.ToTensor()
])

image = transform(image)
my_image = torch.unsqueeze(torch.Tensor(image), 0)
input = torch.tensor(my_image, dtype=torch.float32)
input = input.cuda()

out_head, other_info = head_model_loaded(input)

print(out_head)
print(other_info)

out_tail_loaded = tail_model_loaded(*(*out_head, *other_info))

print(out_tail_loaded)


parse_to_onnx(head_model_loaded)

