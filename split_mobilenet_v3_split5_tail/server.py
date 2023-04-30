from flask import Flask, request, jsonify
from io import BytesIO
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

from inference import mobilenetv3_split_5_tail_inference
from mobilenetv3 import mobilenetv3

os.environ['TAIL_SERVER_URL'] = 'localhost:8001/furcifer_split_mobilenet_v3_split5_tail'
TAIL_SERVER_URL = os.getenv('TAIL_SERVER_URL')

split_position=5
bottleneck_channels=12

model_tail = mobilenetv3.SplitMobileNetV3(num_classes=10, pretrained='True', split_position=split_position, bottleneck_channels=bottleneck_channels,split_name='tail')
model_tail.cuda()
model_tail = torch.nn.DataParallel(model_tail).cuda()

transform = transforms.Compose([
            transforms.ToTensor()
        ])

app = Flask(__name__)

@app.route("/furcifer_split_mobilenet_v3_split5_tail")
def hello():
    return "Hello, World! - head"

@app.route("/furcifer_split_mobilenet_v3_split5_tail", methods=["POST"])
def handle_post_request():
    data = request.get_json()
    head_inference_result = data['head_inference_result']

    response = {}
    response["error"] = False

    try:
        response["tail_inference_result"] = mobilenetv3_split_5_tail_inference(head_inference_result=head_inference_result, model=model_tail)
    except Exception as e:
        print("Error: ", e)
        response["error"] = True
        response["tail_inference_result"] = "Error: " + str(e)
    
    return jsonify(response)

if __name__ == "__main__":
    app.run()