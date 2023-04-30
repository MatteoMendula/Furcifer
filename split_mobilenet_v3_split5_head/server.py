from flask import Flask, request, jsonify
from io import BytesIO
import base64
from PIL import Image
import torch
import requests
import torchvision.transforms as transforms
import os
import json

from inference import mobilenetv3_split_5_head_inference
from mobilenetv3 import mobilenetv3

if os.environ.get('TAIL_SERVER_URL') is None:
    os.environ['TAIL_SERVER_URL'] = 'localhost:8001/furcifer_split_mobilenet_v3_split5_tail'

TAIL_SERVER_URL = os.getenv('TAIL_SERVER_URL')

split_position=5
bottleneck_channels=12

model_head = mobilenetv3.SplitMobileNetV3(num_classes=10, pretrained='True', split_position=split_position, bottleneck_channels=bottleneck_channels,split_name='head')
model_head.cuda()
model_head = torch.nn.DataParallel(model_head).cuda()

transform = transforms.Compose([
            transforms.ToTensor()
        ])

app = Flask(__name__)

@app.route("/furcifer_split_mobilenet_v3_split5_head")
def hello():
    return "Hello, World! - head"

@app.route("/furcifer_split_mobilenet_v3_split5_head", methods=["POST"])
def handle_post_request():
    data = request.get_json()
    img_base64 = data['image']
    img_data = base64.b64decode(img_base64)
    img_bytes = BytesIO(img_data)
    img_pil = Image.open(img_bytes)
    image_unqueezed = transform(img_pil).unsqueeze(0)

    head_payload = {}
    head_payload["error"] = False

    try:
        inference = mobilenetv3_split_5_head_inference(image_unqueezed=image_unqueezed, model=model_head)
        head_payload["head_inference_result"] = inference["head_inference_result"]
        head_payload['head_inference_time'] = inference['head_inference_time']
    except Exception as e:
        print("Error: ", e)
        head_payload["error"] = True
        head_payload["head_inference_result"] = "Error: " + str(e)

    # If there was an error, interrupt the request and return the error message
    if head_payload["error"]:
        return jsonify(head_payload)

    # Send the inference result to the tail server
    headers = {'Content-Type': 'application/json'}
    tail_response = requests.post(TAIL_SERVER_URL, headers=headers, data=head_payload['head_inference_result'])

    # Parse the tail_response content to a JSON object
    tail_payload = json.loads(tail_response.content) 

    # Create a new response object
    response = {}
    response["error"] = tail_payload["error"]
    response['head_inference_time'] = head_payload['head_inference_time']
    response['tail_inference_time'] = tail_payload['tail_inference_time']
    response["inference_result"] = tail_payload["tail_inference_result"]
    
    return jsonify(response)

if __name__ == "__main__":
    app.run()