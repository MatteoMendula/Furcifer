import pickle
from flask import Flask, request, jsonify
from io import BytesIO
import base64
from PIL import Image
import torch
import requests
import torchvision.transforms as transforms
import os
import json
import gzip
import zlib

# FULL
from vision_simple_nano import efficient_net_inference
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0')
model.cuda()
torch.cuda.synchronize()

# SPLIT
from inference import mobilenetv3_split_5_head_inference
from mobilenetv3 import mobilenetv3

if os.environ.get('TAIL_SERVER_URL') is None:
    os.environ['TAIL_SERVER_URL'] = 'http://10.42.0.84:8000/furcifer_split_mobilenet_v3_split5_tail'

TAIL_SERVER_URL = os.getenv('TAIL_SERVER_URL')
print("TAIL_SERVER_URL: ", TAIL_SERVER_URL)

split_position=5
bottleneck_channels=12

model_head = mobilenetv3.SplitMobileNetV3(num_classes=10, pretrained='True', split_position=split_position, bottleneck_channels=bottleneck_channels,split_name='head')
model_head.cuda()
model_head = torch.nn.DataParallel(model_head).cuda()

transform = transforms.Compose([
            transforms.ToTensor()
        ])

app = Flask(__name__)

@app.route("/furcifer_full_and_split")
def hello():
    return "Hello, World! - furcifer_full_and_split"

@app.route("/furcifer_full_and_split_full", methods=["POST"])
def handle_furcifer_full_and_split_full_post_request():
    data = request.get_json()
    img_base64 = data['image']
    img_data = base64.b64decode(img_base64)
    img_bytes = BytesIO(img_data)
    img_pil = Image.open(img_bytes)

    try:
        inference_result = efficient_net_inference(img_pil=img_pil, model=model)
    except Exception as e:
        print("Error: ", e)
        inference_result = "Error: " + str(e)

    # inference_result = efficient_net_inference(img_pil=img_pil, model='efficientnet-b0')

    response = {
        'inference_result': inference_result
    }
    
    return jsonify(response)

@app.route("/furcifer_full_and_split_head", methods=["POST"])
def handle_furcifer_full_and_split_head_post_request():
    data = request.get_json()

    img_base64 = data['image']
    print("type img_base64: ", type(img_base64))

    img_data = base64.b64decode(img_base64)
    print("----- HEAD img_data LEN -----", len(img_data))

    img_bytes = BytesIO(img_data)
    size = len(img_bytes.getbuffer())
    
    print("---- img_bytes ------", size)
    img_pil = Image.open(img_bytes)
    image_unqueezed = transform(img_pil).unsqueeze(0)

    bytes_tensor = image_unqueezed.cpu().detach().numpy().tobytes()
    stream = BytesIO(bytes_tensor)
    size = len(stream.getbuffer())
    print("type image_unqueezed: ", size, image_unqueezed.shape, type(image_unqueezed))    

    head_payload = {}
    head_payload["error"] = False

    try:
        inference = mobilenetv3_split_5_head_inference(image_unqueezed=image_unqueezed, model=model_head)     
        
        head_payload["head_inference_result"] = inference["head_inference_result"].cpu().detach().numpy()
        print("type head_inference_result: ", head_payload["head_inference_result"].dtype)
        print("shape head_inference_result: ", head_payload["head_inference_result"].shape)
        
        bytes_tensor = head_payload["head_inference_result"].tobytes()
        stream = BytesIO(bytes_tensor)
        size = len(stream.getbuffer())
        print("---- bytes_tensor size ------", size)

        compressed_bytes_gzip = gzip.compress(bytes_tensor)
        print("---- compressed_bytes_gzip size  ------", len(compressed_bytes_gzip))

        compressed_bytes_zlib = zlib.compress(bytes_tensor)
        print("---- compressed_bytes_zlib size  ------", len(compressed_bytes_zlib))

        base64_bytes = base64.b64encode(bytes_tensor)
        print("---- base64_bytes size ------", len(base64_bytes))

        buffer = BytesIO()
        torch.save(inference["head_inference_result"].cpu().detach(), buffer)
        size = len(buffer.getbuffer())
        print("---- buffer size ------", size)

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
    json_data = json.dumps(head_payload)
    print("----- HEAD PAYLOAD LEN -----", len(json_data), type(json_data))
    print("----- HEAD PAYLOAD LEN PICKLED-----", pickle.dumps(json_data).__sizeof__())
    # print("----- HEAD PAYLOAD LEN JIT-----", torch.jit.trace(json_data).__sizeof__())
    tail_response = requests.post(TAIL_SERVER_URL, headers=headers, data=json_data)

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