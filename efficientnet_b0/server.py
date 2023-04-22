from flask import Flask, request, jsonify
from io import BytesIO
import base64
from PIL import Image
import torch
import requests

from vision_simple_nano import efficient_net_inference
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0')
model.cuda()
torch.cuda.synchronize()

app = Flask(__name__)

@app.route("/furcifer_efficientnet_b0")
def hello():
    return "Hello, World!"

@app.route("/furcifer_efficientnet_b0", methods=["POST"])
def handle_post_request():
    data = request.get_json()
    img_base64 = data['image']
    img_data = base64.b64decode(img_base64)
    img_bytes = BytesIO(img_data)
    img_pil = Image.open(img_bytes)

    inference_result = "not implemented yet"
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

if __name__ == "__main__":
    app.run()