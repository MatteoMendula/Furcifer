from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import numpy as np

from inference import mobilenetv3_split_5_tail_inference
from mobilenetv3 import mobilenetv3

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
    return "Hello, World! - tail"

@app.route("/furcifer_split_mobilenet_v3_split5_tail", methods=["POST"])
def handle_post_request():

    try:
        data = request.get_json()
    except Exception as e:
        print("Error: ", e)
        return jsonify({"error": True, "tail_inference_result": "Error: " + str(e)})
    
    head_inference_result = np.array(data['head_inference_result'])
    head_inference_result = torch.from_numpy(head_inference_result).cuda().float()

    response = {}
    response["error"] = False

    try:
        inference = mobilenetv3_split_5_tail_inference(head_inference_result=head_inference_result, model=model_tail)
        print("inference", inference)
        response["tail_inference_result"] = inference["tail_inference_result"]
        response['tail_inference_time'] = inference['tail_inference_time']
    except Exception as e:
        print("Error: ", e)
        response["error"] = True
        response["tail_inference_result"] = "Error: " + str(e)
    
    return jsonify(response)

if __name__ == "__main__":
    app.run()