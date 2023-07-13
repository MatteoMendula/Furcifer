from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2 
import time
import threading 
from flask import Flask, request
import argparse
import numpy as np
import requests
import time
import argparse
from PIL import Image
from io import BytesIO
from splitDet import SplitDet

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        default=5000,
        type=int,
        help="Port of the server",
    )
    parser.add_argument(
        "-c",
        "--camera_url",
        default="/dev/video0",
        type=str,
        help="Camera url",
    )
    parser.add_argument(
        "-s",
        "--save_image",
        default=True,
        type=bool,
        help="Save image boolean",
    )
    parser.add_argument(
        "-eu",
        "--edge_url",
        default="http://localhost:3000/predict",
        type=str,
        help="URL of the server",
    )
    return parser.parse_args()

class FlaskApp(threading.Thread):
    def __init__(self, port=5000):
        super(FlaskApp, self).__init__()
        self.app = Flask(__name__)
        self.port = port
        self.inference_running = False
        self.release_camera = False

        @self.app.after_request
        def add_cors_headers(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response

        @self.app.route('/')
        def root():
            return 'Hello, World!'
        
        @self.app.route('/start', methods=['POST'])
        def start():
            self.inference_running = True
            self.release_camera = False
            return 'Started'
        
        @self.app.route('/stop', methods=['POST'])
        def stop():
            self.inference_running = False
            return 'Stopped'
        
        @self.app.route('/release', methods=['POST'])
        def release():
            self.release_camera = True
            return 'Released'

    def run(self):
        self.app.run(port=self.port, host="0.0.0.0")


class SplitTRT():
    def __init__(self, args):
        self.model = SplitDet(engine_path="./head_fp16.trt", precision="fp16")
        self.cap = None
        self.prev = 0
        self.image_index = 0
        self.coco_od_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

        # args 
        self.port = args.port
        self.camera_url = args.camera_url
        self.save_image = args.save_image
        self.edge_url = args.edge_url

        print("Everything loaded")
        self.flask_app = FlaskApp(self.port)
        self.flask_app.start()
        print("Web server started")

    def plot_results(self, detection_result, image):
        fig, ax = plt.subplots(1)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        bboxes = detection_result["boxes"]
        classes = detection_result["labels"]
        confidences = detection_result["scores"]
        for idx in range(len(bboxes)):
            if confidences[idx] < 0.7:
                continue

            if classes[idx] > len(self.coco_od_classes):
                continue

            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(self.coco_od_classes[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
        plt.axis('off')
        if self.image_index % 10 == 0:
            plt.savefig("./detections_imgs/img{}.jpg".format(self.image_index), bbox_inches='tight')
        self.image_index += 1
        plt.close()

    def send_encoder_output(self, prediction):
        start_time = time.time()
        buffer_0 = BytesIO()
        buffer_1 = BytesIO()
        buffer_2 = BytesIO()
        buffer_3 = BytesIO()
        buffer_4 = BytesIO()
        buffer_5 = BytesIO()
        np.save(buffer_0, prediction[0].astype(np.uint8))
        buffer_0.seek(0)
        np.save(buffer_1, prediction[1])
        buffer_1.seek(0)
        np.save(buffer_2, prediction[2])
        buffer_2.seek(0)
        np.save(buffer_3, prediction[3])
        buffer_3.seek(0)
        np.save(buffer_4, prediction[4])
        buffer_4.seek(0)
        np.save(buffer_5, prediction[5])
        buffer_5.seek(0)
        files = {"0": buffer_0, "1": buffer_1, "2": buffer_2, "3": buffer_3, "4": buffer_4, "5": buffer_5}
        return_value = {}
        return_value["error"] = True
        try:
            response = requests.post(self.edge_url, files=files, timeout=1)
            response.raise_for_status()
            return_value["latency"] = [time.time() - start_time]
            if response.status_code == 200:
                return_value["error"] = False
                response_json = response.json()
                return_value['detection'] = response_json['detection']
        except Exception as errh:
            print("HTTP Error: ", errh)
        return return_value

    def run(self):
        while True:
            if self.flask_app.inference_running:
                if self.cap is None:
                    self.cap = cv2.VideoCapture(self.camera_url, cv2.CAP_V4L2)
                ret, frame = self.cap.read()
                image = Image.fromarray(frame)
                start_time = time.time()
                split_det = SplitDet(engine_path="./head_fp16.trt", precision="fp16")
                prediction,self.inf_time = split_det.inference(image)
                edge_response = self.send_encoder_output(prediction)
                time_elapsed = time.time() - start_time
                self.fps = 1/time_elapsed

                if not edge_response["error"] and self.image_index % 10 == 0:
                    self.plot_results(edge_response['detection'], frame)
                self.image_index += 1
                # key = cv2.waitKey(1)
                # if key == ord('q'):
                #     break
            if self.flask_app.release_camera and self.cap is not None:
                self.cap.release()
                self.cap = None
                cv2.destroyAllWindows()
            
if __name__ == "__main__":
    args = parse_arguments()
    trt_split = SplitTRT(args)
    trt_split.run()