import sys
import cv2 
import imutils
from yoloDet import YoloTRT
import time
import threading 
from flask import Flask
import argparse

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


class TRT_Yolov5:
    def __init__(self, args):
        self.model = YoloTRT(library="./libmyplugins.so", engine="./yolov5s.engine", conf=0.5, yolo_ver="v5")
        self.cap = None
        self.prev = 0
        self.image_index = 0

        # args 
        self.port = args.port
        self.camera_url = args.camera_url
        self.save_image = args.save_image

        print("Everything loaded")
        # threading.Thread(target=web, daemon=True).start()
        self.flask_app = FlaskApp(self.port)
        self.flask_app.start()
        print("Web server started")


    def run(self):
        while True:

            if self.flask_app.inference_running:
                if self.cap is None:
                    self.cap = cv2.VideoCapture(self.camera_url, cv2.CAP_V4L2)
                ret, frame = self.cap.read()
                frame = imutils.resize(frame, width=600)
                start_time = time.time()
                detections, self.inf_time = self.model.Inference(frame)
                time_elapsed = time.time() - start_time
                self.fps = 1/time_elapsed
                # cv2.imshow("Output", frame)
                if self.image_index % 10 == 0:
                    cv2.imwrite("./detections_imgs/img{}.jpg".format(self.image_index), frame)
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
    trt_yolov5 = TRT_Yolov5(args)
    trt_yolov5.run()