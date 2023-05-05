import requests
import json
import subprocess
import argparse
import cv2
from flask import Flask, request, jsonify
import threading
import time
import base64
import os
from prometheus_client import start_http_server, Gauge, Info
from io import BytesIO

# ------------------------- PARAMETERS -------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--port",
    nargs="?",
    default=7999,
    type=int,
    help="Port of the server",
)
parser.add_argument(
    "-hi",
    "--host_ip",
    nargs="?",
    default="localhost",
    type=str,
    help="Host of the server",
)
parser.add_argument(
    "-i",
    "--inference_port",
    nargs="?",
    default=9878,
    type=int,
    help="Port of the inference metrics exporter",
)
parser.add_argument(
    "-s",
    "--server_url",
    nargs="?",
    default="http://localhost:8000/furcifer_efficientnet_b0",
    type=str,
    help="URL of the server",
)
args = parser.parse_args()

# ------------------------- WEBCAM SAMPLING AND SENDING -------------------------
class WebcamRequestSender:
    def __init__(self, frame_rate, server_url, inference_metric_exporter):
        self.frame_rate = frame_rate
        self.server_url = server_url
        self.inference_metric_exporter = inference_metric_exporter
        self.is_sampling_from_camera_started = False
        self.frames_to_send = []
        self.vid = cv2.VideoCapture(4)
        self.sample_loop_thread = threading.Thread(target=self.sample_from_camera, args=())
        self.sender_loop_thread = threading.Thread(target=self.send_camera_requests, args=())
        self.sample_loop_thread.start()
        self.sender_loop_thread.start()
        self.last_sending_timestamp = time.time()

        self.current_state = {}
        self.current_state["device_name"] = "furcifer_webcam_request_sender"
        self.current_state["inference_server_url"] = self.server_url
        self.current_state["is_sampling_from_camera_started"] = str(self.is_sampling_from_camera_started)
        self.current_state["timeout_error"] = "False"
        self.current_state["inference_results"] = "None"

        self.inference_metric_exporter.set_device_info(self.current_state)

    def set_frame_rate(self, frame_rate):
        self.frame_rate = int(frame_rate)
    
    def set_server_url(self, server_url):
        self.server_url = server_url
        self.current_state["inference_server_url"] = self.server_url
        self.inference_metric_exporter.set_device_info(self.current_state)

    def set_is_sampling_from_camera_started(self, started):
        self.is_sampling_from_camera_started = started
        self.current_state["is_sampling_from_camera_started"] = str(self.is_sampling_from_camera_started)
        self.inference_metric_exporter.set_device_info(self.current_state)

    def sample_from_camera(self):
        while True:
            if self.is_sampling_from_camera_started:
                prev = 0
                while self.is_sampling_from_camera_started:
                    time_elapsed = time.time() - prev
                    if time_elapsed > 1./self.frame_rate:
                        prev = time.time()
                        _, frame = self.vid.read()
                        self.frames_to_send.append(frame)
        
    def send_async(self, url, json_data, headers, results):
        try:
            response = requests.post(url, data=json_data, headers=headers, timeout=10)
            size_b = len(response.content)
            size_MB = size_b / 1000000
            self.inference_metric_exporter.set_size_last_pkt_received(size_MB)
            results.append(response.text)
        except requests.exceptions.Timeout:
            print("Request timed out after 10 seconds.")
            if "localhost" in self.server_url:
                self.current_state["timeout_error"] = "LOCALHOST"
            else:
                self.current_state["timeout_error"] = "EDGE"
            self.inference_metric_exporter.set_device_info(self.current_state)
    
    def send_camera_requests(self):
        while True:
            start_time = time.time()
            if self.is_sampling_from_camera_started and start_time - self.last_sending_timestamp >= 1:
                print("len(self.frames_to_send)", len(self.frames_to_send))
                self.inference_metric_exporter.set_frames_per_second(len(self.frames_to_send))
                
                print("start_time", start_time)
                print("self.self.last_sending_timestamp", self.last_sending_timestamp)
                self.inference_metric_exporter.set_time_passed_since_last_sending(start_time - self.last_sending_timestamp)
                self.last_sending_timestamp = start_time 

                max_frames_to_send = min(self.frame_rate, len(self.frames_to_send))
                threads = [None] * max_frames_to_send
                results = []
                for i, frame in enumerate(self.frames_to_send[-max_frames_to_send:]):
                    _, buffer = cv2.imencode('.jpg', frame)
                    stream = BytesIO(buffer)
                    size_b = len(stream.getbuffer())
                    size_MB = size_b / 1000000
                    print("---- SENT img_bytes (MB) ------", size_MB)
                    self.inference_metric_exporter.set_size_last_pkt_sent(size_MB)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    headers = {"Content-type": "application/json"}
                    data = {
                        "type_task": "IMAGE_CLASS",
                        "image": img_base64
                    }
                    json_data = json.dumps(data)
                    threads[i] = threading.Thread(target=self.send_async, args=(self.server_url, json_data, headers, results,))
                    threads[i].start()

                self.frames_to_send = []
                # wait for all the threads to finish
                for i in range(len(threads)):
                    threads[i].join()
                print("results", " ".join(results))
                self.current_state["inference_results"] = str(results)
                self.inference_metric_exporter.set_device_info(self.current_state)
                end_time = time.time()
                time_in_ms = (end_time - start_time) * 1000
                print("latency ", time_in_ms)
                self.inference_metric_exporter.set_latency(time_in_ms)
                time.sleep(1)

# ------------------------- INFERENCE METRICS PROMETHEUS EXPORTER -------------------------
class InferenceMetricsExporter(threading.Thread):
    def __init__(self, app_port=9878):
        super(InferenceMetricsExporter, self).__init__()
        self.daemon = True 
        self.app_port = app_port
        self.device_info = Info("furcifer_data_export_info", "Device info")
        self.latency = Gauge("furcifer_latency_ms", "Latency in milli seconds")
        self.frames_per_second = Gauge("furcifer_fps", "Frames per second")
        self.size_last_pkt_sent = Gauge("furcifer_size_last_pkt_sent_MB", "Size of the last packet sent in MB")
        self.size_last_pkt_received = Gauge("furcifer_size_last_pkt_received_MB", "Size of the last packet received in MB")
        self.time_passed_since_last_sending = Gauge("furcifer_time_passed_since_last_sending", "Time passed since last sending")

        self.device_name = "not_set"
        self.inference_server_url = "not_set"
        self.is_sampling_from_camera_started = False
        self.init_server()

    def init_server(self):
        self.device_info.info(
            {
                "device_name": self.device_name,
                "inference_server_url": self.inference_server_url,
                "is_sampling_from_camera_started": str(self.is_sampling_from_camera_started),
            }
        )
        self.latency.set(-1)
        self.frames_per_second.set(-1)
        self.size_last_pkt_sent.set(-1)
        self.size_last_pkt_received.set(-1)
        self.time_passed_since_last_sending.set(-1)
        start_http_server(self.app_port)

    def set_latency(self, latency):
        self.latency.set(latency)

    def set_frames_per_second(self, fps):
        self.frames_per_second.set(fps)

    def set_size_last_pkt_sent(self, size):
        self.size_last_pkt_sent.set(size)

    def set_size_last_pkt_received(self, size):
        self.size_last_pkt_received.set(size)

    def set_device_info(self, device_info):
        self.device_info.info(device_info)

    def set_time_passed_since_last_sending(self, time_passed):
        self.time_passed_since_last_sending.set(time_passed)

# ------------------------- COMMAND WEB SERVER -------------------------
class MyFlaskApp(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_url_rule('/', 'hello', self.hello, methods=['GET'])
        self.add_url_rule('/execute_command', 'execute_command', self.execute_command, methods=['POST'])
        self.add_url_rule('/sample_camera_and_send_image_for_inference', 'sample_camera_and_send_image_for_inference', self.sample_camera_and_send_image_for_inference, methods=['POST'])
        self.add_url_rule('/stop_camera_sampling', 'stop_camera_sampling', self.stop_camera_sampling, methods=['POST'])
        self.add_url_rule('/set_server_url', 'set_server_url', self.set_server_url, methods=['POST'])
        self.webcam_request_sender = None
        self.inference_metric_exporter = None

    def init(self):
        print("Flask server started!")
        if self.inference_metric_exporter == None:
            self.inference_metric_exporter = InferenceMetricsExporter()
        # inference_metric_exporter.setDaemon(True)
        # self.inference_metric_exporter.start()

        # wait for the InferenceMetricsExporter server to start
        print("Waiting for the InferenceMetricsExporter to start...")
        for i in range(5):
            print(5-i)
            time.sleep(1)
        print("InferenceMetricsExporter ready!")

        if self.webcam_request_sender == None:
            self.webcam_request_sender = WebcamRequestSender(-1, args.server_url, self.inference_metric_exporter)
        print("WebcamRequestSender ready to start! - self.is_sampling_from_camera_started: {}".format(self.webcam_request_sender.is_sampling_from_camera_started))

    def hello(self):
        return "Hello, World! This is command server!"
    
    def execute_command(self):
        data = request.get_json()
        response_data = {'received': data}
        if not "key" in data.keys() or data["key"] != "ubicomp2023":
            response_data["result"] = "No key in the request"
            return jsonify(response_data)
        if not "command" in data.keys():
            response_data["result"] = "No command in the request"
        print("command:", data["command"])
        result = self.run_command_and_get_output(data["command"])
        response_data["result"] = result
        return jsonify(response_data)

    def sample_camera_and_send_image_for_inference(self):
        data = request.get_json()
        response_data = {'received': data}
        if not "key" in data.keys() or data["key"] != "ubicomp2023":
            response_data["result"] = "No key in the request"
            return jsonify(response_data)
        if not "n_frames" in data.keys():
            response_data["result"] = "No n_frames in the request"
            return jsonify(response_data)
        print("n_frames:", data["n_frames"])
        self.webcam_request_sender.set_frame_rate(int(data["n_frames"]))
        if not self.webcam_request_sender.is_sampling_from_camera_started:
            self.webcam_request_sender.set_is_sampling_from_camera_started(True)
        response_data["result"] = "Camera sampling started!"
        return jsonify(response_data)
    
    def stop_camera_sampling(self):
        data = request.get_json()
        response_data = {'received': data}
        if not "key" in data.keys() or data["key"] != "ubicomp2023":
            response_data["result"] = "No key in the request"
            return jsonify(response_data)
        self.webcam_request_sender.set_is_sampling_from_camera_started(False)
        response_data["result"] = "Camera sampling stopped!"
        return jsonify(response_data)

    def set_server_url(self):
        data = request.get_json()
        response_data = {'received': data}
        if not "key" in data.keys() or data["key"] != "ubicomp2023":
            response_data["result"] = "No key in the request"
            return jsonify(response_data)
        if not "server_url" in data.keys():
            response_data["result"] = "No server_url in the request"
            return jsonify(response_data)
        print("server_url:", data["server_url"])
        self.webcam_request_sender.set_server_url(data["server_url"])
        response_data["result"] = "Server URL set!"
        return jsonify(response_data)

    def run_command_and_get_output(self, command):
        # check if command is empty
        command_output = {}
        if not command:
            command_output["out_value"] = "No command to run"
            command_output["error"] = True
        _command = command.split()

        # execute the command 
        try:
            command_out_value_bytes = subprocess.run(_command, stdout=subprocess.PIPE, encoding='utf-8')

            command_out_value_string_cleaned = str(command_out_value_bytes.stdout)[:-1].strip()
            command_output["out_value"] = command_out_value_string_cleaned
            command_output["error"] = False
        except Exception as e:
            print("Exception", e.output)
            command_output["out_value"] = "Error in running command %s - output value %s" % command % e.output
            command_output["error"] = True

        return command_output   

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        if not self.debug or os.getenv('WERKZEUG_RUN_MAIN') == 'true':
            with self.app_context():
                self.init()
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)

if __name__ == '__main__':
    app = MyFlaskApp(__name__)
    print("args.port", args.port)
    # print("args.host", args.host)
    app.run(port=args.port, host="0.0.0.0")