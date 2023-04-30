import requests
from io import BytesIO
import base64
from PIL import Image
import json
from threading import Thread
import time

class User():
    def __init__(self,number_request,type_conenction, set_tasks, req_per_sec, port) :
        self.number_request=number_request
        self.type_connection=type_conenction
        self.set_tasks=set_tasks
        self.req_per_sec=req_per_sec
    def send_async(self,url, json_data, headers, results):
        response = requests.post(url, data=json_data, headers=headers, timeout=10)
        results.append(response.text)
    def start(self):
        while(True):
            # Open the image file
            img_file = "000000001675.jpg"
            # img_file = "000000000359.jpg"
            with open(img_file, "rb") as f:
                img_data = f.read()
            # Create a BytesIO object from the image data
            img_bytes = BytesIO(img_data)
            # Convert the image data to a base64-encoded string
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            # Make a POST request with the base64-encoded image string as the request body
            #url = "http://172.24.140.180:8000/img_object_classification"
            url = "http://128.195.55.244:{}/furcifer_efficientnet_b0".format(port)
            # headers = {"Content-type": "text/plain"}
            headers = {"Content-type": "application/json"}
            data = {
                "type_task": "IMAGE_CLASS",
                "image": img_base64
            }
            json_data = json.dumps(data)
            def send_req_per_second():
                start_time = time.time()
                threads = [None] * self.req_per_sec
                results = []
                for i in range(len(threads)):
                    threads[i] = Thread(target=self.send_async, args=(url, json_data, headers, results,))
                    threads[i].start()
                for i in range(len(threads)):
                    threads[i].join()
                print(" ".join(results))
                end_time = time.time()
                time_in_ms = (end_time - start_time) * 1000
                print("Time interval in milliseconds:", time_in_ms)

                print(len(results))

                time.sleep(1)
                send_req_per_second()
            send_req_per_second()
if __name__ == "__main__":
    from sys import argv
    n_requests = 5
    port = 8000
    if len(argv) >= 2:
        port=int(argv[1])
        n_requests=int(argv[2])
    user_1=User(10, 'BAD',set(), n_requests,port)
    user_1.start()












