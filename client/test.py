import requests
from io import BytesIO
import base64
from PIL import Image
import json
from threading import Thread
import time

class User():
    def __init__(self,number_request,type_conenction, set_tasks, req_per_sec, port, start_time,duration) :
        self.number_request=number_request
        self.type_connection=type_conenction
        self.set_tasks=set_tasks
        self.req_per_sec=req_per_sec
        self.start_time=start_time
        self.duration=duration
    def send_async(self,url, json_data, headers, results):
        response = requests.post(url, data=json_data, headers=headers, timeout=10)
        results.append(response.text)
    def start(self):
        while(time.time() < self.start_time+self.duration):
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
            url = "http://localhost:{}/furcifer_efficientnet_b0".format(port)
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
                if (time.time() < self.start_time+self.duration):
                    time.sleep(1)
                    send_req_per_second()   
            send_req_per_second()

import json
from urllib.request import urlopen
import time
import threading
import statistics


class Logger(threading.Thread):
    def __init__(self,start_time,duration,url,metrics) :
        super(Logger, self).__init__()
        self.start_time=start_time
        self.duration=duration
        self.url=url
        self.metrics=metrics
        self.stored_metrics={}

        for e in self.metrics:
            self.stored_metrics[e]=[]
        self.stored_metrics['timestamp']=[]

    def run(self):
        while(time.time() < self.start_time+self.duration):
            self.stored_metrics['timestamp'].append(time.time())
            for i in range(len(self.metrics)):
                new_url=self.url+self.metrics[i]
                response = urlopen(new_url)
                data_json = json.loads(response.read())
                try:
                    val=data_json['data']['result'][0]['value'][1]
                    self.stored_metrics[self.metrics[i]].append(float(val))
                    #print("###################################")
                    #print(data_json['data']['result'][0]['value'])
                except:
                    print("Couldn't get the data, please check the server")
            #end_time=time.time()
            #print(end_time-self.start_time)

    def get_metrics(self,start_time,end_time):
        indexes=[idx for idx, element in enumerate(self.stored_metrics['timestamp']) if (element >= start_time and element<=end_time)]
        #print("max indexes: ", max(indexes))
        for i in range(len(self.metrics)):
            #print("len of metrics" ,len(self.stored_metrics[self.metrics[i]]))
            self.stored_metrics[self.metrics[i]]=[self.stored_metrics[self.metrics[i]][j] for j in indexes]
        return self.stored_metrics   


if __name__ == "__main__":
    from sys import argv
    n_requests = 5
    port = 8000
    duration=10

    #Starting logger code
    metrics=['energon_cpu_in_power_mW','energon_gpu_in_power_mW','energon_cpu_total_usage_percentage',
        'energon_gpu_total_usage_percentage','energon_ram_total_bytes','energon_ram_percent_used_percentage']
    url='http://localhost:9090/api/v1/query?query='

    start_time_user=time.time()
    duration_user=10

    start_time_metrics=time.time()
    duration_metrics=duration_user+3
    log_metrics=Logger(time.time(),duration_metrics,url,metrics)
    log_metrics.setDaemon(True)
    log_metrics.start()
    
    if len(argv) >= 2:
        port=int(argv[1])
        n_requests=int(argv[2])
    
    user_1=User(10, 'BAD',set(), n_requests,port, start_time_user, duration_user)
    user_1.start()

    results_metrics=log_metrics.get_metrics(start_time_user,start_time_user+duration_user)

    for i in range(len(metrics)):
        #print(results_metrics[metrics[i]])
        print("Average of " +metrics[i]+" :", sum(results_metrics[metrics[i]])/len(results_metrics[metrics[i]]))
        print("Variance of "+metrics[i]+" : %s" %(statistics.variance(results_metrics[metrics[i]])))











