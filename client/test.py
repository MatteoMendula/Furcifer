import requests
from io import BytesIO
import base64
from PIL import Image
import json
from threading import Thread
import time

class User():
    def __init__(self,number_request,type_conenction, set_tasks, req_per_sec, ip, port, route, start_time,duration) :
        self.number_request=number_request
        self.type_connection=type_conenction
        self.set_tasks=set_tasks
        self.req_per_sec=req_per_sec
        self.ip = ip
        self.port = port
        self.route = route
        self.start_time=start_time
        self.duration=duration
        self.latencies=[]

    def send_async(self,url, json_data, headers, results):
        response = requests.post(url, data=json_data, headers=headers, timeout=10)
        results.append(response.text)

    def get_latencies(self):
        print(self.latencies)
        return self.latencies
    
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
            url = "http://{}:{}/{}".format(self.ip, self.port, self.route)
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
                self.latencies.append(time_in_ms)
                print("Time interval in milliseconds:", time_in_ms)

                print(len(results))
                if (time.time() < self.start_time+self.duration):
                    time.sleep(1)
                    send_req_per_second()  
                return self.latencies 
            
            self.latencies=send_req_per_second()

import json
from urllib.request import urlopen
import time
import threading
import numpy as np


class Logger(threading.Thread):
    def __init__(self, start_time, duration, url, metrics) :
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
            for metric in self.metrics:
                new_url = self.url + metric
                response = urlopen(new_url)
                data_json = json.loads(response.read())
                try:
                    val=data_json['data']['result'][0]['value'][1]
                    self.stored_metrics[metric].append(float(val))
                except Exception as e:
                    print("Couldn't get the data, please check the server", e)

    def get_metrics(self,start_time,end_time):
        print("len self.stored_metrics['timestamp']", len(self.stored_metrics['timestamp']))
        for metric in self.metrics:
            print("len self.stored_metrics[metric]", len(self.stored_metrics[metric]))
        
        indexes = [idx for idx, element in enumerate(self.stored_metrics['timestamp']) if (element >= start_time and element<=end_time)]
        for metric in self.metrics:
            self.stored_metrics[metric]=[self.stored_metrics[metric][j] for j in indexes]
        self.stored_metrics['timestamp'] = [self.stored_metrics['timestamp'][j] for j in indexes]
        return self.stored_metrics   


if __name__ == "__main__":
    from sys import argv
    port = 8000
    duration = 10
    ip = "localhost"
    route = "furcifer_efficientnet_b0"
    
    if len(argv) >= 2:
        port = int(argv[1])
        n_requests = int(argv[2])
        ip = argv[3]
        route = argv[4]

    #Starting logger code
    metrics = ['energon_cpu_in_power_mW','energon_gpu_in_power_mW','energon_cpu_total_usage_percentage',
        'energon_gpu_total_usage_percentage','energon_ram_used_percentage']
    prometheus_server_url = 'http://localhost:9090/api/v1/query?query='

    start_time_user = time.time()
    duration_user = 10
    num_tests = 10

    start_time_metrics = time.time()
    duration_metrics = (duration_user*num_tests) + 3
    log_metrics = Logger(time.time(), duration_metrics, prometheus_server_url, metrics)
    log_metrics.setDaemon(True)
    log_metrics.start()
    
    avg_var_metrics={}
    n_requests = 1
    for j in range(num_tests):
        print("Testing "+ str(n_requests) + " requests per second")
        start_time_user=time.time()
        user_1=User(10, 'BAD',set(), n_requests, ip, port, route, start_time_user, duration_user)
        user_1.start()

        print("Getting the metrics ")
        results_metrics=log_metrics.get_metrics(start_time_user,start_time_user+duration_user)

        temp_dict_avg={}
        temp_dict_var={}
        for i in range(len(metrics)):
            #print(results_metrics[metrics[i]])
            temp_dict_avg[metrics[i]]=sum(results_metrics[metrics[i]])/len(results_metrics[metrics[i]])
            temp_dict_var[metrics[i]]=np.var(results_metrics[metrics[i]])
            #print("Average of " +metrics[i]+" :", sum(results_metrics[metrics[i]])/len(results_metrics[metrics[i]]))
            #print("Variance of "+metrics[i]+" : %s" %(statistics.variance(results_metrics[metrics[i]])))
        temp_dict_avg['latency']=sum(user_1.get_latencies())/len(user_1.get_latencies())
        temp_dict_var['latency']=np.var(user_1.get_latencies())

        avg_var_metrics[j]={'avg':temp_dict_avg, 'var':temp_dict_var}
        print(avg_var_metrics)

        del user_1
        n_requests+=1










