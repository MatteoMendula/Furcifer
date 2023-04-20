from http.server import SimpleHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

from io import BytesIO
import base64
from PIL import Image
import json
from socketserver import ThreadingMixIn
import torch

from vision_simple_nano import efficient_net_inference
from efficientnet_pytorch import EfficientNet

# model = EfficientNet.from_pretrained('efficientnet-b0')
# model.cuda()
# torch.cuda.synchronize()

class HTTP_handler(SimpleHTTPRequestHandler):

    # set the response header
    def _set_response_json(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    # handle the GET request
    def do_GET(self):
        self._set_response_json()
        if self.path == '/':
            self.wfile.write("[Hello world] GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        json_data = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(json_data)
        # print("received json: " + json_data)

        if self.path == '/img_object_classification':
            img_base64 = data['image']
            img_data = base64.b64decode(img_base64)
            img_bytes = BytesIO(img_data)
            img_pil = Image.open(img_bytes)
            # img.show()
            # img.save('received.jpg')

            # Classify
            # inference_result = efficient_net_inference(img_pil=img_pil, model=model)
            inference_result = efficient_net_inference(img_pil=img_pil, model='efficientnet-b0')

            response = {
                'message': 'Message received. size={} bytes'.format(content_length),
                'inference_result': inference_result
            }
        else:
            response = {
                'message': 'NO route found'
            }

        self._set_response_json()
        self.wfile.write(json.dumps(response).encode('utf-8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def run(server_class=HTTPServer, handler_class=HTTP_handler, port=8000):
    server_address = ('', port)
    # httpd = server_class(server_address, handler_class)
    httpd = ThreadedHTTPServer(server_address, handler_class)
    print('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()