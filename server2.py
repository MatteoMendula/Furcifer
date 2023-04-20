from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from socketserver import ThreadingMixIn
import threading

class Handler(BaseHTTPRequestHandler):
    
    def _set_response_json(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        message =  threading.currentThread().getName()
        self.wfile.write(message)
        self.wfile.write('\n')
        return
    
    def do_POST(self):
        self.send_response(200)
        self.end_headers()
        message =  threading.currentThread().getName()
        print(message)
        response = {
                'message': message
            }
        self.wfile.write(json.dumps(response).encode('utf-8'))
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

if __name__ == '__main__':
    server = ThreadedHTTPServer(('localhost', 8000), Handler)
    print ('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()