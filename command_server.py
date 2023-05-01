import http.server
import json
import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--port",
    nargs="?",
    default=7999,
    type=int,
    help="Port of the server",
)
args = parser.parse_args()

def run_command_and_get_output(command):
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

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        data = {'message': 'Hello, world!'}
        json_data = json.dumps(data).encode('utf-8')
        self.wfile.write(json_data)
        
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length'))
        post_data = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(post_data)
        response_data = {'received': data}

        if "command" in data.keys() and "key" in data.keys() and data["key"] == "ubicomp2023":
            print("command:", data["command"])
            result = run_command_and_get_output(data["command"])
            response_data["result"] = result
        else:
            response_data["result"] = "Invalid key or command"

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        json_response = json.dumps(response_data).encode('utf-8')
        self.wfile.write(json_response)

if __name__ == '__main__':
    server_address = ('', args.port)
    httpd = http.server.HTTPServer(server_address, MyHandler)
    print("Server started on port {}..".format(args.port))
    httpd.serve_forever()