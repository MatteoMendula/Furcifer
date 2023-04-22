import requests
from io import BytesIO
import base64
from PIL import Image
import json

# Open the image file
img_file = "000000001675.jpg"
with open(img_file, "rb") as f:
    img_data = f.read()

# Create a BytesIO object from the image data
img_bytes = BytesIO(img_data)

# Convert the image data to a base64-encoded string
img_base64 = base64.b64encode(img_data).decode('utf-8')

# Make a POST request with the base64-encoded image string as the request body
url = "http://localhost:8000/img_object_classification"
# headers = {"Content-type": "text/plain"}
headers = {"Content-type": "application/json"}

data = {
    "name": "John Doe",
    "age": 30,
    "image": img_base64
}
json_data = json.dumps(data)

response = requests.post(url, data=json_data, headers=headers)

# Print the response from the server
print(response.text)