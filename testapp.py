import requests
import json

# Define the JSON payload
data = {
    "start_time": "23:00",  
    "lat": -37.860,  
    "lng": 145.091,  
    "model": "cnn"  
}

# Convert the data to JSON format
json_data = json.dumps(data)

# Set the URL of your FastAPI server
url = "http://127.0.0.1:8000"  

# Make a POST request to the server
response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})

# Check the response
if response.status_code == 200:
    flow_prediction = response.json()
    print("Flow Prediction:", flow_prediction)
else:
    print("Error:", response.status_code, response.text)
