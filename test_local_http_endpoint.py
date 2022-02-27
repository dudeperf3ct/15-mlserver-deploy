import json
import requests


inputs_string= json.dumps('i like you!')

inference_request = {
    "inputs": [
        {
          "name": "text",
          "shape": [len(inputs_string)],
          "datatype": "BYTES",
          "data": [inputs_string]
        }
    ]
}

endpoint = "http://localhost:8080/v2/models/sentiment-analysis/infer"
response = requests.post(endpoint, json=inference_request)

print(response.json())