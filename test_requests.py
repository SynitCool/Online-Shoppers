import requests
import json

data = {
    "input": [
        0,
        0,
        0,
        0,
        1,
        0,
        0.2,
        0.2,
        0,
        0,
        "Feb",
        1,
        1,
        1,
        1,
        "Returning_Visitor",
        False,
    ]
}

data = json.dumps(data)
url = "http://127.0.0.1:5000/api_test"

print(requests.post(url, data).json())
