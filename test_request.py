"""Send real GET and POST requests; run after starting Docker or Flask."""
import json
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

EXAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    endpoint = f"{base.rstrip('/')}/predict"
    requests = [
        Request(f"{endpoint}?{urlencode(EXAMPLE)}"),
        Request(endpoint, data=json.dumps(EXAMPLE).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST"),
    ]
    results = []
    for request in requests:
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
            assert response.status == 200
            assert type(payload["target"]) is int and payload["target"] in (0, 1)
            results.append(payload)
            print(f"{request.get_method()} {request.full_url}\nHTTP {response.status}\n{json.dumps(payload)}")
    assert results[0] == results[1]


if __name__ == "__main__":
    main()
