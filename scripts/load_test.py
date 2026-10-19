"""Send N random-ish requests to the API so monitoring dashboards have data.

Usage:  python scripts/load_test.py --url http://localhost:8000 -n 200
"""
import argparse
import random
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("-n", type=int, default=200)
a = p.parse_args()

ok = bad = 0
for i in range(a.n):
    body = {"age": random.randint(30, 75), "sex": random.randint(0, 1), "cp": random.randint(1, 4),
            "trestbps": random.randint(100, 180), "chol": random.randint(150, 350),
            "fbs": random.randint(0, 1), "restecg": random.randint(0, 2),
            "thalach": random.randint(90, 195), "exang": random.randint(0, 1),
            "oldpeak": round(random.uniform(0, 4), 1), "slope": random.randint(1, 3),
            "ca": random.randint(0, 3), "thal": random.choice([3, 6, 7])}
    if i % 25 == 0:
        body["cp"] = 9  # deliberately invalid -> 422, shows up in error panel
    r = requests.post(f"{a.url}/predict", json=body, timeout=5)
    ok += r.ok
    bad += not r.ok
    time.sleep(0.05)
print(f"done: {ok} ok, {bad} rejected")
