import json
from neural_query_checker.dataset_generator import get_dataset

with open("config.json") as f:
    config = json.load(f)

dataset = get_dataset(config)
print("Success")