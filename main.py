import json
from training_and_evaluation.custom_trainer import main as NTP_main
from training_and_evaluation.evaluate_model import main as evaluate_main

with open("config.json") as f:
    config = json.load(f)

if config["train_model"]:
    NTP_main()

if config["evaluate_model"]:
    evaluate_main()