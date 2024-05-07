import json
from custom_trainer.NTP_trainer import main as NTP_main

with open("config.json") as f:
    config = json.load(f)

if config["train_model"]:
    NTP_main()