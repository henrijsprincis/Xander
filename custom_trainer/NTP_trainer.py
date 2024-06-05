import time
import json
from datasets import load_dataset
from datasets import set_caching_enabled
set_caching_enabled(False)
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from preprocess_machine_learning.helper_functions import get_save_paths
from preprocess_machine_learning.preprocess_NTP import preprocess_data_query_NTP
from preprocess_sql.database_class import DatabaseClass
import os

with open("config.json") as f:
    config = json.load(f)

def get_data_loader(spider, config):
    input_ids = torch.tensor([data["input_ids"] for data in spider["train"]])
    attention_mask = torch.tensor([data["attention_mask"] for data in spider["train"]])
    labels = torch.tensor([data["labels"] for data in spider["train"]])
    # Create a DataLoader for batching
    train_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    return train_loader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AutoModelForLM = AutoModelForSeq2SeqLM if config["seq2seq"] else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_checkpoint"], 
        model_max_length = config["max_input_length"],
        token=os.environ["HF_TOKEN"],)
    tokenizer.pad_token = tokenizer.eos_token
    checkpoint_path, save_path = get_save_paths(config)
    database_object = DatabaseClass(config["database_path"])
    spider = load_dataset("spider")
    spider["train"] = spider["train"].select([i for i in range(0, 10)])
    spider["validation"] = spider["validation"].select([i for i in range(0, 10)])
    spider = spider.map(
        preprocess_data_query_NTP,
        fn_kwargs={
            "config":config,
            "tokenizer":tokenizer,
            "database_object":database_object,
        }, batched=False)
    if config["use_simple_sql"]:
        # Filter out examples that are too long or use self joins
        spider["train"] = spider["train"].select([i for i in range(0, len(spider["train"])) if not spider["train"][i]["tooLong"]])
        spider["validation"] = spider["validation"].select([i for i in range(0, len(spider["validation"])) if not spider["validation"][i]["tooLong"]])

    print("Total number of queries in training set: " + str(len(spider["train"])))
    print("Total number of queries in validation set: " + str(len(spider["validation"])))

    # Prepare the data
    train_loader = get_data_loader(spider, config)
    model_query = AutoModelForLM.from_pretrained(
        checkpoint_path, 
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True).to(device)
    optimizer = torch.optim.Adam(model_query.parameters(), lr=config["lr"])
    start_time = time.time()
    for epoch in range(config["epochs"]):
        for i, (input_ids_batch, attention_mask_batch, labels_batch) in enumerate(train_loader):
            input_ids = input_ids_batch.to(device)
            attention_mask = attention_mask_batch.to(device)
            labels = labels_batch.to(device)
            outputs = model_query(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print("Epoch: " + str(epoch) + " Iteration: " + str(i))
                print("Loss: " + str(loss.item()))
    print("Total time taken: " + str(time.time() - start_time))
    torch.save(model_query, save_path)

if __name__ == "__main__":
    main()