import time
import json
from datasets import load_dataset
from datasets import set_caching_enabled
set_caching_enabled(False)
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from preprocess_machine_learning.helper_functions import get_save_paths
from preprocess_machine_learning.query_preprocessor import preprocess_query
from preprocess_sql.database_class import DatabaseClass
import os

# 1. Use spider dataset as base
# 2. Load the base model
# 3. Use the base model to generate realistic queries on the training dataset

def get_dataset(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    database_object = DatabaseClass(config["database_path"])
    base_dataset = load_dataset("spider")
    base_dataset["train"] = base_dataset["train"].select([i for i in range(0, 10)])#comment out this line to use the full dataset
    base_dataset["validation"] = base_dataset["validation"].select([i for i in range(0, 10)])
    base_dataset = base_dataset.map(
        preprocess_query,
        fn_kwargs={
            "config":config,
            "tokenizer":tokenizer,
            "database_object":database_object,
        }, batched=False)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_checkpoint"], 
        model_max_length = config["max_input_length"],
        token=os.environ["HF_TOKEN"],)
    tokenizer.pad_token = tokenizer.eos_token
    checkpoint_path, save_path = get_save_paths(config)
    AutoModelForLM = AutoModelForSeq2SeqLM if config["seq2seq"] else AutoModelForCausalLM
    model_query = AutoModelForLM.from_pretrained(
        checkpoint_path, 
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True).to(device)
    