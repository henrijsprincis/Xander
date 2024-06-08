import time
import json
from datasets import load_dataset
from datasets import set_caching_enabled
set_caching_enabled(False)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from preprocess_machine_learning.helper_functions import get_save_paths
from preprocess_machine_learning.query_preprocessor import preprocess_query
from preprocess_sql.database_class import DatabaseClass
from eval.process_sql import SimpleSQL_to_SQL
from eval.getNumberCorrect import beam_search_with_checks, get_error_type_prediction_query
import os

# 1. Use spider dataset as base
# 2. Load the base model
# 3. Use the base model to generate realistic queries on the training dataset

def get_dataset(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    database_object = DatabaseClass(config["database_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_checkpoint"], 
        model_max_length = config["max_input_length"],
        token=os.environ["HF_TOKEN"],)
    tokenizer.pad_token = tokenizer.eos_token
    base_dataset = load_dataset("spider")
    base_dataset["train"] = base_dataset["train"].select([i for i in range(0, 2)])#comment out this line to use the full dataset
    base_dataset["validation"] = base_dataset["validation"].select([i for i in range(0, 2)])
    base_dataset = base_dataset.map(
        preprocess_query,
        fn_kwargs={
            "config":config,
            "tokenizer":tokenizer,
            "database_object":database_object,
        }, batched=False)


    checkpoint_path, save_path = get_save_paths(config)
    AutoModelForLM = AutoModelForSeq2SeqLM if config["seq2seq"] else AutoModelForCausalLM
    model_query = AutoModelForLM.from_pretrained(
        checkpoint_path, 
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True).to(device)
    output_dataset = {"train": [], "validation": []}
    simple_sql_fn = SimpleSQL_to_SQL if config["use_simple_sql"] else None
    for train_or_validation in ["train", "validation"]:
        for data in base_dataset[train_or_validation]:
            queries = beam_search_with_checks(model_query, data, tokenizer, config, 4, False, 4)#TODO replace the max output length
            for query in queries:
                query_labelled = get_error_type_prediction_query(gold_query=data, 
                                                                 predicted_query=query[0], 
                                                                 tokenizer=tokenizer, 
                                                                 simple_sql_fn=simple_sql_fn,
                                                                 config=config)
                output_dataset[train_or_validation].append(query_labelled)
    return output_dataset
    