from eval import process_sql
import time
import json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments
from preprocess_machine_learning.helper_functions import get_save_paths
from preprocess_machine_learning.query_preprocessor import preprocess_query
from preprocess_sql.database_class import DatabaseClass
from eval.getNumberCorrect import evaluate_model


with open("config.json") as f:
    config = json.load(f)

#import torch_directml#torch_directml.device()#"cuda" if torch.cuda.is_available() else "cpu" (useful for AMD GPUs)
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    simple_sql_fn = process_sql.SimpleSQL_to_SQL if config["use_simple_sql"] else None
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"], model_max_length = config["max_input_length"])
    _, save_path = get_save_paths(config)
    database_object = DatabaseClass(config["database_path"])
    spider = load_dataset("spider")
    spider = spider.map(
        preprocess_query,
        fn_kwargs={
            "config":config,
            "tokenizer":tokenizer,
            "database_object":database_object,
        }, batched=False)
    spider["validation"] = spider["validation"].select([i for i in range(0, len(spider["validation"])) if not spider["validation"][i]["tooLong"]])
    print("Total number of queries in training set: " + str(len(spider["train"])))
    print("Total number of queries in validation set: " + str(len(spider["validation"])))

    model_query = torch.load(save_path).to(device)
    start_time = time.time()
    nr_syntax_errors_after, mean_reward, queries_evaluated = evaluate_model(model_query, 
                                                                            tokenizer, 
                                                                            spider, 
                                                                            start_idx = 0, end_idx = 1034, #
                                                                            save_after_eval = True, 
                                                                            use_train = False, 
                                                                            num_beams = 5,
                                                                            retokenize = False, 
                                                                            debug = True, 
                                                                            filename = config["model_checkpoint"].replace("/", "_"),
                                                                            simple_sql_fn = simple_sql_fn, 
                                                                            dbs_full_schema = database_object.dbs_full_schema, 
                                                                            use_best_first_search = config["use_best_first_search"], 
                                                                            check_exec_result = config["check_exec_result"],
                                                                            check_partial_sql = config["check_partial_sql"],
                                                                            check_example = config["check_example"],
                                                                            enum_part_quer_check = config["enum_part_quer_check"], 
                                                                            seq2seq = config["seq2seq"], 
                                                                            device = device)
    print("Total time taken to evaluate model: " + str(time.time() - start_time))