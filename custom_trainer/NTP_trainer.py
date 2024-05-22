from eval import process_sql
import json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, PhiForCausalLM
from preprocess_machine_learning.helper_functions import get_save_paths
from preprocess_machine_learning.preprocess_NTP import preprocess_data_query_NTP
from preprocess_sql.database_class import DatabaseClass

with open("config.json") as f:
    config = json.load(f)

#import torch_directml#torch_directml.device()#"cuda" if torch.cuda.is_available() else "cpu"
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    simple_sql_fn = process_sql.SimpleSQL_to_SQL if config["use_simple_sql"] else None
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"], model_max_length = config["max_input_length"])
    checkpoint_path, save_path = get_save_paths(config)
    database_object = DatabaseClass(config["database_path"])
    spider = load_dataset("spider")#spider["train"].select(list(range(0,20)))
    spider = spider.map(
        preprocess_data_query_NTP,
        fn_kwargs={
            "config":config,
            "tokenizer":tokenizer,
            "database_object":database_object,
        }, batched=False)

    print("Total number of queries in training set: " + str(len(spider["train"])))
    print("Total number of queries in validation set: " + str(len(spider["validation"])))

    model_query = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)
    #optimizer = torch.optim.Adam(model_query.parameters(), lr=config["lr"])
    optimizer = torch.optim.RMSprop(model_query.parameters(), lr=config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=1)#gamma=0.99999

    args = TrainingArguments(
        "./checkpoints",
        evaluation_strategy="steps",
        eval_steps=7000,
        logging_strategy="steps",
        logging_steps=1,
        save_steps=7000,
        learning_rate=config["lr"],
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        weight_decay=0.01,
        save_strategy="no",
        save_total_limit=0,
        num_train_epochs=10,
        fp16=config["half_precision"],
        load_best_model_at_end=False,
        report_to="tensorboard")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
    trainer = Trainer(
        model_query,
        args,
        train_dataset=spider["train"],#.select(list(range(0,3500)))
        eval_dataset=spider["validation"],
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
        tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(save_path)

if __name__ == "__main__":
    main()