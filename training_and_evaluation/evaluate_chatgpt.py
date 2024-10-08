from eval import process_sql
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from preprocess_machine_learning.query_preprocessor import preprocess_query
from preprocess_sql.database_class import DatabaseClass
from open_ai.chat_gpt import ask_chatGPT
import time

with open("config.json") as f:
    config = json.load(f)


def main():
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"], model_max_length = config["max_input_length"])
    database_object = DatabaseClass(config["database_path"])
    spider = load_dataset("spider")
    spider["train"] = spider["train"].select([i for i in range(0, 10)])
    spider["validation"] = spider["validation"]#.select([i for i in range(0, 10)])
    spider = spider.map(
        preprocess_query,
        fn_kwargs={
            "config":config,
            "tokenizer":tokenizer,
            "database_object":database_object,
        }, batched=False)
    # TEMP Comment spider["validation"] = spider["validation"].select([i for i in range(0, len(spider["validation"])) if not spider["validation"][i]["tooLong"]])
    chat_gpt_prompts = []
    chat_gpt_outputs = []
    print("length of spider[validation]:", len(spider["validation"]))
    for i in range(len(spider["validation"])):
        chat_gpt_prompt = tokenizer.decode(spider["validation"][i]["partial_input"])
        chat_gpt_prompt = chat_gpt_prompt.replace("<s>", "").replace("</s>", "").replace("<|user|>", "").replace("<|assistant|>", "").replace("<|end|>", "")
        chat_gpt_prompt += "\nOnly output the SQL query without an explanation and without the triple quotes."
        chat_gpt_prompts.append(chat_gpt_prompt)
    start_time = time.time()
    for i, prompt in enumerate(chat_gpt_prompts):
        chat_gpt_output = ask_chatGPT(prompt, "gpt-3.5-turbo").replace("\n", " ")# Remove newlines. Model choices "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
        chat_gpt_outputs.append(chat_gpt_output)
        print(i)
        if i%10 == 0:
            with open(f"chat_gpt_outputs{i}.txt", "w") as f:
                for chat_gpt_output in chat_gpt_outputs:
                    f.write(f"{chat_gpt_output}\n")
    end_time = time.time()
    print("Time taken:", end_time - start_time)
    with open(f"chat_gpt_outputs.txt", "w") as f:
        for chat_gpt_output in chat_gpt_outputs:
            f.write(f"{chat_gpt_output}\n")
    print("Finished")

if __name__ == "__main__":
    main()