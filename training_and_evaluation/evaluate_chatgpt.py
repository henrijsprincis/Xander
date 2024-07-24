from eval import process_sql
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from preprocess_machine_learning.query_preprocessor import preprocess_query
from preprocess_sql.database_class import DatabaseClass
from open_ai.chat_gpt import ask_chatGPT


with open("config.json") as f:
    config = json.load(f)


def main():
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"], model_max_length = config["max_input_length"])
    database_object = DatabaseClass(config["database_path"])
    spider = load_dataset("spider")
    spider["train"] = spider["train"].select([i for i in range(0, 10)])
    spider["validation"] = spider["validation"].select([i for i in range(0, 10)])
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
    for i in range(len(spider["validation"])):
        chat_gpt_prompt = tokenizer.decode(spider["validation"][i]["partial_input"])
        chat_gpt_prompt = chat_gpt_prompt.replace("<s>", "").replace("</s>", "").replace("<|user|>", "").replace("<|assistant|>", "").replace("<|end|>", "")
        chat_gpt_prompt += "\nOnly output the SQL query without an explanation and without the triple quotes."
        chat_gpt_prompts.append(chat_gpt_prompt)
    for i, prompt in enumerate(chat_gpt_prompts):
        chat_gpt_output = ask_chatGPT(prompt).replace("\n", " ")# Remove newlines
        chat_gpt_outputs.append(chat_gpt_output)
        print(i)
        if i%10 == 0:
            with open(f"chat_gpt_outputs{i}.txt", "w") as f:
                for chat_gpt_output in chat_gpt_outputs:
                    f.write(f"{chat_gpt_output}\n")

    with open(f"chat_gpt_outputs.txt", "w") as f:
        for chat_gpt_output in chat_gpt_outputs:
            f.write(f"{chat_gpt_output}\n")



if __name__ == "__main__":
    main()