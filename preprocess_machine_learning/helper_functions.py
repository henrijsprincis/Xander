def reset_params(model):
    print("Resetting weights")
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_and_time(spider, model_query, tokenizer, trainer, schemas, optimizer, max_input_length, max_output_sequence_length, pad_token, save_name, seq2seq = True, save_after_training = True):
    if seq2seq:
        s_train_smol = spider["train"].select(list(range(0,3)))
        tensor = torch.tensor(s_train_smol["input_ids"], dtype=torch.int).to(device)
        preds_before = model_query.generate(tensor, max_length=max_output_sequence_length, num_beams = 1, do_sample = False)
    else:
        quer = [token for token in spider["train"][0]["partial_input"] if token != pad_token]
        tensor = torch.tensor(quer, dtype=torch.int).reshape(1,-1).to(device)
        preds_before = model_query.generate(tensor, max_length=max_input_length+max_output_sequence_length, num_beams = 1, do_sample = False, pad_token_id = tokenizer.pad_token_id)
    start_train_time = time.time()
    trainer.train()
    end_train_time = time.time()
    print("Time taken to train: " + str(end_train_time - start_train_time))
    preds_after = model_query.generate(tensor, max_length=max_output_sequence_length)
    for i in range(2):
        print("Question: ", s_train_smol["question"][i])
        print("Schema: ", str(schemas[s_train_smol["db_id"][i]]))
        print("Correct SQL: ", tokenizer.decode(s_train_smol["labels"][i], skip_special_tokens=True))
        print("Prediction Before FineTuning: ", tokenizer.decode(preds_before[i], skip_special_tokens=True))
        print("Prediction After FineTuning: ", tokenizer.decode(preds_after[i]))
        print("")  
    if save_after_training:
        trainer.save_model("checkpoints/GoodCheckpoint/"+save_name)
        with open("checkpoints/GoodCheckpoint/"+save_name+"trainHist.pkl", "wb") as fp:   #Pickling
            pickle.dump(trainer.state.log_history, fp)
        
        write_me = ""
        for item in trainer.state.log_history:
            if "loss" in item.keys():
                write_me+=str(item["loss"])+"\n"
        
        write_me += "Time taken to train: " + str(end_train_time - start_train_time)+"\n"
        with open("checkpoints/GoodCheckpoint/"+save_name+"trainHist.csv", "w") as fp:   #Pickling
            fp.write(write_me)


def get_save_name(model_checkpoint, add_execution_result, use_simple_sql):
    return model_checkpoint[-7:]+"Examples"+str(add_execution_result)+"SimpleSQL"+str(use_simple_sql)+"NoWeights"*1
