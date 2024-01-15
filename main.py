from eval import process_sql
import os
import copy
from datasets import load_dataset
import pickle
import time
from sklearn.metrics import confusion_matrix
#ML STUFF
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer, T5Config, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments
from customTrainer.Seq2SeqTrainer import Seq2SeqTrainer
import torch.distributions.categorical as cate
## Test stuff
from preprocess.parse_raw_json import get_schemas_from_json
from eval.getNumberCorrect import str_to_db_path, execute_query, evaluate_model
from open_ai.chat_gpt import ask_chatGPT as ask_chatGPT

def token_definitions(model_checkpoint):
    #T5: 0 padding, 1 BOS, 2EOS
    #BART: 0 BOS, 1 PAD, 2EOS
    #CasualLM: 50256 ALL
    if model_checkpoint == "facebook/bart-base":#bart
        bos_token = 0
        pad_token = 1
        eos_token = 2
    elif model_checkpoint == "Salesforce/codegen-350M-multi":
        bos_token = 50256#we never check for beginning of sequence token ;) so all is well
        eos_token = 50256
        pad_token = eos_token
    else:#T5
        bos_token = 1
        pad_token = 0
        eos_token = 2
    mask_token = 3
    return bos_token, pad_token, eos_token, mask_token

def update_tokenizer(tokenizer, model_checkpoint):
    global bos_token, pad_token, eos_token, mask_token
    bos_token, pad_token, eos_token, mask_token = token_definitions(model_checkpoint=model_checkpoint)
    if nextTokenPred:
        tokenizer.bos_token = tokenizer.decode([bos_token])
        tokenizer.bos_token_id = bos_token
        tokenizer.pad_token = tokenizer.decode([pad_token])
        tokenizer.pad_token_id = pad_token
        tokenizer.eos_token = tokenizer.decode([eos_token])
        tokenizer.eos_token_id = eos_token
        tokenizer.mask_token="[MASK]"
        tokenizer.mask_token_id = mask_token
    return tokenizer

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

def preprocess_data_query_seq2seq(examples, **kwargs):
  add_execution_result = kwargs['add_execution_result']
  use_simpleSQL = kwargs['use_simple_sql']
  invalid=False
  simpleSQL="NOTHING"
  if use_simpleSQL:
      try:
        db_id = str_to_db_path(examples["db_id"])
        schema = process_sql.Schema(process_sql.get_schema(db_id))
        query_ast = process_sql.get_sql(schema, examples["query"])
        simpleSQL = process_sql.sql_tree_to_SimpleSQL(query_ast)#process_sql.SimpleSQL_to_SQL(simpleSQL)
        #check if they execute to the same thing. if not, then mark simpleSQL as invalid
        e_gold = execute_query(examples["db_id"], examples["query"])
        e_conv = execute_query(examples["db_id"], process_sql.SimpleSQL_to_SQL(simpleSQL))
        excpt = examples["query"] == 'SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )'#very annoying
        if e_gold != e_conv or e_gold[0]==-1 or excpt:
            #print("db id: ",examples["db_id"], "examples:", examples["query"])
            invalid = True
      except:
        print("Except")
        invalid = True#do not use if run into exception
      
  # add column types to schema
  s = copy.deepcopy(schemas[examples["db_id"]])#deep copy schema.
  full_schema = dbs_full_schema[examples["db_id"]] # full schema
  for t_name in s.keys():
    for c_idx, c_name in enumerate(s[t_name]):
        #get column type.
        li = [t_name, c_name]
        idx = full_schema["column_names_original"].index(li)
        column_type = full_schema["column_types"][idx]
        s[t_name][c_idx] = c_name+" "+column_type

  schema_str = str(s).replace("'"," ").replace(","," ").replace(":", " ")
  schema_str = " ".join(schema_str.split())#beautiful
  # Execute query
  model_input_string = examples["question"]+"\n"+schema_str+"\n"#add a newline here so everything is consistent
  exec_result = execute_query(examples["db_id"], examples["query"])
  if exec_result[1]==[]:
      first_tuple = []
  else:
      first_tuple = exec_result[1][0]

  if add_execution_result:
    model_input_string += str(first_tuple[:50]) # 50 characters of execution result (fairly sure [:50] should be outside of string)
  model_inputs = tokenizer(model_input_string, max_length=max_input_length, truncation=True, padding='max_length')
  model_inputs["execution_result"] = str(exec_result)
  # Setup the tokenizer for targets
  # Execution Result. Hope this works
  #with tokenizer.as_target_tokenizer():
  if use_simpleSQL and invalid==False:
    labels = tokenizer(simpleSQL, max_length=max_output_sequence_length, truncation=True, padding='max_length')
  else:    
    labels = tokenizer(examples["query"], max_length=max_output_sequence_length, truncation=True, padding='max_length')
  model_inputs["labels"]  = labels["input_ids"]
  model_inputs["schemas"] = str(schemas[examples["db_id"]])
  model_inputs["tooLong"] = False
  model_inputs["simpleSQL"] = simpleSQL
  if pad_token not in model_inputs["input_ids"] or invalid:
    model_inputs["tooLong"] = True
  return model_inputs

def preprocess_data_query_NTP(examples, **kwargs):
    #in casual LM, 
    add_execution_result = kwargs['add_execution_result']
    use_simpleSQL = kwargs['use_simple_sql']
    invalid=False
    simpleSQL="NOTHING"
    if use_simpleSQL:
        try:
            db_id = str_to_db_path(examples["db_id"])
            schema = process_sql.Schema(process_sql.get_schema(db_id))
            query_ast = process_sql.get_sql(schema, examples["query"])
            simpleSQL = process_sql.sql_tree_to_SimpleSQL(query_ast)#process_sql.SimpleSQL_to_SQL(simpleSQL)
            #check if they execute to the same thing. if not, then mark simpleSQL as invalid
            e_gold = execute_query(examples["db_id"], examples["query"])
            e_conv = execute_query(examples["db_id"], process_sql.SimpleSQL_to_SQL(simpleSQL))
            excpt = examples["query"] == 'SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )'#very annoying
            if e_gold != e_conv or e_gold[0]==-1 or excpt:
                #print("db id: ",examples["db_id"], "examples:", examples["query"])
                invalid = True
        except:
            invalid = True#do not use if run into exception
    
    # add column types to schema
    s = copy.deepcopy(schemas[examples["db_id"]])#deep copy schema.
    full_schema = dbs_full_schema[examples["db_id"]] # full schema
    for t_name in s.keys():
        for c_idx, c_name in enumerate(s[t_name]):
            #get column type.
            li = [t_name, c_name]
            idx = full_schema["column_names_original"].index(li)
            column_type = full_schema["column_types"][idx]
            s[t_name][c_idx] = c_name+" "+column_type
    
    schema_str = str(s).replace("'"," ").replace(","," ").replace(":", " ")
    schema_str = " ".join(schema_str.split())#beautiful
    # Execute query
    model_input_string = examples["question"]+"\n"+schema_str+"\n"#add a newline here so everything is consistent
    exec_result = execute_query(examples["db_id"], examples["query"])
    if exec_result[1]==[]:
        first_tuple = []
    else:
        first_tuple = exec_result[1][0]
    
    if add_execution_result:
        model_input_string += str(first_tuple[:50]) # 50 characters of execution result (fairly sure [:50] should be outside of string)
    
    if use_simpleSQL and invalid==False:
        model_inputs = tokenizer(model_input_string + "\n" + simpleSQL, max_length=max_input_length, truncation=True, padding='max_length')
    else:
        model_inputs = tokenizer(model_input_string + "\n" + examples["query"], max_length=max_input_length, truncation=True, padding='max_length')
        
    partial_input = tokenizer(model_input_string + "\n", max_length=max_input_length, truncation=True, padding='max_length')#max_output_sequence_length
    model_inputs["labels"]  = model_inputs["input_ids"]
    model_inputs["schemas"] = str(schemas[examples["db_id"]])
    model_inputs["tooLong"] = False
    model_inputs["simpleSQL"] = simpleSQL
    model_inputs["execution_result"] = str(exec_result)
    model_inputs["partial_input"] = partial_input["input_ids"]
    if pad_token not in model_inputs["input_ids"] or invalid:
        model_inputs["tooLong"] = True
    
    return model_inputs

def remove_too_long(spider):
    len_train = len(spider["train"])
    valid_idxs = set(range(len_train))
    for i in range(len_train):
        if spider["train"][i]["tooLong"]:
            valid_idxs.remove(i)
    spider["train"] = spider["train"].select(list(valid_idxs))
    len_validation = len(spider["validation"])
    valid_idxs = set(range(len_validation))
    for i in range(len_validation):
        if spider["validation"][i]["tooLong"]:
            valid_idxs.remove(i)
    #print(valid_idxs)
    spider["validation"] = spider["validation"].select(list(valid_idxs))
    return spider

def preprocess_data_language(examples):
  # Input
  model_inputs = tokenizer(str(schemas[examples["db_id"]]), max_length=max_input_length, truncation=True, padding='max_length')#"Write an SQL query. "+
  # Setup the tokenizer for targets
  #with tokenizer.as_target_tokenizer():
  labels = tokenizer(examples["question"], max_length=max_output_sequence_length, truncation=True, padding='max_length')
  model_inputs["labels"] = labels["input_ids"]
  model_inputs["schemas"] = str(schemas[examples["db_id"]])
  return model_inputs

def labels_to_onehot(lbls, sz = [200,32100]):# 200, 32100 logits shape
    cuda0 = torch.device('cuda:0')
    output = torch.zeros([sz[0],sz[1]], dtype=torch.float64, device=cuda0)#float64
    for l in range(sz[0]):
        output[l,lbls[0,l]] = 1
    return output

def RLloss_update(model, input_encoder, input_decoder, predicted_logit, reward):
    global optimizer
    output_logits = model(input_ids=input_encoder, decoder_input_ids=input_decoder).logits
    sft_mx = torch.nn.Softmax(dim=0)
    output_logits_softmax = sft_mx(output_logits.view(-1))
    output_logits_gold = torch.clone(output_logits_softmax)
    output_logits_gold[predicted_logit] *= 2
    loss_fn = nn.L1Loss(reduction = "sum")
    loss = loss_fn(output_logits_softmax, output_logits_gold)*reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

evaluate_generator_model         = True # Coming soon
#PRETRAINED
use_good_checkpoint_query        = True
use_good_checkpoint_verifier     = False # Coming soon
#TRAIN
train_model_verifier             = False # Coming soon
#SETTINGS
add_execution_result             = False # True
save_after_training              = True
#use simple sql
use_simple_sql                   = True
#check partial sql
check_exec_result                = False#whether to allow multiple execution results  (aka multAttempts) (Dis is veri misleading variable name)#True
check_partial_sql                = False#False
check_example                    = False#only true if check_partial_sql is also true
enum_part_quer_check             = False#enumerative partial Query checker
model_checkpoint                 = "Salesforce/codet5-small"#"facebook/bart-base"#"Salesforce/codet5-small"#"facebook/bart-base#"Salesforce/codegen-350M-multi"
reset_weights                    = False
save_name                        = model_checkpoint[-7:]+"Examples"+str(add_execution_result)+"SimpleSQL"+str(use_simple_sql)+"NoWeights"*1 # Naming scheme: last6letters of model name. + Examples? + simpleSQL
use_best_first_search            = True#wether to use beamsearch or bestfirstsearch when doing evaluation
#SEQ2SEQ vs LM
seq2seq                          = True #when true use seq2seq (change this)
nextTokenPred                    = not seq2seq #do not change me
simple_sql_fn                    = None
#CHATGPT
query_chat_gpt                   = False
#settings Hyperparams
max_input_length                 = 512
max_output_sequence_length       = 200
half_precision                   = False
device = "cuda" if torch.cuda.is_available() else "cpu"
simple_sql_fn = process_sql.SimpleSQL_to_SQL if use_simple_sql else None
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = max_input_length)#model_checkpoint
tokenizer = update_tokenizer(tokenizer, model_checkpoint)
a,b = model_checkpoint, model_checkpoint
if train_model_verifier:
    from transformersCoderl.src.transformers.models.t5 import T5ForConditionalGeneration
    model_verifier = T5ForConditionalGeneration.from_pretrained(model_checkpoint, tuning_mode="critic", clone_rl_head=False).to(device)
if use_good_checkpoint_query:
    a="./checkpoints/GoodCheckpoint/"+save_name
if use_good_checkpoint_verifier:
    b="./checkpoints/GoodCheckpoint/"+save_name+"Verifier"
    model_verifier.load_state_dict(torch.load(b))

if seq2seq:
    model_query     = AutoModelForSeq2SeqLM.from_pretrained(a).to(device)
else:
    model_query     = AutoModelForCausalLM.from_pretrained(a).to(device)

if reset_weights:
    reset_params(model_query)

lr=4e-05
optimizer = torch.optim.Adam(model_query.parameters(), lr=lr)
schemas, db_names, tables, dbs_full_schema = get_schemas_from_json("./database/tables.json")
for key in dbs_full_schema.keys():
    db_full = dbs_full_schema[key]
    for idx, column_name in enumerate(db_full['column_names_original']):
        #the first element of column name is the table
        table_idx = column_name[0]
        if table_idx == -1:#skip first token
            continue
        table_name = db_full["table_names_original"][table_idx]
        db_full['column_names_original'][idx][0] = table_name.lower()
        db_full['column_names_original'][idx][1] = db_full['column_names_original'][idx][1].lower()

if __name__ == "__main__":
    spider = load_dataset("spider")
    if seq2seq:
        spider = spider.map(preprocess_data_query_seq2seq, fn_kwargs={"add_execution_result":add_execution_result, 'use_simple_sql':use_simple_sql}, batched=False)
    else:
        spider = spider.map(preprocess_data_query_NTP, fn_kwargs={"add_execution_result":add_execution_result, 'use_simple_sql':use_simple_sql}, batched=False)
    print("Total number of queries in training set: " + str(len(spider["train"])))
    print("Total number of queries in validation set: " + str(len(spider["validation"])))
    spider = remove_too_long(spider)
    print("Total number of queries remaining in training set: " + str(len(spider["train"])))
    print("Total number of queries remaining in validation set: " + str(len(spider["validation"])))
    lr = 4e-5#4e-3
    if seq2seq:#training args
        args = Seq2SeqTrainingArguments(
            "./checkpoints",#model_dir
            evaluation_strategy="steps",
            eval_steps=7000,
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="no",
            save_steps=7000,
            learning_rate=lr,#4e-5
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            save_total_limit=0,
            num_train_epochs=1,#50
            predict_with_generate=False,
            fp16=False,
            load_best_model_at_end=False,
            report_to="tensorboard")
        data_collator = DataCollatorForSeq2Seq(tokenizer)
        optimizer = torch.optim.Adam(model_query.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=1)#gamma=0.99999
        trainer = Seq2SeqTrainer(
            model_query,
            args,
            train_dataset=spider["train"],#.select(list(range(0,3500)))
            eval_dataset=spider["validation"],
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
            tokenizer=tokenizer)
    else:
        args = TrainingArguments(
            "./checkpoints",#model_dir
            evaluation_strategy="steps",
            eval_steps=7000,
            logging_strategy="steps",
            logging_steps=10,
            save_steps=7000,
            learning_rate=lr,#4e-5
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
            weight_decay=0.01,
            save_strategy="no",
            save_total_limit=0,
            num_train_epochs=50,#50#predict_with_generate=False,
            fp16=half_precision,
            load_best_model_at_end=False,
            report_to="tensorboard")
        data_collator = DataCollatorForLanguageModeling(tokenizer)
        optimizer = torch.optim.Adam(model_query.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=1)#gamma=0.99999
        trainer = Trainer(
            model_query,
            args,
            train_dataset=spider["train"],#.select(list(range(0,3500)))
            eval_dataset=spider["validation"],
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
            tokenizer=tokenizer)
    ### Code added for demo here! ###
    while True:
        print("Welcome to a demo of the SQL synthesizer!")
        print('Type "1" to train an LLM. \nType "2" to evaluate a trained model without syntax checking.\nType "3" to evaluate a trained model with syntax checking.\nType "4" to exit.\n')
        user_input = input(": ")
        if user_input == "1":
            #train
            train_and_time(spider, model_query, tokenizer, trainer, schemas, optimizer, max_input_length, max_output_sequence_length, pad_token, save_name, seq2seq = seq2seq, save_after_training = save_after_training)
        elif user_input == "2":
            #evaluate
            nr_syntax_errors_after, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, start_idx = 0, end_idx = 1, #
                                                                    save_after_eval = False, use_train = False, num_beams = 5, #let's do best first search
                                                                    retokenize = False, debug = False, filename="TEMP",#FullCodeT5AllSQL
                                                                    simple_sql_fn = simple_sql_fn, dbs_full_schema = dbs_full_schema, use_best_first_search = use_best_first_search, 
                                                                    check_exec_result=check_exec_result, check_partial_sql = check_partial_sql, check_example = check_example,
                                                                    enum_part_quer_check = enum_part_quer_check, seq2seq = seq2seq, demo_mode = True)
        elif user_input == "3":
            check_partial_sql = not check_partial_sql
            status = "on" if check_partial_sql else "off"
            print(f"Syntax checking is now {status}\n")
        elif user_input == "4":
            break
        else:
            print("Did not understand command!")
    exit()

##... eval coming soon ...##
if False:
    nr_syntax_errors_after, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, start_idx = 0, end_idx = 1, #
                                                                            save_after_eval = True, use_train = True, num_beams = 5, #let's do best first search
                                                                            retokenize = False, debug = False, filename="TEMP",#FullCodeT5AllSQL
                                                                            simple_sql_fn = simple_sql_fn, dbs_full_schema = dbs_full_schema, use_best_first_search = use_best_first_search, 
                                                                            check_exec_result=check_exec_result, check_partial_sql = check_partial_sql, check_example = check_example,
                                                                            enum_part_quer_check = enum_part_quer_check, seq2seq = seq2seq)

    #breakpoint()

    if evaluate_generator_model:
        start_validate_time = time.time()
        nr_syntax_errors_after, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, start_idx = 0, end_idx = len(spider["validation"]), #
                                                                                save_after_eval = True, use_train = False, num_beams = 5, #let's do best first search
                                                                                retokenize = False, debug = False, filename=save_name+"MultAttempts"+str(check_exec_result)+"BFS"+str(use_best_first_search)+"SymbolicChecker"+str(check_partial_sql)+"QueryRepairer"+str(enum_part_quer_check),#FullCodeT5AllSQL
                                                                                simple_sql_fn = simple_sql_fn, dbs_full_schema = dbs_full_schema, use_best_first_search = use_best_first_search, 
                                                                                check_exec_result=check_exec_result, check_partial_sql = check_partial_sql, check_example = check_example,
                                                                                enum_part_quer_check = enum_part_quer_check, seq2seq = seq2seq)
        end_validate_time = time.time()
        print("Time taken to validate: " + str(end_validate_time - start_validate_time))
        print("Time taken to train: " + str(end_train_time - start_train_time))
        
    #process_sql.validate_partial_SQL(schemas[db_id], query)


    ### QUERY CHAT GPT #################################################################################################################################################################
    if query_chat_gpt:
        stuff = 0
        
        from additionalCode.chatgptv2 import ask_chatGPT as ask_chatGPT
        start_time = time.time()
        for overall in range(1):
            print("overall", overall)
            questions = ""
            chunks = 5
            for i in range(chunks):
                query=spider["validation"][chunks*overall+i]
                query_text = tokenizer.decode(query["input_ids"],skip_special_tokens=True)
                query_split = query_text.split("\n")
                query_split[0] = str(i+1)+". Write an SQLite query that answers the following question: " + query_split[0]
                query_split[1] = "Given the schema: " + str(query["schemas"])#query_split[1]
                query_split[2] = "That returns the following tuple when executed on a database: " + query_split[2]#query["execution_result"][4:-1]#query_split[2][4:-1]
                query_text = "\n".join(query_split)
                query_text+= "\nDo not provide an explanation. Only provide the SQLite code snippets."
                questions += query_text + "\n\n"
            questions+="Number the snippets 1 to "+str(chunks)
            print(questions)
            time.sleep(6)
            response = ask_chatGPT(questions)    
            with open('./chatGptResponses/bigboi'+str(overall)+'.txt','w') as file:
                file.write(response+"\n")
            time.sleep(6)
        end_time = time.time()
        print("Time taken: ", end_time-start_time)
        breakpoint()
    ### END QUERY CHAT GPT ###

    ## Train critic on the trained language model outputs ###############################################################################################################################
    #model_verifier.load_state_dict(torch.load("./checkpoints/GoodCheckpoint/verifierExamples2"))
    optimizer_v    = torch.optim.Adam(model_verifier.parameters(), lr=4e-5)
    map_to = [93, 73, 86, 87]#yes, executes, runtimeerror, syntaxError
    true_vals      = []
    estimated_vals = []
    sft_mx         = torch.nn.Softmax(dim=0)
    loss_fn        = nn.CrossEntropyLoss()#nn.L1Loss(reduction = "sum")#torch.nn.functional.cross_entropy#nn.CrossEntropyLoss()
    lr_scheduler_v = torch.optim.lr_scheduler.ExponentialLR(optimizer_v,gamma=1)#gamma=0.99999


    def input_ids_to_output(input_ids):
        text = tokenizer.decode(input_ids,skip_special_tokens=True)
        splt = text.split("\n")
        query = splt[-1]
        rest = "\n".join(splt[:-1])
        input_ids = torch.tensor(tokenizer(rest, max_length=max_input_length, truncation=True, padding='max_length')["input_ids"],dtype=torch.int64,device=device).reshape(1,-1)
        decoder_ids = torch.tensor(tokenizer(query, max_length=max_output_sequence_length, truncation=True, padding='max_length')["input_ids"],dtype=torch.int64,device=device).reshape(1,-1)
        return input_ids, decoder_ids

    def train_verifier(model_verifier, model_query, optimizer_v, add_exec_output=True, nr_iters = 1, evaluation = False, items_in_subset = 1, use_train = True, num_beams = 4):
        true_vals,estimated_vals,symbolic_vals=[],[],[]
        if evaluation:
            model_verifier = model_verifier.eval()
        else:
            model_verifier = model_verifier.train()
        sz = len(spider["validation"])
        if use_train:
            sz = len(spider["train"])
        for i in range(nr_iters):#Second model with examples
            for subset in range(int(sz//items_in_subset)):#int(sz//items_in_subset)
                #use beamsearch to sample queries
                params = [items_in_subset * subset, (subset+1) * items_in_subset, save_name+"MultAttempts"+str(check_exec_result)+"BFS"+str(use_best_first_search)+"SymbolicChecker", num_beams, use_train]
                params = [str(param) for param in params]
                filename = "cache/"+"_".join(params)+".p"
                #savestuff
                if os.path.exists(filename):
                    with open( filename, "rb" ) as file:
                        print("using saved")
                        [nr_syntax_errors_after, mean_reward, queries_evaluated] = pickle.load(file)
                else:
                    print("saved not found. Calculating")
                    nr_syntax_errors_after, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, start_idx = items_in_subset * subset , end_idx = (subset+1) * items_in_subset, #len(spider["train"])
                                                                                save_after_eval = False, use_train = use_train, num_beams = num_beams, #let's do best first search
                                                                                retokenize = True, debug = False, filename=save_name+"MultAttempts"+str(check_exec_result)+"BFS"+str(use_best_first_search)+"SymbolicChecker"+str(check_partial_sql),#FullCodeT5AllSQL
                                                                                simple_sql_fn = simple_sql_fn, dbs_full_schema = dbs_full_schema, use_best_first_search = False, 
                                                                                check_exec_result = False, check_partial_sql = False, check_example = False, break_early = False)
                    with open( filename, "wb" ) as file:
                        pickle.dump([nr_syntax_errors_after, mean_reward, queries_evaluated], file)
                all_input_ids, all_decoder_ids, all_g = [],[],[]
                for q in queries_evaluated:#<--- one batch.
                    input_ids, decoder_ids = torch.cuda.LongTensor(q["input_ids"]).reshape(1,512), torch.cuda.LongTensor(q["predicted_input_ids"]).reshape(1,200)
                    all_input_ids.append(input_ids)
                    all_decoder_ids.append(decoder_ids)
                    all_g.append(torch.tensor(map_to.index(q["labels"][1]), dtype=torch.int64, device=device).reshape(1,-1))
                    if evaluation:
                        pred_quer = tokenizer.decode(decoder_ids.squeeze(), skip_special_tokens = True)
                        ex_res = eval(q["execution_result"])[-1]
                        if len(ex_res) > 0:
                            ex_res = ex_res[0]
                        else:
                            ex_res = ()
                        symbolic_vals.append(process_sql.validate_partial_SQL(eval(q["schemas"]), pred_quer, example = ex_res, db_full = dbs_full_schema[q["db_id"]], return_error_type = True))
                        if symbolic_vals[-1] == 3 and all_g[-1].item() == 0:
                            db_full = dbs_full_schema[q["db_id"]]
                            print(pred_quer)
                            print(ex_res)
                            print("columns: ", db_full["column_names_original"])
                            breakpoint()
                error_pred_loss, error_preds = model_verifier(input_ids = torch.cat(all_input_ids,dim=0), error_types = torch.cat(all_g,dim=0), labels = torch.cat(all_decoder_ids,dim=0))
                error_pred_loss*=items_in_subset
                if not evaluation:
                    optimizer_v.zero_grad()
                    error_pred_loss.backward()#retain_graph=True
                    optimizer_v.step()
                with torch.no_grad():
                    nr_correct = 0
                    for pred, true in zip(error_preds, all_g):
                        pred = pred.item()
                        true = true.item()
                        if pred == true:
                            nr_correct+=1
                        if evaluation:
                            estimated_vals.append(pred)
                            true_vals.append(true)
                print("i: ", i)
                print("correct: ", nr_correct/len(all_g))
                print("loss: ", error_pred_loss)
        if evaluation:
            return estimated_vals, true_vals, symbolic_vals
        return model_verifier, optimizer_v

    start_verifier_train_time = time.time()
    if train_model_verifier:
        model_verifier, optimizer_v = train_verifier(model_verifier, model_query, optimizer_v, nr_iters=20, add_exec_output = True, items_in_subset=5)# we only want the full stuff.
        torch.save(model_verifier.state_dict(), "./checkpoints/GoodCheckpoint/"+save_name+"Verifier")
    end_verifier_train_time = time.time()
    estimated_vals, true_vals, symbolic_vals = train_verifier(model_verifier, model_query, optimizer_v, nr_iters=1, add_exec_output = True, evaluation=True, items_in_subset=1, use_train=False)
    end_verifier_validate_time = time.time()
    print("confusion matrix (true vs estimated)")
    c_mat = confusion_matrix(true_vals,estimated_vals)
    print(c_mat)
    print("confusion matrix (true vs symbolic)")
    c_mat = confusion_matrix(true_vals,symbolic_vals)
    print(c_mat)

    #print("Time taken to train verifier: ", str(end_verifier_train_time-start_verifier_train_time))
    #print("Time taken to validate verifier: ", str(end_verifier_validate_time-end_verifier_train_time))
    #breakpoint()
    ## Classical RL ####################################################################################################################################################################
    nr_syntax_errors_before, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, end_idx = len(spider["validation"]), save_after_eval = True, use_train = False, num_beams = 1, retokenize = False, debug=False, filename="PredictionsBefore")
    print("Nr Syntax Errors BEFORE Training: ", nr_syntax_errors_before)
    syntax_throughout_time = []
    ##RL
    number_of_rl_steps = 20#
    for iters in range(number_of_rl_steps):
        print("Iters: ", iters)
        #model_verifier = model_verifier.eval()
        for subset in range(6,7):#700#12
            #print("Doing Subset: ", subset)
            nr_syntax_errors, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, start_idx = subset*1, end_idx = (subset+1)*1, save_after_eval = False, 
                                                                            use_train = True, num_beams = 5, retokenize = True, add_exec_output = True,
                                                                            debug = False, filename="PredictionVanillaTry"+str(iters), break_early = False, subtract_mean_batch = False)
            rewards = 0
            for q in queries_evaluated:#go through the queries.
                inp     = torch.cuda.LongTensor(q["input_ids"]).reshape(1,-1)#Int
                inp_att = torch.cuda.BoolTensor(q["attention_mask"]).reshape(1,-1)
                #dec_att = torch.zeros((1,len(hid[0]))).type(torch.BoolTensor).to(device)
                hid    = q["pred"][:-1].reshape(1,-1)
                print(tokenizer.decode(q["pred"],skip_special_tokens=True))
                reward = q["correct"]
                ret = model_query(input_ids = inp, attention_mask = inp_att, decoder_input_ids = hid)
                prob_state = 1
                for i in range(len(q["pred"])-1):#for every character (except the last)
                    ##critic block
                    if False:
                        with torch.no_grad():
                            decod_inp = tokenizer.decode(q["input_ids"], skip_special_tokens=True).split("\n")
                            decod_inp[-2] = tokenizer.decode(q["pred"][:i+1], skip_special_tokens=True)#partial query (last idx not inclusive)
                            critic_in = tokenizer("\n".join(decod_inp), max_length=512, truncation=True, padding='max_length')
                            ret_v = model_verifier(input_ids = inp, attention_mask = inp_att, decoder_input_ids = hid)
                            logs = ret_v.get("logits").reshape(-1) # pass in start token and get distribution.
                            logs_yen = torch.nn.functional.softmax(logs[map_to])
                            reward = torch.dot(logs_yen,torch.tensor([1.0,0,-1]).to(device))#pass, compile, fail
                            rewards+=reward
                    ###end critic block
                    #tokenizer.decode(torch.argmax(ret.logits,dim=2).tolist()[0])
                    retlg = ret.get("logits")[0,i,:]#get current step's logits
                    p_distrib = sft_mx(retlg)
                    l_p_d = cate.Categorical(p_distrib)
                    action = q["pred"][i+1]#hid[0,i]#l_p_d.log_prob(action)#torch.cuda.IntTensor([1])
                    prob_transition = p_distrib[action]
                    loss = -l_p_d.log_prob(action) * reward  #-mean_reward* prob_state
                    loss.backward(retain_graph=True)
                    prob_state *= prob_transition
                    if action == 2:
                        break
            print("rewards", rewards)
            print("Syn err", nr_syntax_errors)
            optimizer.step()
            optimizer.zero_grad()
            syntax_throughout_time.append(nr_syntax_errors)
        #model_verifier = model_verifier.train()
        #model_verifier, optimizer_v = train_verifier(model_verifier, model_query, optimizer_v, add_exec_output = True, use_prob = True)

        #nr_syntax_errors_after, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, end_idx = len(spider["validation"]), save_after_eval = True, use_train = False, num_beams = 1, retokenize = False, debug=False, filename="PredictionsWithRLValidation")
        #print("Nr Syntax Errors AFTER Training: ", nr_syntax_errors_after)

    #nr_syntax_errors_after, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, end_idx = len(spider["validation"]), save_after_eval = True, use_train = False, num_beams = 1, retokenize = False, debug=False, filename="PredictionsWithRLValidation")
    #print("Nr Syntax Errors AFTER Training: ", nr_syntax_errors_after)

    if save_after_training:
        trainer.save_model("checkpoints/GoodCheckpoint/RLcorrectValidation")