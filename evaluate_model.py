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
from preprocess_sql.parse_raw_json import get_schemas_from_json
from eval.getNumberCorrect import str_to_db_path, execute_query, evaluate_model
from open_ai.chat_gpt import ask_chatGPT as ask_chatGPT
from prompts.simple_sql_rules import prompt_sql

def token_definitions_old(model_checkpoint):
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

def token_definitions(tokenizer):
    bos_token = tokenizer.bos_token_id
    pad_token = tokenizer.pad_token_id
    eos_token = tokenizer.eos_token_id
    mask_token = tokenizer.mask_token_id
    return bos_token, pad_token, eos_token, mask_token

def update_tokenizer(tokenizer, model_checkpoint):
    global bos_token, pad_token, eos_token, mask_token
    bos_token, pad_token, eos_token, mask_token = token_definitions(tokenizer=tokenizer)
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
  model_input_string = "Provide SQL that answers the following question: "+examples["question"]+"\n"+schema_str+"\n"#add a newline here so everything is consistent
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
    user_token = kwargs['user_token']
    assistant_token = kwargs['assistant_token']
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
    
    schema_str = str(s)#.replace("'"," ").replace(","," ").replace(":", " ")
    # Execute query
    model_input_string = user_token
    model_input_string += "Provide SQL that answers the following question: "+examples["question"]+"\n"
    model_input_string += "The schema: \n"+schema_str+"\n"
    model_input_string += "Start your answer with 'SELECT' and end with a semicolon.\n"
    #model_input_string += prompt_sql

    exec_result = execute_query(examples["db_id"], examples["query"])
    if exec_result[1]==[]:
        first_tuple = []
    else:
        first_tuple = exec_result[1][0]
    
    if add_execution_result:
        model_input_string += "This is the partial execution result of the query: "+str(first_tuple[:50]) # 50 characters of execution result (fairly sure [:50] should be outside of string)
    model_input_string += assistant_token
    if use_simpleSQL and invalid==False:
        model_inputs = tokenizer(model_input_string + simpleSQL, max_length=max_input_length, truncation=True)
    else:
        model_inputs = tokenizer(model_input_string + examples["query"], max_length=max_input_length, truncation=True)
        
    partial_input = tokenizer(model_input_string, max_length=max_input_length, truncation=True)#max_output_sequence_length
    model_inputs["labels"]  = model_inputs["input_ids"]
    model_inputs["schemas"] = str(schemas[examples["db_id"]])
    model_inputs["tooLong"] = False
    model_inputs["simpleSQL"] = simpleSQL
    model_inputs["execution_result"] = str(exec_result)
    model_inputs["partial_input"] = partial_input["input_ids"]
    if False and (pad_token not in model_inputs["input_ids"] or invalid):
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

evaluate_generator_model         = True 
#PRETRAINED
use_good_checkpoint_query        = False
#TRAIN
add_execution_result             = False # True
save_after_training              = True
#use simple sql
use_simple_sql                   = False
#check partial sql
check_exec_result                = True#whether to allow multiple execution results  (aka multAttempts) (Dis is veri misleading variable name)#True
check_partial_sql                = False#False
check_example                    = False#only true if check_partial_sql is also true
enum_part_quer_check             = False#enumerative partial Query checker
model_checkpoint                 = "microsoft/Phi-3-mini-128k-instruct"#"facebook/bart-base"#"Salesforce/codet5-small"#"facebook/bart-base#"Salesforce/codegen-350M-multi"
reset_weights                    = False
save_name                        = model_checkpoint[-7:]+"Examples"+str(add_execution_result)+"SimpleSQL"+str(use_simple_sql)+"NoWeights"*1 # Naming scheme: last6letters of model name. + Examples? + simpleSQL
use_best_first_search            = True#wether to use beamsearch or bestfirstsearch when doing evaluation
#SEQ2SEQ vs LM
seq2seq                          = False #when true use seq2seq (change this)
nextTokenPred                    = not seq2seq #do not change me
simple_sql_fn                    = None
#CHATGPT
query_chat_gpt                   = False
#settings Hyperparams
max_input_length                 = 1024
max_output_sequence_length       = 200
half_precision                   = False
#import torch_directml#torch_directml.device()#"cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
simple_sql_fn = process_sql.SimpleSQL_to_SQL if use_simple_sql else None
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = max_input_length)#model_checkpoint
tokenizer = update_tokenizer(tokenizer, model_checkpoint)
a,b = model_checkpoint, model_checkpoint
if use_good_checkpoint_query:
    a="./checkpoints/GoodCheckpoint/"+save_name

if seq2seq:
    model_query     = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small").to(device)#AutoModelForSeq2SeqLM.from_pretrained(a).to(device)
else:
    model_query     = AutoModelForCausalLM.from_pretrained(a, trust_remote_code=True).to(device)

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
    spider["train"] = spider["train"]#.select(list(range(0,20)))
    spider["validation"] = spider["validation"]
    if seq2seq:
        spider = spider.map(preprocess_data_query_seq2seq, fn_kwargs={"add_execution_result":add_execution_result, 'use_simple_sql':use_simple_sql}, batched=False)
    else:
        spider = spider.map(preprocess_data_query_NTP, fn_kwargs={"add_execution_result":add_execution_result, 
                                                                  'use_simple_sql':use_simple_sql,
                                                                  'user_token':"<|user|>",
                                                                  'assistant_token':"<|end|>\n<|assistant|>"}, batched=False)
    print("Total number of queries in training set: " + str(len(spider["train"])))
    print("Total number of queries in validation set: " + str(len(spider["validation"])))
    spider = remove_too_long(spider)
    print("Total number of queries remaining in training set: " + str(len(spider["train"])))
    print("Total number of queries remaining in validation set: " + str(len(spider["validation"])))
    start_validate_time = time.time()
    nr_syntax_errors_after, mean_reward, queries_evaluated = evaluate_model(model_query, tokenizer, spider, start_idx = 0, end_idx = 1034, #
                                                                            save_after_eval = True, use_train = False, num_beams = 4, #let's do best first search
                                                                            retokenize = False, debug = True, filename="phiModelNew",#FullCodeT5AllSQL
                                                                            simple_sql_fn = simple_sql_fn, dbs_full_schema = dbs_full_schema, use_best_first_search = use_best_first_search, 
                                                                            check_exec_result=check_exec_result, check_partial_sql = check_partial_sql, check_example = check_example,
                                                                            enum_part_quer_check = enum_part_quer_check, seq2seq = seq2seq, device = device)
    end_validate_time = time.time()
    print("Time taken to validate: " + str(end_validate_time - start_validate_time))
    print("pass")

    #process_sql.validate_partial_SQL(schemas[db_id], query)
    