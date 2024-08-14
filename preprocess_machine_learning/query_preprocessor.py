
from eval import process_sql
import copy
#ML STUFF
## Test stuff
from eval.getNumberCorrect import str_to_db_path, execute_query, evaluate_model
from open_ai.chat_gpt import ask_chatGPT as ask_chatGPT

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

def get_user_string_tokenizer(tokenizer):
    return "<|user|>"
    
def get_assistant_string_tokenizer(tokenizer):
    return "<|end|>\n<|assistant|>"

def get_model_input_string(user_token, 
                           assistant_token, 
                           schema_str, 
                           examples, 
                           config, 
                           first_tuple,
                           invalid,
                           simpleSQL,):
    task_instructions = user_token
    task_instructions += "Provide SQL that answers the following question: "+examples["question"]+"\n"
    task_instructions += "The schema: \n"+schema_str.replace("'","")+"\n"

    if config["add_execution_result"]:
        task_instructions += "This is the partial execution result of the query: "+str(first_tuple[:50]) # 50 characters of execution result (fairly sure [:50] should be outside of string)
    
    task_instructions += assistant_token
    desired_output = ""
    if config["use_simple_sql"] and invalid==False:
        desired_output += simpleSQL
    else:
        desired_output += examples["query"]
    task_instruction_answer = task_instructions + desired_output
    return task_instruction_answer, task_instructions, desired_output

def preprocess_query(examples, **kwargs):
    config = kwargs['config']
    tokenizer = kwargs['tokenizer']
    database_object = kwargs['database_object']
    user_token = get_user_string_tokenizer(tokenizer)
    assistant_token = get_assistant_string_tokenizer(tokenizer)

    invalid=False
    simpleSQL="NOTHING"
    if config["use_simple_sql"]:
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
    s = copy.deepcopy(database_object.schemas[examples["db_id"]])#deep copy schema.
    full_schema = database_object.dbs_full_schema[examples["db_id"]] # full schema
    for t_name in s.keys():
        for c_idx, c_name in enumerate(s[t_name]):
            #get column type.
            li = [t_name, c_name]
            idx = full_schema["column_names_original"].index(li)
            column_type = full_schema["column_types"][idx]
            s[t_name][c_idx] = c_name+" "+column_type
    
    schema_str = str(s)#.replace("'"," ").replace(","," ").replace(":", " ")
    # Execute query
    exec_result = execute_query(examples["db_id"], examples["query"])
    if exec_result[1]==[]:
        first_tuple = []
    else:
        first_tuple = exec_result[1][0]
    task_instruction_answer, task_instructions, desired_output = get_model_input_string(
        user_token = user_token, 
        assistant_token = assistant_token, 
        schema_str = schema_str, 
        examples = examples, 
        config = config, 
        first_tuple = first_tuple,
        invalid = invalid,
        simpleSQL = simpleSQL,)
    

    tokenizer.pad_token = tokenizer.eos_token
    partial_input = tokenizer(task_instructions, max_length=config["max_input_length"], truncation=True)
    if config["seq2seq"]:
        model_inputs = tokenizer(task_instructions, max_length=config["max_input_length"], truncation=True, padding='max_length')
        model_inputs["labels"] = tokenizer(desired_output, max_length=config["max_output_length"], truncation=True, padding='max_length')["input_ids"]
    else:#NTP
        model_inputs = tokenizer(task_instruction_answer, max_length=config["max_input_length"], truncation=True, padding='max_length')
        normalized_length = min(len(partial_input["input_ids"]),config["max_input_length"])
        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        model_inputs["labels"][:normalized_length] = [-100]*normalized_length
    model_inputs["schemas"] = str(database_object.schemas[examples["db_id"]])
    model_inputs["tooLong"] = False
    model_inputs["simpleSQL"] = simpleSQL
    model_inputs["execution_result"] = str(exec_result)
    model_inputs["partial_input"] = partial_input["input_ids"]
    if invalid:#todo check the length of the input_ids, pad_token not in model_inputs["input_ids"]
        model_inputs["tooLong"] = True
    return model_inputs