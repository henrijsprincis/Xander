import torch
import sqlite3
import gc
from eval import process_sql
from eval.exec_eval import eval_exec_match_more as eval_exec_match_more

import ast #<- convert str(dict) -> dict
import queue#<- priority queue
import time
#evaluate#nr_exactly_correct = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
rewards = {"correct":1, "executes":0.5, "runTimeError" : -0.5, "syntaxError": -1}#0.5
encodings = {1:"y", 0.5:"e", -0.5:"r", -1:"s"}#yes, executes, runtime, syntax
error_counts = {"syntax": 0, "column": 0, "table":0, "other": 0}
exact_errors = {"syntax": [], "column": [], "table":[], "other": []}

def show_debug(dataset, query, i, valid = False):
    print("Q:", dataset[i]["question"])
    print("Schema:", dataset[i]["schemas"])
    print("Predicted Query: \n", query)
    print("Valid syntax: "+str(valid))
    print("Gold Query: \n", dataset[i]["query"],"\n")
    
def get_error_type(error):
    #4 types possible: syntax, column, table, other... 
    #Column, table, and other are classified as runtime errors
    #Rest are syntax errors. 
    if error.find("syntax") != -1:
        return "syntax"
    if error.find("column") != -1:
        return "column"
    if error.find("table") != -1:
        return "table"#print(error)#misuse of aggregate: max()
    return "other"

def get_error_type_and_append(error, query, append = False):
    error_type = get_error_type(error)
    if append:
        exact_errors[error_type].append((error,query))
    return error_type

def str_to_db_path(db_id):
    return f"./database/{db_id}/{db_id}.sqlite"

def execute_query(db_id, query, print_errors = False, error_logging = True):
    db_path =str_to_db_path(db_id)
    conn = sqlite3.connect(db_path)
    
    cursor = conn.cursor()
    try:
        execute = cursor.execute(query)
        execute_result = execute.fetchall()
        return 1, execute_result
    except sqlite3.Error as er:
        error_type = -1
        if error_logging:
            error_type = get_error_type_and_append(str(er), query, append = False)
        error_counts[error_type]+=1
        if print_errors:
            print("couldn't execute")
        return str(er), error_type
    except Exception as e:#for som reason it be possible to exec again
        print(f"exception! {e}")
        return -1, "Failed for unknown reason"

def schema_to_text(sc):
    out = ""
    for t_idx, t_name in enumerate(sc["table_names"]):
        out+= t_name+"\n"+"="*len(t_name)+"\n"
        for tup in zip(sc["column_names"], sc["column_types"]):    
            f_name = tup[0]
            if t_idx == f_name[0]:
                #right column for the table
                name = f_name[1]
                typ = tup[1]
                comb = name+":"+typ
                out+=comb+"\n"
        out+="\n"
    return out

def evaluate_model(model, tokenizer, spider, num_beams=1, max_output_sequence_length=200, 
                save_after_eval = False, use_train = False, debug = False, start_idx = 0, end_idx = 1, 
                retokenize = False, add_exec_output = False, filename="predsRLbeam", 
                do_sample = False, break_early = True, subtract_mean_batch = False, use_prob = False,
                simple_sql_fn=None, dbs_full_schema = {}, use_best_first_search = True, 
                check_exec_result = True, check_partial_sql = True, check_example = True,
                enum_part_quer_check = True, seq2seq = True, device = "cpu"):
    #retokenize => 1. encode whether query executed in label. 2. => add the language models prediction into problem description
    #add_exec_output => only works if retokenize = true ()
    #break early allows to exit loop once the first valid query is found
    #use_prob - when retokenize and add exec output, samples a subsequence with equal probability of being cut off at any point. (OLD)
    #check_exec_result - True when multiple attempts are allowed (and we check whether example tuple is in the output)
    #check_partial_sql - Whether or not to check partial SQL
    bos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    model.eval()
    nr_syntax_errors = 0
    nr_completely_correct = 0
    predicted_queries = ""
    queries_labelled = []
    dataset = spider["train"] if use_train else spider["validation"]
    total_reward = 0
    batch_reward = 0
    
    for i in range(start_idx, end_idx):#len(dataset)
        db_id = dataset[i]["db_id"]
        try:
            if use_best_first_search:
                predictions = bestFirstSearchWithChecks(model, dataset[i], tokenizer, simple_sql_fn, bm_size=num_beams, db_full = dbs_full_schema[db_id], 
                                                        check_exec_result=check_exec_result, check_partial_sql = check_partial_sql, 
                                                        check_example = check_example, seq2seq = seq2seq, device=device,
                                                        enum_part_quer_check=enum_part_quer_check)#
            else:
                predictions = beamSearchWithChecks(model, dataset[i], tokenizer, bm_size=num_beams, check_partial_sql = check_partial_sql, enum_part_quer_check = enum_part_quer_check)#
            predictions = [prediction.squeeze() for prediction in predictions]#squeeze em
        except Exception as e:
            print(f"ooopsssies {e}")
            predictions = [torch.tensor([bos_token, 4803, eos_token], dtype=torch.int64, device=device) for i in range(num_beams)]
            predictions = [prediction.squeeze() for prediction in predictions]#squeeze em
        
        #GOLD
        e, gold_result = execute_query(db_id, dataset[i]["query"])
        #try decoding
        for b in range(num_beams):
            query_org = tokenizer.decode(predictions[b], skip_special_tokens = True)
            query = query_org.strip()
            if simple_sql_fn:
                query = simple_sql_fn(query)
            d = dataset[i]
            d["pred"]=predictions[b]
            e, predicted_result = execute_query(db_id, query)
            d["predicted_result"] = predicted_result
            query += ";" if query.strip()[-1] != ";" else ""
            if e != 1:#failed to run
                if predicted_result == "syntax":
                    d["correct"]=rewards["syntaxError"]
                else:
                    d["correct"]=rewards["runTimeError"]
                nr_syntax_errors+=1
                predicted_queries += query+"\n"
                queries_labelled.append(d)
                if break_early:
                    break
                if debug:
                    show_debug(dataset, query, i)
            elif predicted_result != gold_result:    
                d["correct"]=rewards["executes"]
                predicted_queries += query+"\n"
                queries_labelled.append(d)
                if break_early:
                    break
            else:
                d["correct"]=rewards["correct"]
                #print(query)
                predicted_queries += query+"\n"
                nr_completely_correct+=1
                queries_labelled.append(d)
                if break_early:
                    break
            if debug:
                show_debug(dataset, query, i, valid=True)
                #if i+1 == num_beams: 
            total_reward += d["correct"]
            batch_reward += d["correct"] 
            #retokenize
            if retokenize:#makes label equal to code stuff. 
                tokenized_output = tokenizer(query_org, max_length=200, truncation=True, padding='max_length')#hard code 512
                d["predicted_input_ids"] = tokenized_output["input_ids"]
        
        if subtract_mean_batch:
            for iter in range(num_beams):
                queries_labelled[-1-iter]["correct"] = queries_labelled[-1-iter]["correct"]-batch_reward/num_beams
        if retokenize:
            bad_iters = []
            for iter in range(num_beams):
                execute = encodings[queries_labelled[-1-iter]["correct"]]#This code is atrocious
                queries_labelled[-1-iter]["labels"] = tokenizer(execute, max_length=3, truncation=True, padding='max_length')["input_ids"]#Start token, y token, end token
                if 0 not in queries_labelled[iter]["input_ids"]:
                    bad_iters.append(iter)
                    print("Query too long!")
            for index in sorted(bad_iters, reverse=True):
                del queries_labelled[index]
        batch_reward = 0
        if save_after_eval and i % 10 == 0:
            with open('results/'+filename+str(num_beams)+'PARTIAL.txt', 'w') as f:
                f.write(predicted_queries)
    mean_reward = total_reward/(num_beams*(end_idx-start_idx))
    #cleanup
    del predictions
    gc.collect()
    #print("Syntax errors: ", nr_syntax_errors)
    #print("Mean reward!:", mean_reward)
    #print("Nr completely correct: ", nr_completely_correct)
    if save_after_eval:
        with open('results/'+filename+str(num_beams)+'.txt', 'w') as f:
            f.write(";\n".join([query.replace("\n"," ") for query in predicted_queries.split(";")]))
    model.train()
    return nr_syntax_errors, mean_reward, queries_labelled

def expand_branches(model_query, hids, probs, inp, inp_att, hid, bm_size, seq2seq, bos_token, eos_token):
    sft = torch.nn.Softmax(dim=0)
    next_states = []
    next_probs  = []
    for idx, hid in enumerate(hids):
        #probs[idx]=probability
        #generate a token (only if last item on hid isn't equal to end token)
        if hid[0][-1] == eos_token: # if last item equals EOS token, then 
            next_states.append(hid)
            next_probs.append(probs[idx])
        else:
            with torch.no_grad():
                if seq2seq:
                    ret = model_query(input_ids = inp, attention_mask = inp_att, decoder_input_ids = hid)
                else:
                    inp_att = torch.ones((hid.shape)).to(device)#, attention_mask = inp_att
                    ret = model_query(input_ids = hid, attention_mask = inp_att)
                    #breakpoint()
                latest_logit = sft(ret.logits[0,-1,:])
                sorted, indices = torch.sort(latest_logit, 0, descending=True)
                for i in range(bm_size):#copy current beam.
                    pred_token =  indices[i].reshape(1,1)
                    prob_token = sorted[i]
                    sequence = torch.cat((hid, pred_token), dim = 1)
                    sequence_prob = probs[idx]*prob_token.item()
                    next_states.append(sequence)
                    next_probs.append(sequence_prob)
    return next_states, next_probs

def cancel_substring(input_prompt: torch.tensor, query: str, tokenizer):
    #in the case we are doing casual language modelling, we want to cancel the substring
    input_prompt = tokenizer.decode( input_prompt, skip_special_tokens = True).strip()
    return query.replace(input_prompt,"").strip()

def beamSearchWithChecks(model_query, q, tokenizer, bm_size = 2, check_partial_sql = True):
    bos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    schema  = ast.literal_eval(q["schemas"])
    inp     = torch.tensor(q["input_ids"]).reshape(1,-1).to(device)
    inp_att = torch.tensor(q["attention_mask"]).reshape(1,-1).to(device)
    hid     = torch.ones((1,1), dtype=torch.int64, device=device)*bos_token
    states  = [hid]#start off with zeros.
    probs   = [1]#initially probability is 1
    keep_generating = True
    while keep_generating:
        #step 1 expand
        states, probs = expand_branches(model_query, states, probs, inp, inp_att, hid, bm_size)
        #step 2, sort by probs
        sorted, indices = torch.sort(torch.tensor(probs), 0, descending=True)
        states = [states[idx] for idx in indices]#sorted
        probs = [probs[idx] for idx in indices]
        #step 3, filter bad states
        good_states = []
        substr_valid = True
        for idx, partial_state in enumerate(states):
            substr = tokenizer.decode(partial_state.squeeze(), skip_special_tokens=True)
            if check_partial_sql:
                substr_valid = process_sql.validate_partial_SQL(schema, substr)
            if substr_valid:
                good_states.append(idx)
        if good_states == []:
            print("ALL beams failed. Replacing query with a simple select ")
            return [torch.tensor([bos_token, 4803, eos_token], dtype=torch.int64, device=device) for i in range(bm_size)]
            
        #step 4, get rid of bad states
        states = [states[idx] for idx in good_states]
        probs  = [probs[idx] for idx in good_states]
        #step 5, retain only top beamsearch branches
        states=states[:bm_size]
        probs=probs[:bm_size]
        #finally check if we are done
        keep_generating = False
        for current_state in states:
            if current_state[0][-1] != eos_token:
                keep_generating = True
    return states

def bestFirstSearchWithChecks(model_query, q, tokenizer, simple_sql_fn, bm_size = 2, check_example = True, db_full = {}, 
                              check_exec_result = True, check_partial_sql = True, enum_part_quer_check = True,
                              seq2seq = True, device = "cpu"):
    bos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    schema  = ast.literal_eval(q["schemas"])
    inp     = torch.tensor(q["input_ids"]).reshape(1,-1).to(device)
    inp_att = torch.tensor(q["attention_mask"]).reshape(1,-1).to(device)
    hid     = torch.ones((1,1), dtype=torch.int64, device=device)*bos_token
    pri_q_sz= 10000#2000
    states  = queue.PriorityQueue(10000)#gets lowest priority element first -1 is highest priority.
    if seq2seq:
        states.put(((-1, time.time()), hid))#priority, state
    else:
        hid = tokenizer(tokenizer.decode(q["partial_input"],skip_special_tokens=False), return_tensors = "pt")["input_ids"].to(device)
        states.put(((-1, time.time()), hid))
    start_time = time.time()
    max_runtime = 60
    number_complete_queries = 0
    while not states.empty():#or while que not empty (same thing)
        if time.time() - start_time > max_runtime:
            print(q["query"])
            print("RAN OUT OF TIME, RETURNING EMPTY SELECT")
            print("number_complete_queries: ", number_complete_queries)
            return [torch.tensor([bos_token, 4803, eos_token], dtype=torch.int64, device=device) for i in range(bm_size)]
        #step 1 get highest priority branch (state)
        try:
            prob, state = states.get()
            prob = prob[0]#get the probability
        except:
            print("excption getting item, continuing")
            continue
        if state[0][-1] == eos_token: # check if we are done
            quer = tokenizer.decode(state.squeeze(), skip_special_tokens=True)
            if not seq2seq:
                quer = cancel_substring(q["partial_input"], quer, tokenizer)
                quer = quer.replace("<s>","").replace("</s>","")
            if simple_sql_fn:#if we r using simple SQL
                quer_conv = simple_sql_fn(quer)
            else:
                quer_conv = quer
            if quer_conv == "":
                continue
            g_exec = execute_query(q["db_id"], q["query"])[-1]
            g_exec_set = [set(elem) for elem in g_exec]#list containing sets
            #get close queries
            close_queries = [] 
            if enum_part_quer_check:
                close_queries = process_sql.get_1_off_queries(quer_conv, db_full)#get 1 off queries
                #breakpoint()
            close_queries = [quer_conv] + close_queries
            for quer_conv in close_queries:
                number_complete_queries +=1
                if time.time() - start_time > max_runtime:
                    print(q["query"])
                    print("RAN OUT OF TIME, RETURNING EMPTY SELECT")
                    print("number_complete_queries: ", number_complete_queries)
                    return [torch.tensor([bos_token, 4803, eos_token], dtype=torch.int64, device=device) for i in range(bm_size)]
                quer_exec = execute_query(q["db_id"], quer_conv)[-1]
                #check they are equal using original code
                quer_exec_set = [set(elem) for elem in quer_exec]
                if check_exec_result:
                    [exec_score, thin] = eval_exec_match_more(str_to_db_path(q["db_id"]), quer_conv, q["query"], False, False, False)
                    if exec_score:
                        return [torch.tensor([[bos_token]+tokenizer(quer_conv)["input_ids"]], device=device)]
                    if len(g_exec)==0:#check first elem
                        if g_exec == quer_exec:
                            return[state]
                        else:
                            continue
                    else:
                        first_elem = g_exec_set[0]#g_exec
                    if len(quer_exec) == 0:
                        continue
                    try:#try to check if the first element is in quer_exec result
                        if first_elem in quer_exec_set:
                            return [torch.tensor([[bos_token]+tokenizer(quer_conv)["input_ids"]], device=device)]
                            #return [state]
                        else:
                            continue
                    except:
                        continue
                else:
                    return [torch.tensor([[bos_token]+tokenizer(quer_conv)["input_ids"]], device=device)]
        if state[0][-1] == eos_token:
            continue
        #step 1 expand
        next_states, probs = expand_branches(model_query, [state], [prob], inp, inp_att, hid, bm_size, seq2seq, bos_token, eos_token)
        #step 3, filter bad states
        good_states = []
        for idx, partial_state in enumerate(next_states):
            substr = tokenizer.decode(partial_state.squeeze(), skip_special_tokens=True)
            if not seq2seq:
                substr = cancel_substring(q["partial_input"], substr, tokenizer)
                lower = substr.lower()
                if (lower.count(";") == 1) or (lower.count("select") == lower.count("union none") and lower.count("select") > 0):
                    #print(lower)
                    #the query is done#breakpoint()
                    next_states[idx] = torch.cat((partial_state, torch.tensor(eos_token, device=device).reshape(1,-1)),dim=1)

            if idx==0:
                pass
                #print(substr)
            
            substr_valid = True
            if check_partial_sql:# if we want to perform syntax checking on partial query
                if check_example and eval(q["execution_result"])[-1] != []:#check example
                    example = eval(q["execution_result"])[-1][0]
                    substr_valid = process_sql.validate_partial_SQL(schema, substr, example = example, db_full = db_full)#requires types from full schema
                else:#chek syntax
                    substr_valid = process_sql.validate_partial_SQL(schema, substr)
            if substr_valid:#substr_valid
                good_states.append(idx)
            else:
                print("question: ", tokenizer.decode(partial_state.squeeze(), skip_special_tokens=True))
                print("schema: ", schema)
                print("substr: ", substr)
                x=10
            #    breakpoint()
        #step 4, get rid of bad states
        next_states = [next_states[idx] for idx in good_states]
        probs       = [probs[idx] for idx in good_states] #-0.1
        #retain all of them
        for idx, state in enumerate(next_states):
            if states.qsize() < pri_q_sz:
                priority = (probs[idx], time.time())
                states.put((priority, state))

    print(q["query"])  
    print("We have exhausted priority que")
    return [torch.tensor([bos_token, 4803, eos_token], dtype=torch.int64, device=device) for i in range(bm_size)]