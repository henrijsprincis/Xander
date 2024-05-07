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