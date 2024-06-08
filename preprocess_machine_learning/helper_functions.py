def reset_params(model):
    print("Resetting weights")
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def get_save_name(model_checkpoint, add_execution_result, use_simple_sql, no_weights = 0):
    return model_checkpoint[-7:]+"Examples"+str(add_execution_result)+"SimpleSQL"+str(use_simple_sql)+"NoWeights"*no_weights

def get_save_paths(config):
    checkpoint_path = config["model_checkpoint"]
    save_path = "./checkpoints/GoodCheckpoint/"+get_save_name(
            config["model_checkpoint"],
            config["add_execution_result"],
            config["use_simple_sql"])
    if config["use_good_checkpoint_query"]:
        checkpoint_path = save_path
    return checkpoint_path, save_path

