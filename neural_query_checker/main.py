import json
from neural_query_checker.dataset_generator import get_dataset
from neural_query_checker.train_neural_query_checker import train_verifier_model, evaluate_verifier_model
from sklearn.metrics import confusion_matrix


with open("config.json") as f:
    config = json.load(f)

dataset = get_dataset(config)
model_verifier = train_verifier_model(config, dataset)
true_errors, predicted_errors = evaluate_verifier_model(dataset, model_verifier)
print(confusion_matrix(true_errors, predicted_errors))

print("Success")