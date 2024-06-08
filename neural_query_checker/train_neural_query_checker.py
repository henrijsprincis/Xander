from CodeRL.src.transformers.models.t5 import T5ForConditionalGeneration
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(list):
    return torch.tensor(list, dtype=torch.int64).to(device).reshape(1, -1)

def train_verifier_model(config, dataset):
    model_verifier = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small", tuning_mode="critic", clone_rl_head=False).to(device)
    optimizer = torch.optim.Adam(model_verifier.parameters(), lr=config["lr"])
    model_verifier.train()
    for i in range(1):#50 epochs
        for item in dataset["train"]:
            input_ids = to_tensor(item["input_ids"])
            decoder_ids = to_tensor(item["decoder_ids"])
            error_types = to_tensor(item["error_type"])
            loss, prediction = model_verifier(input_ids=input_ids, error_types=error_types, labels=decoder_ids)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
    return model_verifier


def evaluate_verifier_model(dataset, model_verifier):
    model_verifier.eval()
    predicted_errors = []
    true_errors = []
    with torch.no_grad():
        for item in dataset["validation"]:
            input_ids = to_tensor(item["input_ids"])
            decoder_ids = to_tensor(item["decoder_ids"])
            error_types = to_tensor(item["error_type"])
            _, prediction = model_verifier(input_ids=input_ids, error_types=error_types, labels=decoder_ids)
            true_errors.append(item["error_type"])
            predicted_errors.append(prediction.item())
    return true_errors, predicted_errors