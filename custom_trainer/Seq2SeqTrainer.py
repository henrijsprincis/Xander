from transformers import Trainer
from torch import nn
import torch


softmax = nn.Softmax(dim=2)
class Seq2SeqTrainer(Trainer):#compute normal loss
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        #softmax_logits = softmax(logits)#.type(torch.float64)
        nr_classes = max(logits.shape)
        labels_onehot = nn.functional.one_hot(labels,num_classes=nr_classes).type(torch.float16)#labels_to_onehot(labels)#nn.functional.one_hot()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1,nr_classes), labels_onehot.view(-1,nr_classes))#per token error #We can then do err.backward()
        return (loss, outputs) if return_outputs else loss