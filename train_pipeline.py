import numpy as np
import torch

from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from utils import load_tokenized_dataset

import evaluate
# %%
model_path_deberta = 'microsoft/deberta-v3-small'
classes, class2id, id2class, tokenizer_deberta, tokenized_dataset = load_tokenized_dataset()
# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer_deberta)
# %%
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred):

   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
# %%
model = AutoModelForSequenceClassification.from_pretrained(
    model_path_deberta,
    num_labels=len(classes),
    id2label=id2class,
    label2id=class2id,
    problem_type = "multi_label_classification"
)
# %%
training_args = TrainingArguments(
   output_dir="v1",
   learning_rate=2e-5,
   per_device_train_batch_size=1, # should prolly be >1
   per_device_eval_batch_size=1, # should prolly be >1
   num_train_epochs=2, # should prolly be >5
   weight_decay=0.01,
   evaluation_strategy="epoch",
   save_strategy="epoch",
   load_best_model_at_end=True,
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["test"],
   tokenizer=tokenizer_deberta,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()