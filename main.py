# ASSUMES MODEL IS TRAINED AND LOCATED AT ./v1/checkpoint; CP part is at end
import numpy as np
import torch

from transformers import AutoModelForSequenceClassification
from utils import load_tokenized_dataset
# %%
checkpoint_path = "./v1/checkpoint-8277"

# Load the model from the checkpoint
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# %%
model_path_deberta = 'microsoft/deberta-v3-small'
classes, class2id, id2class, tokenizer_deberta, tokenized_dataset = load_tokenized_dataset()
# %%
# RETURNS LOGITS
def predict_on_single_ex(idx, verbose=False):
    ex_ = tokenized_dataset['test'][idx]
    text = f"{ex_['title']}.\n{ex_['content']}"
    if verbose:
        print("INPUT:\n{}".format(text))
        print("LABEL (ORIGINAL):\n{}".format(ex_['all_labels']))
        print("LABEL (ONE-HOT):\n{}".format(ex_['labels']))
        # print("labels (one-hot): {}".format(torch.nonzero(torch.tensor(ex_['labels'])).squeeze()))
        print("DECODE ONE-HOT (double checking)")
        for i, val in enumerate(ex_['labels']):
            if val > 0: print(id2class[i])
        
    inputs = tokenizer_deberta(text, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    return outputs.logits

def sigmoid(x):
   return 1/(1 + np.exp(-x))
# %%
out = predict_on_single_ex(153, verbose=True)

# softmax_pred = torch.softmax(outputs.squeeze(), dim=0)
# sigmoid_pred = sigmoid(outputs.squeeze())
# %%
def predict_on_dataset(dataset, model, tokenizer, device, verbose=False):
    predictions = []
    logits_ = []
    sigmoids_ = []
    for example in dataset:
        text = f"{example['title']}.\n{example['content']}"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            logits_.append(logits)
            sigmoids_.append(torch.sigmoid(logits.squeeze()))
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            predictions.append(predicted_label)

        if verbose:
            print("Input:", text)
            print("Predicted Label:", predicted_label)

    return logits_, sigmoids_, predictions

logits_, sigmoids_, predictions = predict_on_dataset(tokenized_dataset['test'], model, tokenizer_deberta, device, verbose=True)

# %%
logits_[:5]
# %%
logits_ = [lg.squeeze() for lg in logits_]
logits_[:5]
# %%
logits_stacked = torch.stack(logits_)
sgmd_stacked = torch.stack(sigmoids_)

labels = tokenized_dataset['test']['labels'][:]
labels = torch.tensor(labels)
# %%
THRESHOLD = 0.66
CNT = 50

preds = []
for row in sgmd_stacked[:CNT]:
  tmp = []
  for j, colVal in enumerate(row):
    if colVal > 0.666:
      tmp.append(j)
  preds.append(tmp)

true_labels = []
for row in labels[:CNT]:
  tmp = []
  for j, colVal in enumerate(row):
    if colVal > 0:
      tmp.append(j)
  true_labels.append(tmp)

print("predictions")
print(preds)
print("true labels")
print(true_labels)
# %%
# Problem setup
n=125 # number of calibration points
alpha = 0.33 # 1-alpha is the desired false negative rate

def false_negative_rate(prediction_set, gt_labels):
    return 1-((prediction_set * gt_labels).sum(axis=1)/gt_labels.sum(axis=1)).mean()
# %%
sgmd = sgmd_stacked.to("cpu")
labels = labels.to("cpu")
# %%
# CONFORMAL PREDICTION PART
# Split the softmax scores into calibration and validation sets (save the shuffling)
idx = np.array([1] * n + [0] * (sgmd.shape[0]-n)) > 0
np.random.shuffle(idx)
cal_sgmd, val_sgmd = sgmd[idx,:], sgmd[~idx,:]
cal_labels, val_labels = labels[idx], labels[~idx]


from scipy.optimize import brentq

# Run the conformal risk control procedure
def lamhat_threshold(lam): return false_negative_rate(cal_sgmd>=lam, cal_labels) - ((n+1)/n*alpha - 1/(n+1))
lamhat = brentq(lamhat_threshold, 0, 1)
prediction_sets = val_sgmd >= lamhat

# Calculate empirical FNR
print(f"The empirical FNR is: {false_negative_rate(prediction_sets, val_labels)} and the threshold value is: {lamhat}")

# The empirical FNR is: 0.3346353769302368 and the threshold value is: 0.13474587351020462