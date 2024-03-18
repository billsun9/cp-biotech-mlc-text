import numpy as np
import torch

from datasets import load_dataset

dataset = load_dataset('knowledgator/events_classification_biotech')

print(dataset)

print(dataset['train'])

# %%
dataset['train']
# %%
dataset['train'][1000:1100]['all_labels']
# %%
def getCounts(labels_list): # List[List[Str]]
  d = {}
  for y in labels_list:
    for label in y:
      try: d[label] += 1
      except KeyError: d[label] = 1
  return d

train_counts = getCounts(dataset['train']['all_labels']) 
test_counts = getCounts(dataset['test']['all_labels'])

print("TRAIN")
print(sorted(
    [(cnt, label) for label, cnt in train_counts.items()],
    reverse=True
))
print("TEST")
print(sorted(
    [(cnt, label) for label, cnt in test_counts.items()],
    reverse=True
))