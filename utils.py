import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer

# PREPROCESS_FN IS SPECIFIC TO THIS DATASET
def load_tokenized_dataset(
        dataset_path='knowledgator/events_classification_biotech',
        model_path='microsoft/deberta-v3-small'):
    
    dataset = load_dataset(dataset_path)
    
    classes = [class_ for class_ in dataset['train'].features['label 1'].names if class_]
    class2id = {class_:id for id, class_ in enumerate(classes)}
    id2class = {id:class_ for class_, id in class2id.items()}
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def preprocess_function(example):
       text = f"{example['title']}.\n{example['content']}"
       all_labels = example['all_labels']# .split(', ')
       labels = [0. for i in range(len(classes))]
       for label in all_labels:
           label_id = class2id[label]
           labels[label_id] = 1.
    
       example = tokenizer(text, truncation=True)
       example['labels'] = labels
       return example
    
    tokenized_dataset = dataset.map(preprocess_function)
    
    return classes, class2id, id2class, tokenizer, tokenized_dataset