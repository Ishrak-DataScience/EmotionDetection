# Define Data Pipeline Class
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class EmotionDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.labels = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Text']
        labels = row[self.labels].values.astype(float)
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }
        
#train_dataset = EmotionDataset(data_path="/mnt/data/eng.csv", tokenizer_name="distilbert-base-uncased")