import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Define Data Pipeline Class
class EmotionDataset(Dataset):
    def __init__(self, file_path, tokenizer_name, max_length=128):
        self.data = pd.read_csv(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.labels = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']  # Assuming these are column names
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Text']
        labels = row[self.labels].values.astype(float)  # Convert to float for multi-label
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Define Model Class
class EmotionModel:
    def __init__(self, model_name, num_labels=5):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")
    
    def train(self, train_dataset, val_dataset, output_dir="./results"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
    
    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
        labels = labels.astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        accuracy = accuracy_score(labels, predictions)
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

# Define Prediction Class
class EmotionPredictor:
    def __init__(self, model, tokenizer_name):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            predictions = torch.sigmoid(logits).cpu().numpy()[0]  # Apply sigmoid to get probabilities
        return {emotion: prob for emotion, prob in zip(['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'], predictions)}

# Load and Preprocess Data
train_dataset = EmotionDataset(file_path="/mnt/data/eng.csv", tokenizer_name="distilbert-base-uncased")
val_dataset = EmotionDataset(file_path="/mnt/data/eng.csv", tokenizer_name="distilbert-base-uncased")

# Initialize and Train Model
emotion_model = EmotionModel(model_name="distilbert-base-uncased")
emotion_model.train(train_dataset, val_dataset)

# Prediction Example
predictor = EmotionPredictor(emotion_model.model, tokenizer_name="distilbert-base-uncased")
sample_text = "I'm thrilled to be part of this event!"
prediction = predictor.predict(sample_text)
print(prediction)
