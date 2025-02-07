import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import XLMRobertaTokenizer, XLMRobertaModel, BertTokenizer, BertModel, ElectraTokenizer, ElectraModel, get_scheduler
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = '/home/moi5208f/LLM_emtion_detection/code/EmotionDetection/Final_dataset.csv'

# Emotion columns (including neutral as per the dataset)
emotion_columns = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

# Load dataset
df = pd.read_csv(file_path)

# Select required columns
df = df[['id', 'text'] + emotion_columns]

# Convert emotion columns to binary labels array
labels = df[emotion_columns].values

# Define a Custom dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)  # Float for BCEWithLogitsLoss
        }

# Define the EmotionClassifier
class EmotionClassifier(nn.Module):
    def __init__(self, transformer_model, num_classes=6):
        super(EmotionClassifier, self).__init__()
        self.transformer = transformer_model
        self.fc = nn.Linear(768, num_classes)  # Project 768 hidden states to 6 output classes
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output"):
            pooled_output = outputs.pooler_output  # Use the pooler output if available
        else:
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)  # Mean pooling
        logits = self.fc(pooled_output)
        return logits

# Tokenizers and models
model_configs = {
    'xlm-roberta': {
        'tokenizer': XLMRobertaTokenizer.from_pretrained('xlm-roberta-base'),
        'model': XLMRobertaModel.from_pretrained('xlm-roberta-base')
    },
    'bert-multilingual': {
        'tokenizer': BertTokenizer.from_pretrained('bert-base-multilingual-cased'),
        'model': BertModel.from_pretrained('bert-base-multilingual-cased')
    },
    'electra': {
        'tokenizer': ElectraTokenizer.from_pretrained('google/electra-base-discriminator'),
        'model': ElectraModel.from_pretrained('google/electra-base-discriminator')
    },
    'electraXL': {
        'tokenizer': ElectraTokenizer.from_pretrained('google/electra-large-discriminator'),
        'model': ElectraModel.from_pretrained('google/electra-large-discriminator')
    }
}

# Hyperparameters
max_len = 128
batch_size = 16
epochs = 6
learning_rate = 2e-5
weight_decay = 1e-4  # L2 Regularization

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], labels, test_size=0.2, stratify=labels.argmax(axis=1), random_state=42
)

# Unified plotting
f1_scores = {}
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Loop through models
for model_name, config in model_configs.items():
    tokenizer = config['tokenizer']
    transformer_model = config['model']
    
    train_dataset = EmotionDataset(train_texts.to_numpy(), train_labels, tokenizer, max_len)
    val_dataset = EmotionDataset(val_texts.to_numpy(), val_labels, tokenizer, max_len)
    
    sampler = WeightedRandomSampler(train_labels.sum(axis=1), num_samples=len(train_labels), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the classifier model
    model = EmotionClassifier(transformer_model, num_classes=len(emotion_columns))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-8, weight_decay=weight_decay)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_loader))
    
    # Define BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = 0.5
    f1_scores[model_name] = []

    for epoch in range(epochs):
        # Training process (unchanged)
        pass

    # Save plot
    plt.savefig(os.path.join(output_dir, f"{model_name}_f1_score_plot_withoutBiLSTM.png"))

plt.show()
