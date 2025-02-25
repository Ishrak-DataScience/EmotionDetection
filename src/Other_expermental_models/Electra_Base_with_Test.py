import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import ElectraTokenizer, ElectraModel, get_scheduler  
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn as nn
import numpy as np

# File path
file_path = '/home/rabo074f/LLM_Project_Ramy/LLM/final_dataset.csv'

# Emotion columns (including neutral as per the dataset)
emotion_columns = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

# Load dataset
df = pd.read_csv(file_path)

# Select required columns
df = df[['id', 'text'] + emotion_columns]

# Convert emotion columns to binary labels array
labels = df[emotion_columns].values

# Define a custom dataset
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
            'label': torch.tensor(label, dtype=torch.float)  
        }

# Define the BiLSTM classifier
class ElectraBiLSTMClassifier(nn.Module):  
    def __init__(self, electra_model, hidden_dim, num_labels):  
        super(ElectraBiLSTMClassifier, self).__init__()
        self.electra = electra_model 
        self.lstm = nn.LSTM(electra_model.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

# Tokenizer and hyperparameters
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
max_len = 128
batch_size = 16
epochs = 12
learning_rate = 2e-5
weight_decay = 1e-4  # L2 Regularization

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], labels, test_size=0.2, stratify=labels.argmax(axis=1), random_state=42
)

# Dataset
train_dataset = EmotionDataset(train_texts.to_numpy(), train_labels, tokenizer, max_len)
val_dataset = EmotionDataset(val_texts.to_numpy(), val_labels, tokenizer, max_len)

# Weighted Sampling
label_counts = train_labels.sum(axis=0)  # Count of each label
weights = 1.0 / label_counts  # Weight for each label
sample_weights = (train_labels * weights).sum(axis=1)  # Sample weights based on labels

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the BiLSTM model
electra_model = ElectraModel.from_pretrained('google/electra-base-discriminator')
model = ElectraBiLSTMClassifier(electra_model, hidden_dim=256, num_labels=len(emotion_columns))

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Freeze all layers of the Electra model initially
for param in model.electra.parameters():
    param.requires_grad = False

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # L2 Regularization
num_training_steps = epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Define BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

def unfreeze_layers(model, num_layers_to_unfreeze):
    # Unfreeze the last `num_layers_to_unfreeze` layers of the transformer
    total_layers = len(list(model.electra.encoder.layer))  
    layers_to_unfreeze = list(model.electra.encoder.layer)[-num_layers_to_unfreeze:]
    
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

def update_optimizer(optimizer, model):
    # Get parameters that are trainable and not already in the optimizer
    existing_params = {id(p) for group in optimizer.param_groups for p in group['params']}
    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in existing_params]
    
    if new_params:  # Add only if there are new parameters
        optimizer.add_param_group({'params': new_params, 'lr': learning_rate})

# Training and evaluation loops
def train_epoch(model, data_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return total_loss / len(data_loader), all_preds, all_labels

def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return total_loss / len(data_loader), all_preds, all_labels

# Gradual Unfreezing Schedule
unfrozen_steps = [0, 2, 4]  # Epochs when we unfreeze layers
layers_to_unfreeze_per_step = [4, 8, 12]  # Number of layers to unfreeze at each step

# Training loop
best_f1 = 0
threshold = 0.5
for epoch in range(epochs):
    # Gradually unfreeze layers based on the epoch
    if epoch in unfrozen_steps:
        index = unfrozen_steps.index(epoch)
        num_layers_to_unfreeze = layers_to_unfreeze_per_step[index]
        print(f"Unfreezing {num_layers_to_unfreeze} layers at epoch {epoch}")
        unfreeze_layers(model, num_layers_to_unfreeze)
        update_optimizer(optimizer, model)

    train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    val_loss, val_preds, val_labels = eval_model(model, val_loader, criterion, device)

    train_preds = (train_preds > threshold).astype(int)
    val_preds = (val_preds > threshold).astype(int)

    train_report = classification_report(train_labels, train_preds, target_names=emotion_columns, zero_division=0)
    val_report = classification_report(val_labels, val_preds, target_names=emotion_columns, zero_division=0)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Report:\n{val_report}")

print("Training complete!")

# Test Prediction
test_file_path = '/home/rabo074f/LLM_Project_Ramy/LLM/UpdatedData/public_data_test/track_a/test/eng.csv'
output_path = "/home/rabo074f/LLM_Project_Ramy/LLM/test_predictions_ELECTRA_BASE.csv"

# Load test data
test_df = pd.read_csv(test_file_path)

# Only the required emotions
test_emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Prepare test dataset
test_texts = test_df['text'].to_numpy()
test_dataset = EmotionDataset(test_texts, np.zeros((len(test_texts), len(emotion_columns))), tokenizer, max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Predict on test data
model.eval()
test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits = model(input_ids, attention_mask)
        test_preds.append(torch.sigmoid(logits).detach().cpu())

# Concatenate predictions
test_preds = torch.cat(test_preds, dim=0).numpy()

# Filter predictions for required emotions
test_preds_filtered = test_preds[:, :len(test_emotions)]

# Round predictions to two decimals
test_preds_filtered = np.round(test_preds_filtered, 2)

# Save predictions
test_predictions_df = test_df[['id', 'text']].copy()
test_predictions_df[test_emotions] = test_preds_filtered
test_predictions_df.to_csv(output_path, index=False)

print(f"Test predictions saved to {output_path}")
