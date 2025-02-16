import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import XLMRobertaTokenizer, XLMRobertaModel, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/mois020f/LLM_emtion_detection/code/EmotionDetection/final_dataset.csv'

emotion_columns = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

df = pd.read_csv(file_path)

df = df[['id', 'text'] + emotion_columns]

labels = df[emotion_columns].values

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

class XLMRobertaBiLSTMClassifier(nn.Module):
    def __init__(self, xlmroberta_model, hidden_dim, num_labels):
        super(XLMRobertaBiLSTMClassifier, self).__init__()
        self.xlmroberta = xlmroberta_model
        self.lstm = nn.LSTM(xlmroberta_model.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlmroberta(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

def get_optimizer_with_llrd(model, base_lr, decay_rate):
    param_groups = []
    param_groups.append({"params": model.fc.parameters(), "lr": base_lr})
    for i, layer in enumerate(model.xlmroberta.encoder.layer[::-1]):
        lr = base_lr * (decay_rate ** i)
        param_groups.append({"params": layer.parameters(), "lr": lr})
    param_groups.append({"params": model.xlmroberta.embeddings.parameters(), "lr": base_lr * (decay_rate ** len(model.xlmroberta.encoder.layer))})
    return AdamW(param_groups)

num_layers_to_freeze = int(input("Enter the number of layers to freeze: "))
num_epochs = int(input("Enter the number of epochs: "))

max_len = 256
batch_size = 16
base_lr = 5e-5
decay_rate = 0.9
weight_decay = 1e-4

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], labels, test_size=0.2, stratify=labels.argmax(axis=1), random_state=42
)

train_dataset = EmotionDataset(train_texts.to_numpy(), train_labels, tokenizer, max_len)
val_dataset = EmotionDataset(val_texts.to_numpy(), val_labels, tokenizer, max_len)

label_counts = train_labels.sum(axis=0)
weights = 1.0 / label_counts
sample_weights = (train_labels * weights).sum(axis=1)

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

xlmroberta_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
model = XLMRobertaBiLSTMClassifier(xlmroberta_model, hidden_dim=256, num_labels=len(emotion_columns))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for i, layer in enumerate(model.xlmroberta.encoder.layer):
    if i < num_layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False

print("Fixed layers:")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(name)

print("Trainable layers:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

optimizer = get_optimizer_with_llrd(model, base_lr, decay_rate)
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

criterion = nn.BCEWithLogitsLoss()

def plot_single_model_losses(train_losses, val_losses, num_epochs, model_name, output_file):
    """
    Plot training and validation losses for a single model.

    Args:
        train_losses: List of training losses for each epoch.
        val_losses: List of validation losses for each epoch.
        num_epochs: Total number of epochs.
        model_name: Name of the model (e.g., 'BERT', 'RoBERTa').
        output_file: Path to save the plot.
    """
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label=f"{model_name} Training Loss", marker='o')
    plt.plot(epochs, val_losses, label=f"{model_name} Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.close()
    
def plot_metrics(f1_scores, num_epochs, model_name, output_file):
    """
    Plot F1-score for the model.

    Args:
        f1_scores: List of F1-scores for each epoch.
        num_epochs: Total number of epochs.
        model_name: Name of the model.
        output_file: Path to save the plot.
    """
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, f1_scores, label=f"{model_name} Validation F1-Score", marker="x", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("F1-Score")
    plt.title(f"{model_name} Training & Validation Metrics")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.close()


def evaluate_model_with_f1(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy() > 0.5
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")
    f1 = f1_score(all_labels, all_preds, average="micro")
    return total_loss / len(loader), precision, recall, f1

train_losses = []
val_losses = []
f1_scores = []
patience = 3
no_improvement_epochs = 0
best_f1 = 0

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

train_loss_plot_path = os.path.join(output_dir, "XlmRberta_frozen_IIRD_training_validation_loss.png")
f1_score_plot_path = os.path.join(output_dir, "lmRberta_frozen_IIRD_validation_f1_score.png")


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
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

    train_losses.append(total_loss / len(train_loader))
    val_loss, precision, recall, f1 = evaluate_model_with_f1(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    f1_scores.append(f1)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= patience:
        print("Early stopping due to no improvement in F1-score.")
        break

plot_single_model_losses(train_losses, val_losses, epoch+1, "XLM-RoBERTa", "training_validation_loss.png")
plot_metrics(f1_scores, epoch+1, "XLM-RoBERTa", "validation_f1_score.png")
print("Training complete! Plots saved.")
