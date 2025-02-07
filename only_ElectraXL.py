import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import XLMRobertaTokenizer, XLMRobertaModel, BertTokenizer, BertModel, ElectraTokenizer, ElectraModel, get_scheduler
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = '/home/mois020f/LLM_emtion_detection/code/EmotionDetection/final_dataset.csv'

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
# Define a classifier model
class EmotionClassifier(nn.Module):
    def __init__(self, transformer_model, num_classes=6):
        super(EmotionClassifier, self).__init__()
        self.transformer = transformer_model
        self.fc = nn.Linear(self.transformer.config.hidden_size, num_classes)
        #self.fc = nn.Linear(768, num_classes)  # Project 768 hidden states to 6 output classes

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output"):
            pooled_output = outputs.pooler_output  # Use the pooler output if available
        else:
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)  # Mean pooling
        logits = self.fc(pooled_output)
        return logits

# Function to freeze lower layers
def freeze_lower_layers(model, freeze_until_layer):
    """
    Freeze lower layers for transformer models.
    
    Args:
        model: Hugging Face transformer model (e.g., BertModel, RobertaModel).
        freeze_until_layer: Number of lower layers to freeze (int).
    """
    for param in model.embeddings.parameters():
        param.requires_grad = False
    
    for i, layer in enumerate(model.encoder.layer):
        if i < freeze_until_layer:
            for param in layer.parameters():
                param.requires_grad = False
    
    print(f"Froze the first {freeze_until_layer} layers of the model.")
    
# Tokenizers and models
model_configs = {
     'electraXL': {
        'tokenizer': ElectraTokenizer.from_pretrained('google/electra-large-discriminator'),
        'model': ElectraModel.from_pretrained('google/electra-large-discriminator')
    }
     
   
}

# Hyperparameters
max_len = 128
batch_size = 16
epochs = 8
learning_rate = 2e-5
weight_decay = 1e-4  # L2 Regularization

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], labels, test_size=0.2, stratify=labels.argmax(axis=1), random_state=42
)

# Unified plotting
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Loop through models
best_f1 = 0
threshold = 0.5
    
f1_scores = {}

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
    
    # Freeze variable number of layers
    freeze_until_layer = 16  # Example: freeze first 12 layers
    freeze_lower_layers(model.transformer, freeze_until_layer)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate)
    #optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-8,weight_decay=weight_decay) #good for small dataset
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
      # Training and evaluation
    criterion = nn.BCEWithLogitsLoss()
    f1_scores[model_name] = []

    
    # Training loop
    for epoch in range(epochs):
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

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                logits = model(input_ids, attention_mask)
                val_preds.append(torch.sigmoid(logits).cpu())
                val_labels.append(labels.cpu())

        val_preds = torch.cat(val_preds, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        val_preds = (val_preds > threshold).astype(int)
        val_f1 = classification_report(val_labels, val_preds, target_names=emotion_columns, zero_division=0, output_dict=True)['weighted avg']['f1-score']
        f1_scores[model_name].append(val_f1)
        print(f"{model_name} | Epoch {epoch + 1}/{epochs} | Validation F1: {val_f1:.4f}")

        # Plot F1 Scores
    print(f"Final F1 Scores: {f1_scores}")

    plt.figure(figsize=(10, 6))
    for model_name, scores in f1_scores.items():
        plt.plot(range(1, epochs + 1), scores, label=model_name)
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Model Comparison by F1 Score')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "model_comparison_f1_score_onlyElectraXL_16_frozen_BiLSTM.png"))
    plt.show()


