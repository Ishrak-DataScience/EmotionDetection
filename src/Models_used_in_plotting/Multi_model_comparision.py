import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer,RobertaModel, BertTokenizer, BertModel, ElectraTokenizer, ElectraModel, get_scheduler
from torch.optim import AdamW,Adam
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
    'electraXL': {
        'tokenizer': ElectraTokenizer.from_pretrained('google/electra-large-discriminator'),
        'model': ElectraModel.from_pretrained('google/electra-large-discriminator')
    },
    'roberta': {
        'tokenizer': RobertaTokenizer.from_pretrained('roberta-base'),
        'model': RobertaModel.from_pretrained('roberta-base')
    },
    'bert-English': {
        'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
        'model': BertModel.from_pretrained('bert-base-uncased')
    },
    'electra': {
        'tokenizer': ElectraTokenizer.from_pretrained('google/electra-base-discriminator'),
        'model': ElectraModel.from_pretrained('google/electra-base-discriminator')
    } 
    
}

# Hyperparameters
max_len = 128
batch_size = 16
epochs = 14
learning_rate = 2e-5
#weight_decay = 1e-4  # L2 Regularization

# Gradual Unfreezing Schedule
unfrozen_steps = [1, 2, 4, 6, 8, 10, 12]  # Epochs when we unfreeze layers
layers_to_unfreeze_per_step = [1, 2, 3, 4, 6, 8, 12]  # Number of layers to unfreeze at each step

def unfreeze_layers(model, num_layers_to_unfreeze):
    # Unfreeze the last `num_layers_to_unfreeze` layers of the transformer
    total_layers = len(list(model.transformer.encoder.layer))
    layers_to_unfreeze = list(model.transformer.encoder.layer)[-num_layers_to_unfreeze:]

    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

def update_optimizer(optimizer, model):
    # Get parameters that are trainable and not already in the optimizer
    existing_params = {id(p) for group in optimizer.param_groups for p in group['params']}
    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in existing_params]

    if new_params:  # Add only if there are new parameters
        optimizer.add_param_group({'params': new_params, 'lr': learning_rate})

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
    model = EmotionClassifier(transformer_model, num_classes=len(emotion_columns))

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate)
    #optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-8,weight_decay=weight_decay)
    #optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # L2 Regularization
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Define BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # Training and evaluation
    best_f1 = 0
    threshold = 0.5
    f1_scores[model_name] = []

    train_loss_plot_path = os.path.join(output_dir, f"{model_name}_training_validation_loss.png")
    f1_score_plot_path = os.path.join(output_dir, f"{model_name}_validation_f1_score.png")

    for epoch in range(epochs):
        # Gradually unfreeze layers based on the epoch
        if epoch in unfrozen_steps:
            index = unfrozen_steps.index(epoch)
            num_layers_to_unfreeze = layers_to_unfreeze_per_step[index]
            print(f"Unfreezing {num_layers_to_unfreeze} layers at epoch {epoch}")
            unfreeze_layers(model, num_layers_to_unfreeze)
            update_optimizer(optimizer, model)

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

        # Validation
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

        val_report = classification_report(val_labels, val_preds, target_names=emotion_columns, zero_division=0, output_dict=True)
        val_f1 = val_report['weighted avg']['f1-score']
        f1_scores[model_name].append(val_f1)

        print(f"{model_name} | Epoch {epoch + 1}/{epochs} | Validation F1: {val_f1:.4f}")

# Plot F1 Scores
plt.figure(figsize=(10, 6))
for model_name, scores in f1_scores.items():
    plt.plot(range(1, epochs + 1), scores, label=model_name)

plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Model Comparison by F1 Score')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "model_comparison_f1_score_without_weighted_decay_english_only_base_models.png"))
plt.show()
