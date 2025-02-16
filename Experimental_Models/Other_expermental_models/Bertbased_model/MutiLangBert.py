import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load all CSV files from the specified folder
#folder_path = '/home/rabo074f/LLM_EMTION_DETECTION/train'
folder_path= "/home/mois020f/LLM_emtion_detection/code/EmotionDetection/public_data/train/track_a"
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Combine all CSV files into a single DataFrame
df = pd.concat((pd.read_csv(file) for file in all_files), ignore_index=True)

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
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Preprocess data with Multilingual BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
max_len = 128
labels = df[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']].idxmax(axis=1).factorize()[0]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], labels, test_size=0.2, random_state=42
)

train_dataset = EmotionDataset(train_texts.to_numpy(), train_labels, tokenizer, max_len)
val_dataset = EmotionDataset(val_texts.to_numpy(), val_labels, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load Multilingual BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits


        loss.backward()
        optimizer.step()
        total_loss += loss.depatch().cpu().item()
        optimizer.zero_grad()

    return total_loss / len(data_loader)

# Validation loop
def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return classification_report(true_labels, predictions, target_names=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])

# Train the model
epochs = 10
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}")

    val_report = eval_model(model, val_loader, device)
    print(f"Validation Report:\n{val_report}")

# Save the model
torch.save(model.state_dict(), 'emotion_model_multilingual.pth')