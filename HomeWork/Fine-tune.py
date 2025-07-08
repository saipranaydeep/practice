import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EventDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        labels = self.labels[idx]

        # Tokenize the sentence
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

class MultiLabelMiniLM(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', num_labels=3, dropout_rate=0.3):
        super(MultiLabelMiniLM, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_data(file_path='dataset.csv'):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Dataset shape: {df.shape}")

        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())

        # Check for missing values
        print(f"\nMissing values:\n{df.isnull().sum()}")

        # Get sentences and labels
        sentences = df['sentence'].tolist()
        labels = df[['title', 'datetime', 'location']].values.tolist()

        # Display class distribution
        print(f"\nClass distribution:")
        print(f"Title: {df['title'].sum()}/{len(df)} ({df['title'].mean():.2%})")
        print(f"Datetime: {df['datetime'].sum()}/{len(df)} ({df['datetime'].mean():.2%})")
        print(f"Location: {df['location'].sum()}/{len(df)} ({df['location'].mean():.2%})")

        return sentences, labels

    except FileNotFoundError:
        print("dataset.csv not found. Please make sure the file is in the current directory.")
        return None, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def train_model(model, train_loader, val_loader, epochs=10, lr=2e-5, device='cuda'):
    """Train the model with early stopping and regularization"""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    model.train()

    for epoch in range(epochs):
        total_train_loss = 0

        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 50)

        # Training phase
        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                # Get predictions
                predictions = torch.sigmoid(outputs) > 0.5
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Exact match accuracy (all labels must match)
        exact_match = accuracy_score(all_labels, all_predictions)
        # Hamming loss (average across all labels)
        hamming_loss_val = hamming_loss(all_labels, all_predictions)

        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Exact Match Accuracy: {exact_match:.4f}')
        print(f'Hamming Loss: {hamming_loss_val:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = torch.sigmoid(outputs) > 0.5

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    exact_match = accuracy_score(all_labels, all_predictions)
    hamming_loss_val = hamming_loss(all_labels, all_labels)

    print(f"\nTest Results:")
    print(f"Exact Match Accuracy: {exact_match:.4f}")
    print(f"Hamming Loss: {hamming_loss_val:.4f}")

    # Per-class metrics
    class_names = ['title', 'datetime', 'location']
    for i, class_name in enumerate(class_names):
        class_acc = accuracy_score(all_labels[:, i], all_predictions[:, i])
        print(f"{class_name.capitalize()} Accuracy: {class_acc:.4f}")

    return all_predictions, all_labels

def predict_class_combination(predictions):
    """Convert predictions to 8-class format"""
    class_combinations = []
    for pred in predictions:
        # Convert binary predictions to class number (0-7)
        class_num = pred[0] * 4 + pred[1] * 2 + pred[2] * 1
        class_combinations.append(class_num)
    return class_combinations

def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Main training pipeline"""
    # Load data
    sentences, labels = load_data('dataset.csv')
    if sentences is None:
        return

    # Split data
    train_sentences, temp_sentences, train_labels, temp_labels = train_test_split(
        sentences, labels, test_size=0.3, random_state=42, stratify=labels
    )

    val_sentences, test_sentences, val_labels, test_labels = train_test_split(
        temp_sentences, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"\nData split:")
    print(f"Train: {len(train_sentences)} samples")
    print(f"Validation: {len(val_sentences)} samples")
    print(f"Test: {len(test_sentences)} samples")

    # Initialize tokenizer
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create datasets
    train_dataset = EventDataset(train_sentences, train_labels, tokenizer)
    val_dataset = EventDataset(val_sentences, val_labels, tokenizer)
    test_dataset = EventDataset(test_sentences, test_labels, tokenizer)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = MultiLabelMiniLM(model_name=model_name, num_labels=3, dropout_rate=0.3)
    model.to(device)

    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=15, lr=2e-5, device=device
    )

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Evaluate model
    predictions, true_labels = evaluate_model(model, test_loader, device)

    # Convert to 8-class format
    pred_classes = predict_class_combination(predictions)
    true_classes = predict_class_combination(true_labels)

    print(f"\n8-Class Classification Results:")
    print(f"Classes: 0=(0,0,0), 1=(0,0,1), 2=(0,1,0), 3=(0,1,1), 4=(1,0,0), 5=(1,0,1), 6=(1,1,0), 7=(1,1,1)")

    # Print classification report for 8-class
    from collections import Counter
    print(f"\nPredicted class distribution: {Counter(pred_classes)}")
    print(f"True class distribution: {Counter(true_classes)}")

    # Save model
    torch.save(model.state_dict(), 'final_model.pth')
    print("\nModel saved as 'final_model.pth'")

if __name__ == "__main__":
    main()
