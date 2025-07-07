# Cell 1: Install and Import Dependencies
"""
Run this cell first to install required packages if not already installed:
!pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # Import from torch.optim instead of transformers
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("All dependencies imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Cell 2: Load Local MiniLM Model and Tokenizer
"""
Load your local MiniLM model from ./minilm folder
"""

# Path to your local model
MODEL_PATH = "./minilm"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please check the path.")

# Load tokenizer and model from local directory
print("Loading local MiniLM model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"Tokenizer loaded successfully from {MODEL_PATH}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Cell 3: Define Custom Dataset Class
"""
Custom Dataset class for handling event classification data
"""

class EventDataset(Dataset):
    """Custom Dataset for Event Classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

print("EventDataset class defined successfully!")

# Cell 4: Define Model Architecture
"""
MiniLM-based Multi-label Event Classifier
"""

class MiniLMEventClassifier(nn.Module):
    """MiniLM-based Multi-label Event Classifier"""
    
    def __init__(self, model_path, num_labels=3, dropout_rate=0.3):
        super(MiniLMEventClassifier, self).__init__()
        self.num_labels = num_labels
        
        # Load pre-trained MiniLM from local path
        self.bert = AutoModel.from_pretrained(model_path)
        
        # Freeze early layers to prevent overfitting
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Only train last 4 layers
        total_layers = len(self.bert.encoder.layer)
        layers_to_freeze = max(0, total_layers - 4)
        
        for i in range(layers_to_freeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"Frozen {layers_to_freeze} layers out of {total_layers} total layers")
        
        # Classification head with dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# Initialize model with local MiniLM
model = MiniLMEventClassifier(model_path=MODEL_PATH, num_labels=3, dropout_rate=0.3)
print("Model initialized successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Cell 5: Data Preparation Functions
"""
Functions to prepare data for training (labels are already in multi-label format)
"""

def prepare_data(texts, labels, tokenizer, test_size=0.2, val_size=0.1):
    """Prepare train, validation, and test datasets"""
    
    # Labels are already in multi-label format [title, datetime, location]
    # Convert to numpy array for easier handling
    labels_array = np.array(labels)
    
    # Create a single label for stratification (combination of all three labels)
    stratify_labels = []
    for label in labels:
        # Convert [title, datetime, location] to single integer for stratification
        stratify_label = label[0] * 4 + label[1] * 2 + label[2]  # Binary to decimal
        stratify_labels.append(stratify_label)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=test_size + val_size, random_state=42, 
        stratify=stratify_labels
    )
    
    # Create stratify labels for second split
    temp_stratify = []
    for label in y_temp:
        temp_stratify.append(label[0] * 4 + label[1] * 2 + label[2])
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size + val_size), 
        random_state=42, stratify=temp_stratify
    )
    
    # Create datasets
    train_dataset = EventDataset(X_train, y_train, tokenizer)
    val_dataset = EventDataset(X_val, y_val, tokenizer)
    test_dataset = EventDataset(X_test, y_test, tokenizer)
    
    return train_dataset, val_dataset, test_dataset

def analyze_data_split(train_dataset, val_dataset, test_dataset):
    """Analyze the data split to ensure good distribution"""
    
    def get_label_stats(dataset):
        labels = [dataset[i]['labels'].numpy() for i in range(len(dataset))]
        labels_array = np.array(labels)
        return {
            'total': len(labels),
            'title': labels_array[:, 0].sum(),
            'datetime': labels_array[:, 1].sum(),
            'location': labels_array[:, 2].sum()
        }
    
    train_stats = get_label_stats(train_dataset)
    val_stats = get_label_stats(val_dataset)
    test_stats = get_label_stats(test_dataset)
    
    print("Data Split Analysis:")
    print("=" * 50)
    print(f"{'Split':<12} {'Total':<8} {'Title':<8} {'DateTime':<8} {'Location':<8}")
    print("-" * 50)
    print(f"{'Train':<12} {train_stats['total']:<8} {train_stats['title']:<8} {train_stats['datetime']:<8} {train_stats['location']:<8}")
    print(f"{'Validation':<12} {val_stats['total']:<8} {val_stats['title']:<8} {val_stats['datetime']:<8} {val_stats['location']:<8}")
    print(f"{'Test':<12} {test_stats['total']:<8} {test_stats['title']:<8} {test_stats['datetime']:<8} {test_stats['location']:<8}")
    
    return train_stats, val_stats, test_stats

print("Data preparation functions defined!")

# Cell 6: Load Your CSV Data
"""
Load your CSV data with columns: sentence, title, datetime, location
"""

def load_csv_data(csv_path):
    """
    Load your CSV data with multi-label format
    
    Expected CSV format:
    sentence, title, datetime, location
    "Meeting tomorrow", 0, 1, 0
    "Conference at hotel", 1, 0, 1
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV loaded successfully with shape: {df.shape}")
        
        # Display first few rows
        print("\nFirst 5 rows of your data:")
        print(df.head())
        
        # Check for required columns
        required_columns = ['sentence', 'title', 'datetime', 'location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\nWarning: Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            
            # Try to map common column names
            column_mapping = {
                'text': 'sentence',
                'sent': 'sentence',
                'content': 'sentence',
                'message': 'sentence',
                'date': 'datetime',
                'time': 'datetime',
                'date_time': 'datetime',
                'loc': 'location',
                'place': 'location',
                'venue': 'location'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name in missing_columns:
                    df = df.rename(columns={old_name: new_name})
                    print(f"Mapped '{old_name}' to '{new_name}'")
        
        # Extract texts and labels
        texts = df['sentence'].astype(str).tolist()
        
        # Create multi-label arrays
        labels = []
        for _, row in df.iterrows():
            label = [int(row['title']), int(row['datetime']), int(row['location'])]
            labels.append(label)
        
        return texts, labels, df
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        print("Please check the file path and try again.")
        return None, None, None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None, None

def create_sample_data():
    """Create sample data for demonstration if CSV not available"""
    sample_data = {
        'sentence': [
            "No event information here",
            "Just a regular sentence",
            "Annual Conference 2024",
            "Tech Summit happening soon",
            "Meeting scheduled for tomorrow at 3 PM",
            "Workshop on January 15th",
            "Event will be held at Convention Center",
            "Conference room A is booked",
            "AI Summit at Silicon Valley",
            "Workshop on January 15th at 2 PM",
            "Concert tomorrow evening at Madison Square Garden",
            "AI Conference 2024 on March 15th at Stanford University"
        ],
        'title': [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        'datetime': [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        'location': [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
    }
    
    df = pd.DataFrame(sample_data)
    texts = df['sentence'].tolist()
    labels = [[row['title'], row['datetime'], row['location']] for _, row in df.iterrows()]
    
    return texts, labels, df

# Load your data - UPDATE THIS PATH TO YOUR CSV FILE
CSV_PATH = "your_event_data.csv"  # Update this path!

print("Loading data...")
texts, labels, df = load_csv_data(CSV_PATH)

# If CSV loading fails, use sample data
if texts is None:
    print("Using sample data for demonstration...")
    texts, labels, df = create_sample_data()

print(f"\nLoaded {len(texts)} samples")

# Analyze label distribution
labels_array = np.array(labels)
print(f"\nLabel distribution:")
print(f"Title: {labels_array[:, 0].sum()} samples ({labels_array[:, 0].mean():.2%})")
print(f"DateTime: {labels_array[:, 1].sum()} samples ({labels_array[:, 1].mean():.2%})")
print(f"Location: {labels_array[:, 2].sum()} samples ({labels_array[:, 2].mean():.2%})")

# Show combination patterns
print(f"\nLabel combination patterns:")
unique_combinations = {}
for label in labels:
    key = tuple(label)
    unique_combinations[key] = unique_combinations.get(key, 0) + 1

for combo, count in sorted(unique_combinations.items()):
    title, datetime, location = combo
    print(f"Title={title}, DateTime={datetime}, Location={location}: {count} samples")

# Display sample texts for each combination
print(f"\nSample texts for each combination:")
for combo, count in sorted(unique_combinations.items()):
    if count > 0:
        title, datetime, location = combo
        matching_indices = [i for i, label in enumerate(labels) if tuple(label) == combo]
        sample_idx = matching_indices[0]
        print(f"[{title},{datetime},{location}]: \"{texts[sample_idx]}\"")

print(f"\nData loading completed successfully!")

# Cell 7: Prepare Datasets
"""
Split data into train, validation, and test sets
"""

train_dataset, val_dataset, test_dataset = prepare_data(texts, labels, tokenizer)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Analyze data split
train_stats, val_stats, test_stats = analyze_data_split(train_dataset, val_dataset, test_dataset)

# Check a sample
sample = train_dataset[0]
print(f"\nSample input shape: {sample['input_ids'].shape}")
print(f"Sample label: {sample['labels'].numpy()}")
print(f"Sample interpretation: Title={sample['labels'][0]}, DateTime={sample['labels'][1]}, Location={sample['labels'][2]}")

# Check if we have enough samples for each class
labels_array = np.array(labels)
min_samples = min(labels_array[:, 0].sum(), labels_array[:, 1].sum(), labels_array[:, 2].sum())
if min_samples < 10:
    print(f"\nWarning: Some classes have very few samples (minimum: {min_samples})")
    print("Consider collecting more data for better model performance.")

# Cell 8: Training Functions
"""
Training and validation functions
"""

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predicted = torch.sigmoid(outputs) > 0.5
        correct_predictions += (predicted == labels).all(dim=1).sum().item()
        total_predictions += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predicted == labels).all(dim=1).sum().item()
            total_predictions += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy

print("Training functions defined!")

# Cell 9: Training Configuration
"""
Set up training configuration and hyperparameters
"""

# Training hyperparameters
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 3

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Setup optimizer and scheduler
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Loss function
criterion = nn.BCEWithLogitsLoss()

print(f"Training configuration:")
print(f"- Device: {device}")
print(f"- Epochs: {EPOCHS}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Learning rate: {LEARNING_RATE}")
print(f"- Total steps: {total_steps}")
print(f"- Warmup steps: {int(0.1 * total_steps)}")

# Cell 10: Training Loop
"""
Main training loop with early stopping
"""

# Training tracking
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Early stopping
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print("Starting training...")
print("=" * 60)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)
    
    # Training
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, criterion, device
    )
    
    # Validation
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print("✓ New best model saved!")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("\n✓ Best model loaded!")

print("\nTraining completed!")

# Cell 11: Plot Training History
"""
Visualize training progress
"""

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy', marker='o')
    ax2.plot(val_accuracies, label='Validation Accuracy', marker='s')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

# Cell 12: Model Evaluation
"""
Evaluate model on test set with detailed analysis
"""

def evaluate_model(model, test_dataset, device):
    """Evaluate model on test set"""
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs)
            predictions = probabilities > 0.5
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Exact match accuracy
    exact_match = accuracy_score(all_labels, all_predictions)
    
    # Hamming loss
    hamming = hamming_loss(all_labels, all_predictions)
    
    # Per-class metrics
    class_names = ['Title', 'DateTime', 'Location']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Exact Match Accuracy: {exact_match:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Subset Accuracy: {exact_match:.4f}")
    
    # Calculate per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = accuracy_score(all_labels[:, i], all_predictions[:, i])
        print(f"{class_name}: {class_acc:.4f}")
    
    print("\nPer-class Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix for each class
    print("\nConfusion Analysis:")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        tp = np.sum((all_labels[:, i] == 1) & (all_predictions[:, i] == 1))
        fp = np.sum((all_labels[:, i] == 0) & (all_predictions[:, i] == 1))
        fn = np.sum((all_labels[:, i] == 1) & (all_predictions[:, i] == 0))
        tn = np.sum((all_labels[:, i] == 0) & (all_predictions[:, i] == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return exact_match, hamming, all_predictions, all_labels, all_probabilities

# Evaluate the model
exact_match, hamming, test_predictions, test_labels, test_probabilities = evaluate_model(model, test_dataset, device)

# Show some examples of predictions
print("\nSample Predictions:")
print("=" * 80)
test_texts_sample = [texts[i] for i in range(len(test_dataset))][:10]  # First 10 test samples
for i in range(min(10, len(test_predictions))):
    pred = test_predictions[i]
    true = test_labels[i]
    prob = test_probabilities[i]
    
    print(f"\nExample {i+1}:")
    print(f"Text: {test_texts_sample[i] if i < len(test_texts_sample) else 'N/A'}")
    print(f"True:  Title={true[0]}, DateTime={true[1]}, Location={true[2]}")
    print(f"Pred:  Title={pred[0]}, DateTime={pred[1]}, Location={pred[2]}")
    print(f"Prob:  Title={prob[0]:.3f}, DateTime={prob[1]:.3f}, Location={prob[2]:.3f}")
    
    # Check if prediction is correct
    correct = np.array_equal(pred, true)
    print(f"Match: {'✓' if correct else '✗'}")
    print("-" * 80)

# Cell 13: Prediction Function
"""
Function to make predictions on new texts
"""

def predict_events(model, tokenizer, texts, device):
    """Predict on new texts"""
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            pred = (probs > 0.5).cpu().numpy()[0]
            
            predictions.append(pred)
            probabilities.append(probs.cpu().numpy()[0])
    
    return np.array(predictions), np.array(probabilities)

# Test predictions on sample texts
test_texts = [
    "Machine Learning Workshop next Friday at the university",
    "Birthday party at my house tomorrow",
    "Conference call scheduled for 2 PM",
    "Just a regular sentence with no event details",
    "Annual Tech Summit 2024 on March 15th at Silicon Valley Convention Center"
]

predictions, probabilities = predict_events(model, tokenizer, test_texts, device)

print("Example Predictions:")
print("=" * 60)
for i, (text, pred, prob) in enumerate(zip(test_texts, predictions, probabilities)):
    print(f"\nText: {text}")
    print(f"Predictions: Title={pred[0]}, Date/Time={pred[1]}, Location={pred[2]}")
    print(f"Probabilities: Title={prob[0]:.3f}, Date/Time={prob[1]:.3f}, Location={prob[2]:.3f}")
    print("-" * 60)

# Cell 14: Save Model
"""
Save the trained model and tokenizer
"""

# Save model state dict
torch.save(model.state_dict(), 'event_classifier_model.pth')

# Save tokenizer
tokenizer.save_pretrained('event_classifier_tokenizer')

print("Model and tokenizer saved successfully!")
print("Files saved:")
print("- event_classifier_model.pth")
print("- event_classifier_tokenizer/")

# Cell 15: Model Loading (for future use)
"""
Code to load the saved model for future use
"""

def load_trained_model(model_path='event_classifier_model.pth', 
                      tokenizer_path='event_classifier_tokenizer',
                      original_model_path='./minilm'):
    """Load the trained model for inference"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Initialize model architecture
    model = MiniLMEventClassifier(model_path=original_model_path, num_labels=3)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, tokenizer

# Example of loading the model
# loaded_model, loaded_tokenizer = load_trained_model()
# print("Model loaded successfully for inference!")

print("\nTraining pipeline completed successfully!")
print("You can now use the trained model for event classification.")
