import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AdamW, 
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback
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
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

class MiniLMEventClassifier(nn.Module):
    """MiniLM-based Multi-label Event Classifier"""
    
    def __init__(self, model_name='microsoft/MiniLM-L12-H384-uncased', num_labels=3, dropout_rate=0.3):
        super(MiniLMEventClassifier, self).__init__()
        self.num_labels = num_labels
        
        # Load pre-trained MiniLM
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers to prevent overfitting
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Only train last 4 layers
        for layer in self.bert.encoder.layer[:-4]:
            for param in layer.parameters():
                param.requires_grad = False
        
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

class EventClassificationTrainer:
    """Trainer class for Event Classification"""
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def prepare_data(self, texts, labels, test_size=0.2, val_size=0.1):
        """Prepare train, validation, and test datasets"""
        
        # Convert class labels to multi-label format
        multi_labels = []
        for label in labels:
            if label == 0:  # 0,0,0
                multi_labels.append([0, 0, 0])
            elif label == 1:  # 1,0,0
                multi_labels.append([1, 0, 0])
            elif label == 2:  # 0,1,0
                multi_labels.append([0, 1, 0])
            elif label == 3:  # 0,0,1
                multi_labels.append([0, 0, 1])
            elif label == 4:  # 1,0,1
                multi_labels.append([1, 0, 1])
            elif label == 5:  # 1,1,0
                multi_labels.append([1, 1, 0])
            elif label == 6:  # 0,1,1
                multi_labels.append([0, 1, 1])
            elif label == 7:  # 1,1,1
                multi_labels.append([1, 1, 1])
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, multi_labels, test_size=test_size + val_size, random_state=42, stratify=labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42
        )
        
        # Create datasets
        train_dataset = EventDataset(X_train, y_train, self.tokenizer)
        val_dataset = EventDataset(X_val, y_val, self.tokenizer)
        test_dataset = EventDataset(X_test, y_test, self.tokenizer)
        
        return train_dataset, val_dataset, test_dataset
    
    def train_epoch(self, dataloader, optimizer, scheduler, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = torch.sigmoid(outputs) > 0.5
                correct_predictions += (predicted == labels).all(dim=1).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_dataset, val_dataset, epochs=10, batch_size=16, learning_rate=2e-5, 
              patience=3, weight_decay=0.01):
        """Main training loop with early stopping"""
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function with class weights for handling imbalanced data
        criterion = nn.BCEWithLogitsLoss()
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Training on {self.device}")
        print(f"Total training steps: {total_steps}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler, criterion)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print("New best model saved!")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Best model loaded!")
    
    def evaluate(self, test_dataset):
        """Evaluate model on test set"""
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.sigmoid(outputs) > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Exact match accuracy
        exact_match = accuracy_score(all_labels, all_predictions)
        
        # Hamming loss
        hamming = hamming_loss(all_labels, all_predictions)
        
        # Per-class metrics
        class_names = ['Title', 'Date/Time', 'Location']
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Exact Match Accuracy: {exact_match:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
        print("\nPer-class Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
        
        return exact_match, hamming
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Validation Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy', marker='o')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', marker='s')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, texts):
        """Predict on new texts"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs)
                pred = (probs > 0.5).cpu().numpy()[0]
                
                predictions.append(pred)
        
        return np.array(predictions)

# Example usage and data preparation
def create_sample_data():
    """Create sample data for demonstration"""
    # This is just example data - replace with your actual dataset
    sample_texts = [
        "No event information here",
        "Annual Conference 2024",
        "Meeting scheduled for tomorrow at 3 PM",
        "Event will be held at Convention Center",
        "Tech Summit at Silicon Valley",
        "Workshop on January 15th at 2 PM",
        "Concert tomorrow evening at Madison Square Garden",
        "AI Conference 2024 on March 15th at Stanford University"
    ]
    
    sample_labels = [0, 1, 2, 3, 4, 5, 6, 7]  # Corresponding to your 8 classes
    
    return sample_texts, sample_labels

def main():
    """Main function to run the training pipeline"""
    
    # Load your data here
    # texts, labels = load_your_data()  # Replace with your data loading function
    texts, labels = create_sample_data()  # Using sample data for demonstration
    
    # Initialize tokenizer and model
    model_name = 'microsoft/MiniLM-L12-H384-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MiniLMEventClassifier(model_name=model_name, num_labels=3, dropout_rate=0.3)
    
    # Initialize trainer
    trainer = EventClassificationTrainer(model, tokenizer)
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(texts, labels)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train the model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=15,
        batch_size=16,
        learning_rate=2e-5,
        patience=3,
        weight_decay=0.01
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    trainer.evaluate(test_dataset)
    
    # Save the model
    torch.save(model.state_dict(), 'event_classifier_model.pth')
    tokenizer.save_pretrained('event_classifier_tokenizer')
    
    print("\nModel saved successfully!")
    
    # Example prediction
    test_texts = [
        "Machine Learning Workshop next Friday",
        "Birthday party at my house",
        "Conference call at 2 PM"
    ]
    
    predictions = trainer.predict(test_texts)
    
    print("\nExample predictions:")
    for text, pred in zip(test_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: Title={pred[0]}, Date/Time={pred[1]}, Location={pred[2]}")
        print("-" * 50)

if __name__ == "__main__":
    main()
