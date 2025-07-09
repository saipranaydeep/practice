import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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

class InferenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])

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
            'sentence': sentence
        }

def load_trained_model(model_path='final_model.pth', model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Load the trained model from saved state dict"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model architecture
    model = MultiLabelMiniLM(model_name=model_name, num_labels=3, dropout_rate=0.3)

    # Load saved weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model, device
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please make sure the model is trained and saved.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_single_sentence(sentence, model, tokenizer, device):
    """Predict labels for a single sentence"""
    model.eval()

    # Tokenize the sentence
    encoding = tokenizer(
        sentence,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        predictions = (probabilities > 0.5).astype(int)

    return predictions, probabilities

def predict_batch(sentences, model, tokenizer, device, batch_size=16):
    """Predict labels for multiple sentences"""
    dataset = InferenceDataset(sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_probabilities = []

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

    return np.array(all_predictions), np.array(all_probabilities)

def convert_to_8_class(predictions):
    """Convert binary predictions to 8-class format (0-7)"""
    class_numbers = []
    for pred in predictions:
        # Convert binary array to decimal: title*4 + datetime*2 + location*1
        class_num = pred[0] * 4 + pred[1] * 2 + pred[2] * 1
        class_numbers.append(class_num)
    return class_numbers

def interpret_predictions(predictions, probabilities):
    """Interpret predictions with class names"""
    class_names = ['title', 'datetime', 'location']
    class_combinations = {
        0: "(0,0,0) - No title, No datetime, No location",
        1: "(0,0,1) - No title, No datetime, Has location",
        2: "(0,1,0) - No title, Has datetime, No location",
        3: "(0,1,1) - No title, Has datetime, Has location",
        4: "(1,0,0) - Has title, No datetime, No location",
        5: "(1,0,1) - Has title, No datetime, Has location",
        6: "(1,1,0) - Has title, Has datetime, No location",
        7: "(1,1,1) - Has title, Has datetime, Has location"
    }

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        class_8 = convert_to_8_class([pred])[0]

        result = {
            'prediction': pred.tolist(),
            'probabilities': prob.tolist(),
            'class_8': class_8,
            'interpretation': class_combinations[class_8],
            'confidence': {
                'title': prob[0],
                'datetime': prob[1],
                'location': prob[2]
            }
        }
        results.append(result)

    return results

def main():
    """Main inference pipeline"""
    # Load model and tokenizer
    model, device = load_trained_model('final_model.pth')
    if model is None:
        return

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    print("Model loaded successfully!")
    print(f"Using device: {device}")

    # Example 1: Single sentence prediction
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Sentence Prediction")
    print("="*60)

    test_sentence = "Join us for the Annual Conference on March 15th at the downtown convention center."
    predictions, probabilities = predict_single_sentence(test_sentence, model, tokenizer, device)

    print(f"Sentence: {test_sentence}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")

    # Interpret the result
    results = interpret_predictions([predictions], [probabilities])
    result = results[0]

    print(f"\nInterpretation:")
    print(f"- 8-Class: {result['class_8']} - {result['interpretation']}")
    print(f"- Confidence Scores:")
    for label, confidence in result['confidence'].items():
        print(f"  * {label.capitalize()}: {confidence:.3f}")

    # Example 2: Batch prediction
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Prediction")
    print("="*60)

    test_sentences = [
        "The weather is beautiful today.",
        "Don't forget the meeting tomorrow at 2 PM.",
        "Visit our new branch in downtown Seattle.",
        "Summer Festival starts on July 20th at Central Park.",
        "Check out the latest product updates."
    ]

    batch_predictions, batch_probabilities = predict_batch(test_sentences, model, tokenizer, device)
    batch_results = interpret_predictions(batch_predictions, batch_probabilities)

    for i, (sentence, result) in enumerate(zip(test_sentences, batch_results)):
        print(f"\nSentence {i+1}: {sentence}")
        print(f"Class: {result['class_8']} - {result['interpretation']}")
        print(f"Confidences: T={result['confidence']['title']:.3f}, "
              f"D={result['confidence']['datetime']:.3f}, "
              f"L={result['confidence']['location']:.3f}")

    # Example 3: Load and predict from CSV
    print("\n" + "="*60)
    print("EXAMPLE 3: Predict from CSV (Optional)")
    print("="*60)

    # Uncomment below to predict on a new CSV file
    """
    try:
        # Load new data
        df = pd.read_csv('new_data.csv')  # Should have 'sentence' column
        sentences = df['sentence'].tolist()

        # Make predictions
        predictions, probabilities = predict_batch(sentences, model, tokenizer, device)

        # Add predictions to dataframe
        df['pred_title'] = predictions[:, 0]
        df['pred_datetime'] = predictions[:, 1]
        df['pred_location'] = predictions[:, 2]
        df['prob_title'] = probabilities[:, 0]
        df['prob_datetime'] = probabilities[:, 1]
        df['prob_location'] = probabilities[:, 2]
        df['class_8'] = convert_to_8_class(predictions)

        # Save results
        df.to_csv('predictions.csv', index=False)
        print("Predictions saved to 'predictions.csv'")

    except FileNotFoundError:
        print("No 'new_data.csv' found. Skipping CSV prediction example.")
    """

    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter sentences to classify (type 'quit' to exit):")

    while True:
        user_input = input("\nEnter sentence: ").strip()
        if user_input.lower() == 'quit':
            break

        if user_input:
            predictions, probabilities = predict_single_sentence(user_input, model, tokenizer, device)
            results = interpret_predictions([predictions], [probabilities])
            result = results[0]

            print(f"Result: Class {result['class_8']} - {result['interpretation']}")
            print(f"Confidences: Title={result['confidence']['title']:.3f}, "
                  f"DateTime={result['confidence']['datetime']:.3f}, "
                  f"Location={result['confidence']['location']:.3f}")

if __name__ == "__main__":
    main()
