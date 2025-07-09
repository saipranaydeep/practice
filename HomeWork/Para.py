import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import spacy
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# Load spaCy model (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy English model not found. Please install it with:")
    print("python -m spacy download en_core_web_sm")
    exit(1)

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

def preprocess_article(article_text):
    """Clean and preprocess the article text"""
    # Remove extra whitespace and normalize
    article_text = re.sub(r'\s+', ' ', article_text.strip())

    # Remove special characters that might interfere with sentence splitting
    article_text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\"\'\n]', '', article_text)

    return article_text

def split_into_sentences(article_text, min_length=10, max_length=512):
    """Split article into sentences using spaCy with filtering"""
    # Preprocess the article
    article_text = preprocess_article(article_text)

    # Use spaCy for sentence segmentation
    doc = nlp(article_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Filter sentences
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Filter out very short or very long sentences
        if min_length <= len(sentence) <= max_length:
            # Remove sentences that are mostly numbers or special characters
            if re.search(r'[a-zA-Z]', sentence):
                filtered_sentences.append(sentence)

    return filtered_sentences

def advanced_sentence_splitting(article_text, min_length=10, max_length=512, use_entities=True):
    """Advanced sentence splitting with spaCy features"""
    # Preprocess the article
    article_text = preprocess_article(article_text)

    # Process with spaCy
    doc = nlp(article_text)

    sentences_info = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()

        # Skip if too short or too long
        if not (min_length <= len(sentence_text) <= max_length):
            continue

        # Skip if mostly numbers or special characters
        if not re.search(r'[a-zA-Z]', sentence_text):
            continue

        sentence_info = {
            'text': sentence_text,
            'start': sent.start,
            'end': sent.end
        }

        # Extract entities if requested
        if use_entities:
            entities = []
            for ent in sent.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
            sentence_info['entities'] = entities

        sentences_info.append(sentence_info)

    return sentences_info
    """Predict labels for multiple sentences"""
    dataset = InferenceDataset(sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_probabilities = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying sentences"):
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

def get_class_interpretation(class_num):
    """Get interpretation for class number"""
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
    return class_combinations.get(class_num, "Unknown class")

def analyze_article_advanced(article_text, model, tokenizer, device, use_entities=True):
    """Advanced pipeline to analyze an article with spaCy features"""
    print("Splitting article into sentences using spaCy...")
    sentences_info = advanced_sentence_splitting(article_text, use_entities=use_entities)

    print(f"Found {len(sentences_info)} sentences")

    if len(sentences_info) == 0:
        print("No valid sentences found in the article.")
        return None

    # Extract just the sentence texts for prediction
    sentences = [s['text'] for s in sentences_info]

    print("Classifying sentences...")
    predictions, probabilities = predict_sentences(sentences, model, tokenizer, device)

    # Convert to 8-class format
    class_numbers = convert_to_8_class(predictions)

    # Create results dataframe with enhanced information
    results_data = {
        'sentence_id': range(1, len(sentences) + 1),
        'sentence': sentences,
        'title': predictions[:, 0],
        'datetime': predictions[:, 1],
        'location': predictions[:, 2],
        'title_prob': probabilities[:, 0],
        'datetime_prob': probabilities[:, 1],
        'location_prob': probabilities[:, 2],
        'class_8': class_numbers,
        'interpretation': [get_class_interpretation(c) for c in class_numbers]
    }

    # Add entity information if available
    if use_entities:
        results_data['entities'] = [s.get('entities', []) for s in sentences_info]
        results_data['entity_count'] = [len(s.get('entities', [])) for s in sentences_info]

        # Extract specific entity types
        person_entities = []
        org_entities = []
        date_entities = []
        location_entities = []

        for s in sentences_info:
            entities = s.get('entities', [])
            persons = [e['text'] for e in entities if e['label'] in ['PERSON']]
            orgs = [e['text'] for e in entities if e['label'] in ['ORG']]
            dates = [e['text'] for e in entities if e['label'] in ['DATE', 'TIME']]
            locations = [e['text'] for e in entities if e['label'] in ['GPE', 'LOC']]

            person_entities.append(', '.join(persons))
            org_entities.append(', '.join(orgs))
            date_entities.append(', '.join(dates))
            location_entities.append(', '.join(locations))

        results_data['person_entities'] = person_entities
        results_data['org_entities'] = org_entities
        results_data['date_entities'] = date_entities
        results_data['location_entities'] = location_entities

    results_df = pd.DataFrame(results_data)
    return results_df

def generate_enhanced_summary_report(results_df):
    """Generate an enhanced summary report with spaCy entity information"""
    if results_df is None:
        return

    total_sentences = len(results_df)

    print("\n" + "="*80)
    print("ENHANCED ARTICLE ANALYSIS SUMMARY")
    print("="*80)

    print(f"Total sentences analyzed: {total_sentences}")

    # Class distribution
    print(f"\nClass Distribution:")
    class_counts = results_df['class_8'].value_counts().sort_index()
    for class_num, count in class_counts.items():
        percentage = (count / total_sentences) * 100
        print(f"  Class {class_num}: {count} sentences ({percentage:.1f}%) - {get_class_interpretation(class_num)}")

    # Feature distribution
    print(f"\nFeature Distribution:")
    title_count = results_df['title'].sum()
    datetime_count = results_df['datetime'].sum()
    location_count = results_df['location'].sum()

    print(f"  Sentences with titles: {title_count} ({title_count/total_sentences*100:.1f}%)")
    print(f"  Sentences with datetime: {datetime_count} ({datetime_count/total_sentences*100:.1f}%)")
    print(f"  Sentences with location: {location_count} ({location_count/total_sentences*100:.1f}%)")

    # Entity information if available
    if 'entity_count' in results_df.columns:
        print(f"\nEntity Analysis:")
        total_entities = results_df['entity_count'].sum()
        sentences_with_entities = (results_df['entity_count'] > 0).sum()
        print(f"  Total entities found: {total_entities}")
        print(f"  Sentences with entities: {sentences_with_entities} ({sentences_with_entities/total_sentences*100:.1f}%)")

        if 'person_entities' in results_df.columns:
            person_sentences = (results_df['person_entities'].str.len() > 0).sum()
            org_sentences = (results_df['org_entities'].str.len() > 0).sum()
            date_sentences = (results_df['date_entities'].str.len() > 0).sum()
            location_sentences = (results_df['location_entities'].str.len() > 0).sum()

            print(f"  Sentences with person entities: {person_sentences}")
            print(f"  Sentences with organization entities: {org_sentences}")
            print(f"  Sentences with date entities: {date_sentences}")
            print(f"  Sentences with location entities: {location_sentences}")

    # High confidence predictions
    print(f"\nHigh Confidence Predictions (>0.8):")
    high_conf_title = (results_df['title_prob'] > 0.8).sum()
    high_conf_datetime = (results_df['datetime_prob'] > 0.8).sum()
    high_conf_location = (results_df['location_prob'] > 0.8).sum()

    print(f"  High confidence titles: {high_conf_title}")
    print(f"  High confidence datetime: {high_conf_datetime}")
    print(f"  High confidence locations: {high_conf_location}")

def display_enhanced_sample_results(results_df, n_samples=5):
    """Display sample results with entity information"""
    if results_df is None:
        return

    print(f"\n" + "="*80)
    print(f"ENHANCED SAMPLE RESULTS (First {n_samples} sentences)")
    print("="*80)

    for i in range(min(n_samples, len(results_df))):
        row = results_df.iloc[i]
        print(f"\nSentence {row['sentence_id']}: {row['sentence']}")
        print(f"  Class: {row['class_8']} - {row['interpretation']}")
        print(f"  Probabilities: Title={row['title_prob']:.3f}, DateTime={row['datetime_prob']:.3f}, Location={row['location_prob']:.3f}")

        # Show entities if available
        if 'entities' in results_df.columns and len(row['entities']) > 0:
            print(f"  Entities found:")
            for entity in row['entities']:
                print(f"    - {entity['text']} ({entity['label']}: {entity['description']})")

        # Show extracted entities if available
        if 'person_entities' in results_df.columns:
            if row['person_entities']:
                print(f"  Persons: {row['person_entities']}")
            if row['org_entities']:
                print(f"  Organizations: {row['org_entities']}")
            if row['date_entities']:
                print(f"  Dates: {row['date_entities']}")
            if row['location_entities']:
                print(f"  Locations: {row['location_entities']}")

def filter_sentences_by_class(results_df, target_classes):
    """Filter sentences by specific classes"""
    if results_df is None:
        return None

    filtered_df = results_df[results_df['class_8'].isin(target_classes)]
    return filtered_df

def save_results(results_df, output_file='article_analysis_results.csv'):
    """Save results to CSV file"""
    if results_df is None:
        return

    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

def load_article_from_file(file_path):
    """Load article from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    """Main pipeline for article analysis"""
    # Load model and tokenizer
    model, device = load_trained_model('final_model.pth')
    if model is None:
        return

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    print("Model loaded successfully!")
    print(f"Using device: {device}")

    # Example 1: Analyze article from text
    print("\n" + "="*80)
    print("EXAMPLE 1: Analyze Sample Article")
    print("="*80)

    sample_article = """
    The Annual Tech Conference 2024 will be held on March 15th at the downtown Convention Center.
    This year's theme focuses on artificial intelligence and machine learning. The event will feature
    keynote speakers from major tech companies. Registration is open until February 28th.

    The conference will take place from 9:00 AM to 5:00 PM. Lunch will be served at the main hall.
    Attendees can network during coffee breaks scheduled throughout the day. The event is expected
    to attract over 500 participants from around the world.

    For more information, visit our website or contact the organizing committee. The venue is easily
    accessible by public transportation. Parking is available at the convention center for a fee.
    """

    # Analyze the article using advanced spaCy features
    results_df = analyze_article_advanced(sample_article, model, tokenizer, device, use_entities=True)

    # Generate enhanced summary report
    generate_enhanced_summary_report(results_df)

    # Display enhanced sample results
    display_enhanced_sample_results(results_df)

    # Save results
    save_results(results_df, 'sample_article_results.csv')

    # Example 2: Filter sentences by class
    print("\n" + "="*80)
    print("EXAMPLE 2: Filter Sentences by Class")
    print("="*80)

    # Show only sentences with events (classes 4,5,6,7 - has title)
    event_sentences = filter_sentences_by_class(results_df, [4, 5, 6, 7])
    print(f"Found {len(event_sentences)} sentences with event titles:")
    for _, row in event_sentences.iterrows():
        print(f"  - {row['sentence']}")

    # Example 3: Load article from file (uncomment to use)
    print("\n" + "="*80)
    print("EXAMPLE 3: Load Article from File")
    print("="*80)

    """
    # To use this, create a text file named 'article.txt' with your article content
    article_text = load_article_from_file('article.txt')
    if article_text:
        results_df = analyze_article_advanced(article_text, model, tokenizer, device, use_entities=True)
        generate_enhanced_summary_report(results_df)
        save_results(results_df, 'file_article_results.csv')
    """

    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Paste your article text below (press Enter twice to finish):")

    article_lines = []
    while True:
        try:
            line = input()
            if line == "" and len(article_lines) > 0:
                break
            article_lines.append(line)
        except KeyboardInterrupt:
            break

    if article_lines:
        user_article = "\n".join(article_lines)
        print(f"\nAnalyzing article with {len(user_article)} characters...")

        # Choose analysis method
        use_entities = input("Use advanced entity extraction? (y/n): ").lower() == 'y'

        if use_entities:
            results_df = analyze_article_advanced(user_article, model, tokenizer, device, use_entities=True)
            generate_enhanced_summary_report(results_df)
            display_enhanced_sample_results(results_df, n_samples=3)
        else:
            # Use basic analysis
            sentences = split_into_sentences(user_article)
            predictions, probabilities = predict_sentences(sentences, model, tokenizer, device)
            class_numbers = convert_to_8_class(predictions)

            results_df = pd.DataFrame({
                'sentence_id': range(1, len(sentences) + 1),
                'sentence': sentences,
                'title': predictions[:, 0],
                'datetime': predictions[:, 1],
                'location': predictions[:, 2],
                'title_prob': probabilities[:, 0],
                'datetime_prob': probabilities[:, 1],
                'location_prob': probabilities[:, 2],
                'class_8': class_numbers,
                'interpretation': [get_class_interpretation(c) for c in class_numbers]
            })

            generate_enhanced_summary_report(results_df)
            display_enhanced_sample_results(results_df, n_samples=3)

        # Ask if user wants to save results
        save_choice = input("\nSave results to CSV? (y/n): ").lower()
        if save_choice == 'y':
            filename = input("Enter filename (default: user_article_results.csv): ").strip()
            if not filename:
                filename = 'user_article_results.csv'
            save_results(results_df, filename)

if __name__ == "__main__":
    main()
