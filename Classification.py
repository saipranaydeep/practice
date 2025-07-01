import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class EventSentenceClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load spacy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Spacy model not found. NER features will be limited.")
            self.nlp = None
        
        # Define class labels
        self.classes = ['title', 'datetime', 'location', 'combination', 'not_useful']
        
        # Define patterns and keywords
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 12/31/2023
            r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}\b',  # 15 January 2023
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',  # Weekdays
            r'\b(today|tomorrow|yesterday)\b',  # Relative dates
        ]
        
        self.time_patterns = [
            r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b',  # 2:30 PM
            r'\b\d{1,2}\s*(am|pm|AM|PM)\b',  # 2 PM
            r'\b(morning|afternoon|evening|night)\b',  # Time periods
        ]
        
        self.location_keywords = [
            'at', 'venue', 'location', 'address', 'street', 'avenue', 'road', 'building',
            'hall', 'center', 'centre', 'room', 'floor', 'hotel', 'restaurant', 'park',
            'stadium', 'theater', 'theatre', 'auditorium', 'campus', 'university', 'school'
        ]
        
        self.title_indicators = [
            'presents', 'announces', 'invites', 'welcome', 'join', 'attend', 'celebration',
            'conference', 'workshop', 'seminar', 'meeting', 'event', 'party', 'festival',
            'concert', 'show', 'exhibition', 'launch', 'opening', 'ceremony'
        ]
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-:/.@]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(tokens)
        except:
            return text
    
    def extract_features(self, sentences):
        """Extract various features from sentences"""
        features = []
        
        for sentence in sentences:
            sentence_features = {}
            
            # Basic text features
            sentence_features['length'] = len(sentence)
            sentence_features['word_count'] = len(sentence.split())
            sentence_features['has_numbers'] = bool(re.search(r'\d', sentence))
            sentence_features['has_email'] = bool(re.search(r'\S+@\S+', sentence))
            sentence_features['has_url'] = bool(re.search(r'http[s]?://|www\.', sentence))
            
            # Date and time features
            date_matches = sum(1 for pattern in self.date_patterns if re.search(pattern, sentence, re.IGNORECASE))
            time_matches = sum(1 for pattern in self.time_patterns if re.search(pattern, sentence, re.IGNORECASE))
            
            sentence_features['date_patterns'] = date_matches
            sentence_features['time_patterns'] = time_matches
            sentence_features['datetime_score'] = date_matches + time_matches
            
            # Location features
            location_score = sum(1 for keyword in self.location_keywords if keyword in sentence.lower())
            sentence_features['location_score'] = location_score
            
            # Title features
            title_score = sum(1 for keyword in self.title_indicators if keyword in sentence.lower())
            sentence_features['title_score'] = title_score
            
            # NER features if available
            if self.nlp:
                doc = self.nlp(sentence)
                sentence_features['person_entities'] = len([ent for ent in doc.ents if ent.label_ == 'PERSON'])
                sentence_features['org_entities'] = len([ent for ent in doc.ents if ent.label_ == 'ORG'])
                sentence_features['gpe_entities'] = len([ent for ent in doc.ents if ent.label_ == 'GPE'])
                sentence_features['date_entities'] = len([ent for ent in doc.ents if ent.label_ == 'DATE'])
                sentence_features['time_entities'] = len([ent for ent in doc.ents if ent.label_ == 'TIME'])
            else:
                sentence_features.update({
                    'person_entities': 0, 'org_entities': 0, 'gpe_entities': 0,
                    'date_entities': 0, 'time_entities': 0
                })
            
            # Combination features
            feature_count = sum([
                sentence_features['datetime_score'] > 0,
                sentence_features['location_score'] > 0,
                sentence_features['title_score'] > 0
            ])
            sentence_features['is_combination'] = feature_count >= 2
            
            features.append(sentence_features)
        
        return pd.DataFrame(features)
    
    def prepare_data(self, sentences, labels):
        """Prepare data for training"""
        # Preprocess text
        processed_sentences = [self.preprocess_text(s) for s in sentences]
        
        # Extract TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(processed_sentences)
        
        # Extract additional features
        additional_features = self.extract_features(sentences)
        
        # Combine features
        combined_features = np.hstack([
            tfidf_features.toarray(),
            additional_features.values
        ])
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        return combined_features, encoded_labels
    
    def train(self, sentences, labels):
        """Train the classifier"""
        print("Preparing data...")
        X, y = self.prepare_data(sentences, labels)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        # Print results
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        print("\nFeature Importance (Top 10):")
        feature_names = (list(self.vectorizer.get_feature_names_out()) + 
                        list(self.extract_features(['dummy']).columns))
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for i in range(10):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return self.model
    
    def predict(self, sentences):
        """Predict classes for new sentences"""
        processed_sentences = [self.preprocess_text(s) for s in sentences]
        tfidf_features = self.vectorizer.transform(processed_sentences)
        additional_features = self.extract_features(sentences)
        
        combined_features = np.hstack([
            tfidf_features.toarray(),
            additional_features.values
        ])
        
        predictions = self.model.predict(combined_features)
        probabilities = self.model.predict_proba(combined_features)
        
        return self.label_encoder.inverse_transform(predictions), probabilities
    
    def predict_single(self, sentence):
        """Predict class for a single sentence"""
        predictions, probabilities = self.predict([sentence])
        
        result = {
            'sentence': sentence,
            'predicted_class': predictions[0],
            'confidence': max(probabilities[0])
        }
        
        # Show all class probabilities
        for i, class_name in enumerate(self.label_encoder.classes_):
            result[f'{class_name}_probability'] = probabilities[0][i]
        
        return result

# Example usage and demo
def create_sample_data():
    """Create sample training data"""
    sentences = [
        # Title examples
        "Annual Tech Conference 2024",
        "Join us for the Grand Opening Ceremony",
        "You're invited to Sarah's Birthday Party",
        "Digital Marketing Workshop - Advanced Strategies",
        
        # DateTime examples
        "The event will be held on March 15, 2024",
        "Join us at 7:00 PM on Friday",
        "Registration starts at 9:00 AM",
        "Save the date: December 25th",
        
        # Location examples
        "Venue: Grand Ballroom, Marriott Hotel",
        "Address: 123 Main Street, Downtown",
        "Location: Conference Room A, 5th Floor",
        "At Central Park Amphitheater",
        
        # Combination examples
        "March 20th at 6 PM - Downtown Convention Center",
        "Saturday, 2 PM at the Community Hall on Oak Street",
        "Annual Gala Dinner - December 31st, 8 PM at Royal Hotel",
        
        # Not useful examples
        "Thank you for your attention",
        "Please bring your own materials",
        "Refreshments will be provided",
        "Contact us for more information"
    ]
    
    labels = [
        'title', 'title', 'title', 'title',
        'datetime', 'datetime', 'datetime', 'datetime',
        'location', 'location', 'location', 'location',
        'combination', 'combination', 'combination',
        'not_useful', 'not_useful', 'not_useful', 'not_useful'
    ]
    
    return sentences, labels

# Demo
if __name__ == "__main__":
    print("Event Sentence Classifier Demo")
    print("=" * 40)
    
    # Create classifier
    classifier = EventSentenceClassifier()
    
    # Create sample data
    sentences, labels = create_sample_data()
    
    # Train the model
    classifier.train(sentences, labels)
    
    # Test predictions
    test_sentences = [
        "AI Summit 2024 - Transforming the Future",
        "Event Date: January 20, 2025 at 3:00 PM",
        "Venue: Tech Hub, Silicon Valley",
        "Join us on Friday, 7 PM at the Innovation Center",
        "Please RSVP by tomorrow"
    ]
    
    print("\n" + "="*40)
    print("PREDICTIONS:")
    print("="*40)
    
    for sentence in test_sentences:
        result = classifier.predict_single(sentence)
        print(f"\nSentence: {sentence}")
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
