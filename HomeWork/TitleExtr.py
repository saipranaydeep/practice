import re
import json
from typing import List, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedEventTitleExtractor:
    def __init__(self):
        self.transformers_available = False
        self.openai_available = False
        self.spacy_available = False
        
        # Try to import libraries
        try:
            import transformers
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            self.transformers_available = True
            self.transformers = transformers
            print("✓ Transformers available")
        except ImportError:
            print("✗ Transformers not available. Install with: pip install transformers torch")
        
        try:
            import openai
            self.openai = openai
            self.openai_available = True
            print("✓ OpenAI available")
        except ImportError:
            print("✗ OpenAI not available. Install with: pip install openai")
        
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
            print("✓ SpaCy available")
        except:
            print("✗ SpaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    
    def extract_with_ner_model(self, sentence: str) -> str:
        """Extract title using Named Entity Recognition models"""
        if not self.transformers_available:
            return ""
        
        try:
            # Use a pre-trained NER model
            ner_pipeline = self.transformers.pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            entities = ner_pipeline(sentence)
            
            # Look for entities that could be event titles
            candidates = []
            for entity in entities:
                if entity['entity_group'] in ['MISC', 'ORG'] and entity['score'] > 0.8:
                    candidates.append(entity['word'])
            
            # Return the longest candidate
            if candidates:
                return max(candidates, key=len)
            
        except Exception as e:
            print(f"NER model error: {e}")
        
        return ""
    
    def extract_with_qa_model(self, sentence: str) -> str:
        """Extract title using Question-Answering model"""
        if not self.transformers_available:
            return ""
        
        try:
            qa_pipeline = self.transformers.pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
            
            questions = [
                "What is the event title?",
                "What is the name of the event?",
                "What event is being described?",
                "What is the title of this event?"
            ]
            
            best_answer = ""
            best_score = 0
            
            for question in questions:
                try:
                    result = qa_pipeline(question=question, context=sentence)
                    if result['score'] > best_score and len(result['answer']) > 3:
                        best_answer = result['answer']
                        best_score = result['score']
                except:
                    continue
            
            return best_answer if best_score > 0.3 else ""
            
        except Exception as e:
            print(f"QA model error: {e}")
        
        return ""
    
    def extract_with_classification_model(self, sentence: str) -> str:
        """Extract title using text classification and span detection"""
        if not self.transformers_available:
            return ""
        
        try:
            # Use a model fine-tuned for token classification
            tokenizer = self.transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Tokenize and look for title patterns
            tokens = tokenizer.tokenize(sentence)
            
            # Simple heuristic: look for capitalized consecutive tokens
            title_tokens = []
            in_title = False
            
            for i, token in enumerate(tokens):
                if token.startswith('##'):
                    if in_title:
                        title_tokens.append(token[2:])
                    continue
                
                if token[0].isupper() and len(token) > 2:
                    if not in_title:
                        title_tokens = [token]
                        in_title = True
                    else:
                        title_tokens.append(token)
                elif in_title and token.lower() in ['and', 'of', 'the', 'for', 'in', 'on', 'at']:
                    title_tokens.append(token)
                elif in_title:
                    break
            
            if len(title_tokens) >= 2:
                return ' '.join(title_tokens)
            
        except Exception as e:
            print(f"Classification model error: {e}")
        
        return ""
    
    def extract_with_llm(self, sentence: str, api_key: Optional[str] = None) -> str:
        """Extract title using OpenAI's GPT models"""
        if not self.openai_available or not api_key:
            return ""
        
        try:
            client = self.openai.OpenAI(api_key=api_key)
            
            prompt = f"""
            Extract the event title from the following sentence. Return only the title, nothing else.
            If there's no clear event title, return "NONE".
            
            Sentence: "{sentence}"
            
            Event Title:"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            title = response.choices[0].message.content.strip()
            return title if title != "NONE" else ""
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
        
        return ""
    
    def extract_with_local_llm(self, sentence: str) -> str:
        """Extract title using a local LLM via transformers"""
        if not self.transformers_available:
            return ""
        
        try:
            # Use a smaller, local model for text generation
            generator = self.transformers.pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                max_length=100,
                num_return_sequences=1,
                temperature=0.3
            )
            
            prompt = f"Extract the event title from: {sentence}\nEvent title:"
            
            result = generator(prompt, max_new_tokens=20, do_sample=True, pad_token_id=50256)
            generated_text = result[0]['generated_text']
            
            # Extract the part after "Event title:"
            if "Event title:" in generated_text:
                title = generated_text.split("Event title:")[-1].strip()
                # Clean up the response
                title = re.sub(r'[^\w\s\-&]', '', title).strip()
                return title if len(title) > 3 else ""
            
        except Exception as e:
            print(f"Local LLM error: {e}")
        
        return ""
    
    def extract_with_ensemble(self, sentence: str, openai_api_key: Optional[str] = None) -> Dict[str, str]:
        """Use multiple models and combine results"""
        results = {}
        
        # Try different methods
        methods = [
            ("ner", self.extract_with_ner_model),
            ("qa", self.extract_with_qa_model),
            ("classification", self.extract_with_classification_model),
            ("llm", lambda s: self.extract_with_llm(s, openai_api_key)),
            ("local_llm", self.extract_with_local_llm),
            ("regex", self.extract_title_regex)  # Fallback
        ]
        
        for method_name, method_func in methods:
            try:
                result = method_func(sentence)
                if result:
                    results[method_name] = result
            except Exception as e:
                print(f"Error with {method_name}: {e}")
        
        return results
    
    def extract_title_regex(self, sentence: str) -> str:
        """Fallback regex method"""
        patterns = [
            r'Event:\s*(.+?)(?:\s*-|\s*\||\s*at|\s*on|$)',
            r'^(.+?)\s*-\s*(?:on|at|\d{1,2}[\/\-]\d{1,2}|\w+\s+\d{1,2})',
            r'(?:Join us for|Attend|Welcome to|Come to)\s+(.+?)(?:\s*-|\s*\||\s*at|\s*on|$)',
            r'"([^"]+)"',
            r"'([^']+)'",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'\s+', ' ', title)
                title = title.strip('.,!?;:')
                if len(title) > 3:
                    return title
        return ""
    
    def get_best_title(self, sentence: str, openai_api_key: Optional[str] = None) -> str:
        """Get the best title using ensemble approach"""
        results = self.extract_with_ensemble(sentence, openai_api_key)
        
        if not results:
            return ""
        
        # Priority order for methods
        priority = ["llm", "qa", "ner", "classification", "local_llm", "regex"]
        
        for method in priority:
            if method in results and results[method]:
                return results[method]
        
        # If no prioritized method worked, return any result
        return list(results.values())[0] if results else ""
    
    def extract_titles_batch(self, sentences: List[str], method: str = "ensemble", 
                           openai_api_key: Optional[str] = None) -> List[Dict]:
        """Extract titles from multiple sentences"""
        results = []
        
        for i, sentence in enumerate(sentences):
            if method == "ensemble":
                title = self.get_best_title(sentence, openai_api_key)
                ensemble_results = self.extract_with_ensemble(sentence, openai_api_key)
            elif method == "ner":
                title = self.extract_with_ner_model(sentence)
                ensemble_results = {}
            elif method == "qa":
                title = self.extract_with_qa_model(sentence)
                ensemble_results = {}
            elif method == "llm":
                title = self.extract_with_llm(sentence, openai_api_key)
                ensemble_results = {}
            else:
                title = self.extract_title_regex(sentence)
                ensemble_results = {}
            
            results.append({
                'index': i,
                'sentence': sentence,
                'title': title,
                'all_results': ensemble_results
            })
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = AdvancedEventTitleExtractor()
    
    # Example sentences
    example_sentences = [
        "Join us for the Annual Technology Conference 2024 on March 15th at the Convention Center.",
        "Event: Spring Music Festival - Live performances by local artists",
        "The 'Digital Marketing Workshop' will be held next Tuesday from 2-4 PM.",
        "ANNUAL CHARITY GALA happening this Saturday at 7 PM",
        "Python Programming Bootcamp starts Monday - Register now!",
        "We're excited to announce our Summer Concert Series beginning June 1st.",
        "The quarterly board meeting is scheduled for tomorrow at 10 AM.",
        "Don't miss the International Food Festival this weekend in downtown."
    ]
    
    print("Advanced Event Title Extraction Results:")
    print("=" * 60)
    
    # Test with different methods
    for sentence in example_sentences[:3]:  # Test first 3 sentences
        print(f"\nSentence: {sentence}")
        print("-" * 40)
        
        # Try different methods
        if extractor.transformers_available:
            ner_result = extractor.extract_with_ner_model(sentence)
            qa_result = extractor.extract_with_qa_model(sentence)
            print(f"NER Model: {ner_result or 'No title found'}")
            print(f"QA Model: {qa_result or 'No title found'}")
        
        # Ensemble approach
        ensemble_results = extractor.extract_with_ensemble(sentence)
        print(f"Ensemble Results: {ensemble_results}")
        
        best_title = extractor.get_best_title(sentence)
        print(f"Best Title: {best_title or 'No title found'}")
        print("=" * 60)
    
    # Batch processing example
    print("\nBatch Processing with Ensemble:")
    batch_results = extractor.extract_titles_batch(example_sentences, method="ensemble")
    for result in batch_results:
        if result['title']:
            print(f"Title: {result['title']}")
            print(f"All methods: {result['all_results']}")
            print("-" * 30)
