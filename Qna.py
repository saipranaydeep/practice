import re
import json
import torch
from typing import List, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

class LocalDistilBERTEventExtractor:
    def __init__(self, model_path: str = "./distilbert-base-cased-distilled-squad"):
        """
        Initialize the extractor with local DistilBERT model
        
        Args:
            model_path: Path to the local model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.qa_pipeline = None
        
        self.load_model()
    
    def load_model(self):
        """Load the local DistilBERT model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
            
            print(f"Loading model from: {self.model_path}")
            
            # Load tokenizer and model from local path
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
            
            # Create pipeline with local model
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            
            print("✓ Local DistilBERT model loaded successfully")
            print(f"✓ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Make sure you have the model files in the correct directory structure:")
            print("- config.json")
            print("- pytorch_model.bin (or model.safetensors)")
            print("- tokenizer.json")
            print("- tokenizer_config.json")
            print("- vocab.txt")
    
    def extract_title_qa(self, sentence: str, confidence_threshold: float = 0.1) -> Dict[str, Union[str, float]]:
        """
        Extract event title using Question-Answering approach
        
        Args:
            sentence: Input sentence
            confidence_threshold: Minimum confidence score
            
        Returns:
            Dictionary with title and confidence score
        """
        if not self.qa_pipeline:
            return {"title": "", "confidence": 0.0, "method": "qa"}
        
        # List of questions to ask about the event title
        questions = [
            "What is the event title?",
            "What is the name of the event?",
            "What event is being described?",
            "What is the title of this event?",
            "What is the event called?",
            "What is the name of this activity?",
            "What is the main event?",
            "What is the event name mentioned?"
        ]
        
        best_answer = ""
        best_score = 0.0
        best_question = ""
        
        for question in questions:
            try:
                result = self.qa_pipeline(
                    question=question, 
                    context=sentence,
                    max_answer_len=50,
                    handle_impossible_answer=True
                )
                
                answer = result['answer'].strip()
                score = result['score']
                
                # Filter out very short or invalid answers
                if (score > best_score and 
                    len(answer) > 3 and 
                    score > confidence_threshold and
                    not answer.lower() in ['the', 'this', 'that', 'it', 'event', 'activity']):
                    
                    best_answer = answer
                    best_score = score
                    best_question = question
                    
            except Exception as e:
                print(f"Error with question '{question}': {e}")
                continue
        
        # Clean up the answer
        if best_answer:
            best_answer = self._clean_title(best_answer)
        
        return {
            "title": best_answer,
            "confidence": best_score,
            "method": "qa",
            "question_used": best_question
        }
    
    def extract_title_span_detection(self, sentence: str) -> Dict[str, Union[str, float]]:
        """
        Extract title using span detection approach
        Find the most likely title span in the sentence
        """
        if not self.tokenizer or not self.model:
            return {"title": "", "confidence": 0.0, "method": "span"}
        
        try:
            # Tokenize the sentence
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            
            # Use the model to find answer spans for a generic question
            question = "What is the main event or activity mentioned?"
            qa_inputs = self.tokenizer(
                question, 
                sentence, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**qa_inputs)
                
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Find the best span
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            if start_idx <= end_idx:
                # Extract the span
                answer_tokens = qa_inputs['input_ids'][0][start_idx:end_idx+1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Calculate confidence
                start_score = torch.max(start_scores).item()
                end_score = torch.max(end_scores).item()
                confidence = (start_score + end_score) / 2
                
                return {
                    "title": self._clean_title(answer),
                    "confidence": confidence,
                    "method": "span"
                }
            
        except Exception as e:
            print(f"Error in span detection: {e}")
        
        return {"title": "", "confidence": 0.0, "method": "span"}
    
    def extract_title_multiple_questions(self, sentence: str) -> List[Dict]:
        """
        Ask multiple specific questions and rank results
        """
        if not self.qa_pipeline:
            return []
        
        # More specific questions for different event types
        specific_questions = [
            "What conference is mentioned?",
            "What workshop is described?",
            "What meeting is scheduled?",
            "What festival is happening?",
            "What concert is mentioned?",
            "What exhibition is described?",
            "What training is offered?",
            "What seminar is mentioned?",
            "What competition is described?",
            "What celebration is happening?",
            "What program is mentioned?",
            "What course is offered?",
            "What show is described?",
            "What gathering is mentioned?"
        ]
        
        results = []
        
        for question in specific_questions:
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=sentence,
                    max_answer_len=50
                )
                
                answer = result['answer'].strip()
                if len(answer) > 3 and result['score'] > 0.1:
                    results.append({
                        "title": self._clean_title(answer),
                        "confidence": result['score'],
                        "question": question,
                        "method": "specific_qa"
                    })
                    
            except Exception as e:
                continue
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize the extracted title"""
        if not title:
            return ""
        
        # Remove extra spaces
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove common prefixes/suffixes that aren't part of the title
        prefixes_to_remove = [
            'the event', 'event', 'the activity', 'activity',
            'the program', 'program', 'the session', 'session',
            'this', 'that', 'a', 'an', 'the meeting', 'meeting'
        ]
        
        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix.lower()):
                title = title[len(prefix):].strip()
                break
        
        # Remove trailing punctuation
        title = title.strip('.,!?;:-')
        
        # Capitalize properly
        if title and not title.isupper():
            title = title.title()
        
        return title
    
    def extract_title_ensemble(self, sentence: str) -> Dict:
        """
        Use multiple approaches and combine results
        """
        results = []
        
        # Method 1: Standard QA
        qa_result = self.extract_title_qa(sentence)
        if qa_result['title']:
            results.append(qa_result)
        
        # Method 2: Span detection
        span_result = self.extract_title_span_detection(sentence)
        if span_result['title']:
            results.append(span_result)
        
        # Method 3: Multiple specific questions
        specific_results = self.extract_title_multiple_questions(sentence)
        results.extend(specific_results[:3])  # Top 3 results
        
        # Method 4: Regex fallback
        regex_result = self._extract_title_regex(sentence)
        if regex_result:
            results.append({
                "title": regex_result,
                "confidence": 0.5,
                "method": "regex"
            })
        
        if not results:
            return {"title": "", "confidence": 0.0, "method": "none"}
        
        # Sort by confidence and return the best result
        results.sort(key=lambda x: x['confidence'], reverse=True)
        best_result = results[0]
        
        # Add all results for reference
        best_result['all_results'] = results
        
        return best_result
    
    def _extract_title_regex(self, sentence: str) -> str:
        """Fallback regex method"""
        patterns = [
            r'Event:\s*(.+?)(?:\s*-|\s*\||\s*at|\s*on|$)',
            r'^(.+?)\s*-\s*(?:on|at|\d{1,2}[\/\-]\d{1,2}|\w+\s+\d{1,2})',
            r'(?:Join us for|Attend|Welcome to|Come to)\s+(.+?)(?:\s*-|\s*\||\s*at|\s*on|$)',
            r'"([^"]+)"',
            r"'([^']+)'",
            r'(?:the|The)\s+([A-Z][^.!?]*?)(?:\s+(?:is|will be|starts|begins|happening))',
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
    
    def extract_titles_batch(self, sentences: List[str], method: str = "ensemble") -> List[Dict]:
        """
        Extract titles from multiple sentences
        
        Args:
            sentences: List of input sentences
            method: 'qa', 'span', 'ensemble', or 'specific'
        """
        results = []
        
        print(f"Processing {len(sentences)} sentences using {method} method...")
        
        for i, sentence in enumerate(sentences):
            print(f"Processing sentence {i+1}/{len(sentences)}")
            
            if method == "qa":
                result = self.extract_title_qa(sentence)
            elif method == "span":
                result = self.extract_title_span_detection(sentence)
            elif method == "specific":
                specific_results = self.extract_title_multiple_questions(sentence)
                result = specific_results[0] if specific_results else {"title": "", "confidence": 0.0}
            else:  # ensemble
                result = self.extract_title_ensemble(sentence)
            
            results.append({
                "index": i,
                "sentence": sentence,
                "title": result.get("title", ""),
                "confidence": result.get("confidence", 0.0),
                "method": result.get("method", ""),
                "details": result
            })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.model:
            return {"status": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__,
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": self.model.config.max_position_embeddings,
            "device": next(self.model.parameters()).device if hasattr(self.model, 'parameters') else "unknown"
        }

# Example usage
if __name__ == "__main__":
    # Initialize with your local model path
    MODEL_PATH = "./distilbert-base-cased-distilled-squad"  # Change this to your actual path
    
    extractor = LocalDistilBERTEventExtractor(MODEL_PATH)
    
    # Check if model loaded successfully
    if extractor.model is None:
        print("Model loading failed. Please check your model path.")
        exit(1)
    
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
    
    # Model information
    print("Model Information:")
    print(json.dumps(extractor.get_model_info(), indent=2))
    print("\n" + "="*80 + "\n")
    
    # Test individual methods
    test_sentence = example_sentences[0]
    print(f"Test Sentence: {test_sentence}\n")
    
    # QA Method
    qa_result = extractor.extract_title_qa(test_sentence)
    print(f"QA Method: {qa_result['title']} (confidence: {qa_result['confidence']:.3f})")
    
    # Ensemble Method
    ensemble_result = extractor.extract_title_ensemble(test_sentence)
    print(f"Ensemble Method: {ensemble_result['title']} (confidence: {ensemble_result['confidence']:.3f})")
    print(f"Method used: {ensemble_result['method']}")
    
    print("\n" + "="*80 + "\n")
    
    # Batch processing
    print("Batch Processing Results:")
    batch_results = extractor.extract_titles_batch(example_sentences[:4], method="ensemble")
    
    for result in batch_results:
        print(f"Sentence: {result['sentence']}")
        print(f"Title: {result['title']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Method: {result['method']}")
        print("-" * 60)
