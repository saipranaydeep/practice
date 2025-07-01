"""
Advanced Date and Time Extractor using specialized libraries

Required installations:
pip install python-dateutil spacy datefinder parsedatetime duckling-python
python -m spacy download en_core_web_sm

Optional for better performance:
pip install transformers torch
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

# Core libraries
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta
import datefinder
import parsedatetime

# NLP libraries
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

# Optional: Duckling (Facebook's time parsing library)
try:
    from duckling import DucklingWrapper
    DUCKLING_AVAILABLE = True
except ImportError:
    DUCKLING_AVAILABLE = False
    print("Warning: Duckling not available. Install with: pip install duckling-python")

# Optional: Transformers for advanced NER
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")


class AdvancedDateTimeExtractor:
    def __init__(self, use_spacy=True, use_duckling=False, use_transformers=False):
        """
        Initialize the advanced extractor with various backends
        
        Args:
            use_spacy: Use spaCy for NER-based extraction
            use_duckling: Use Duckling for robust parsing
            use_transformers: Use transformer models for entity recognition
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_duckling = use_duckling and DUCKLING_AVAILABLE
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        
        # Initialize parsedatetime calendar
        self.cal = parsedatetime.Calendar()
        
        # Initialize spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: en_core_web_sm model not found. Run: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        # Initialize Duckling
        if self.use_duckling:
            try:
                self.duckling = DucklingWrapper()
            except Exception as e:
                print(f"Warning: Duckling initialization failed: {e}")
                self.use_duckling = False
        
        # Initialize Transformers NER
        if self.use_transformers:
            try:
                # Using a model trained for temporal expression recognition
                self.ner_pipeline = pipeline(
                    "ner", 
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
            except Exception as e:
                print(f"Warning: Transformers initialization failed: {e}")
                self.use_transformers = False
        
        # Regex patterns for edge cases
        self.additional_patterns = [
            # Relative time patterns
            r'\b(in\s+)?(\d+)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b',
            r'\b(next|last|this)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(morning|afternoon|evening|night|noon|midnight|dawn|dusk)\b',
            r'\b(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)\b',
            # Specific time formats
            r'\b(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?\b',
            r'\b(\d{1,2})\s*o\'?clock\b',
            # Date formats
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',
            r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.additional_patterns]

    def extract_with_datefinder(self, text: str) -> List[Dict]:
        """Extract dates using datefinder library"""
        results = []
        try:
            matches = list(datefinder.find_dates(text))
            for i, date in enumerate(matches):
                results.append({
                    'method': 'datefinder',
                    'datetime': date,
                    'iso_format': date.isoformat(),
                    'confidence': 0.8,
                    'index': i
                })
        except Exception as e:
            print(f"Datefinder error: {e}")
        return results

    def extract_with_dateutil(self, text: str) -> List[Dict]:
        """Extract dates using dateutil parser"""
        results = []
        
        # Split text into potential date chunks
        words = text.split()
        
        # Try parsing different combinations of words
        for i in range(len(words)):
            for j in range(i + 1, min(i + 6, len(words) + 1)):  # Try up to 5-word combinations
                chunk = ' '.join(words[i:j])
                try:
                    parsed_date = dateutil_parser.parse(chunk, fuzzy=True)
                    results.append({
                        'method': 'dateutil',
                        'datetime': parsed_date,
                        'iso_format': parsed_date.isoformat(),
                        'text_chunk': chunk,
                        'confidence': 0.7,
                        'start_pos': i,
                        'end_pos': j
                    })
                except (ValueError, TypeError):
                    continue
        
        # Remove duplicates
        unique_results = []
        seen_dates = set()
        for result in results:
            date_key = result['datetime'].replace(microsecond=0)
            if date_key not in seen_dates:
                seen_dates.add(date_key)
                unique_results.append(result)
        
        return unique_results

    def extract_with_parsedatetime(self, text: str) -> List[Dict]:
        """Extract dates using parsedatetime library"""
        results = []
        try:
            time_struct, parse_status = self.cal.parse(text)
            if parse_status > 0:  # Successful parse
                parsed_date = datetime(*time_struct[:6])
                results.append({
                    'method': 'parsedatetime',
                    'datetime': parsed_date,
                    'iso_format': parsed_date.isoformat(),
                    'confidence': 0.9,
                    'parse_status': parse_status
                })
        except Exception as e:
            print(f"Parsedatetime error: {e}")
        return results

    def extract_with_spacy(self, text: str) -> List[Dict]:
        """Extract dates using spaCy NER"""
        if not self.use_spacy:
            return []
            
        results = []
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME', 'EVENT']:
                    # Try to parse the entity text
                    try:
                        parsed_date = dateutil_parser.parse(ent.text, fuzzy=True)
                        results.append({
                            'method': 'spacy',
                            'datetime': parsed_date,
                            'iso_format': parsed_date.isoformat(),
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 0.75
                        })
                    except:
                        # If parsing fails, still record the entity
                        results.append({
                            'method': 'spacy',
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 0.5,
                            'parsing_failed': True
                        })
        except Exception as e:
            print(f"SpaCy error: {e}")
        return results

    def extract_with_duckling(self, text: str) -> List[Dict]:
        """Extract dates using Duckling"""
        if not self.use_duckling:
            return []
            
        results = []
        try:
            duckling_results = self.duckling.parse(text)
            for result in duckling_results:
                if result.get('dim') in ['time', 'duration']:
                    value = result.get('value', {})
                    if 'value' in value:
                        try:
                            parsed_date = dateutil_parser.parse(value['value'])
                            results.append({
                                'method': 'duckling',
                                'datetime': parsed_date,
                                'iso_format': parsed_date.isoformat(),
                                'text': result.get('body'),
                                'confidence': 0.95,
                                'duckling_data': result
                            })
                        except:
                            pass
        except Exception as e:
            print(f"Duckling error: {e}")
        return results

    def extract_with_regex(self, text: str) -> List[Dict]:
        """Extract dates using enhanced regex patterns"""
        results = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.finditer(text)
            for match in matches:
                matched_text = match.group()
                try:
                    # Try to parse the matched text
                    parsed_date = dateutil_parser.parse(matched_text, fuzzy=True)
                    results.append({
                        'method': 'regex',
                        'pattern_index': i,
                        'datetime': parsed_date,
                        'iso_format': parsed_date.isoformat(),
                        'text': matched_text,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.6
                    })
                except:
                    # Record the match even if parsing fails
                    results.append({
                        'method': 'regex',
                        'pattern_index': i,
                        'text': matched_text,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.3,
                        'parsing_failed': True
                    })
        
        return results

    def extract_datetime(self, text: str) -> Dict:
        """
        Extract datetime information using multiple methods
        
        Args:
            text: Input text containing date/time information
            
        Returns:
            Dictionary with extraction results from all methods
        """
        results = {
            'original_text': text,
            'datefinder': self.extract_with_datefinder(text),
            'dateutil': self.extract_with_dateutil(text),
            'parsedatetime': self.extract_with_parsedatetime(text),
            'spacy': self.extract_with_spacy(text),
            'duckling': self.extract_with_duckling(text),
            'regex': self.extract_with_regex(text)
        }
        
        # Combine and rank results
        results['combined'] = self.combine_results(results)
        results['best_matches'] = self.get_best_matches(results['combined'])
        
        return results

    def combine_results(self, results: Dict) -> List[Dict]:
        """Combine results from all methods and remove duplicates"""
        all_results = []
        
        for method, method_results in results.items():
            if method in ['original_text', 'combined', 'best_matches']:
                continue
            all_results.extend(method_results)
        
        # Remove duplicates based on datetime similarity
        unique_results = []
        seen_datetimes = []
        
        for result in all_results:
            if 'datetime' not in result:
                unique_results.append(result)
                continue
                
            is_duplicate = False
            current_dt = result['datetime']
            
            for seen_dt in seen_datetimes:
                # Consider datetimes within 1 minute as duplicates
                if abs((current_dt - seen_dt).total_seconds()) < 60:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_datetimes.append(current_dt)
                unique_results.append(result)
        
        # Sort by confidence
        unique_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return unique_results

    def get_best_matches(self, combined_results: List[Dict]) -> List[Dict]:
        """Get the most likely datetime extractions"""
        # Filter out results with parsing failures
        valid_results = [r for r in combined_results if 'datetime' in r and not r.get('parsing_failed', False)]
        
        # Group by similar datetime values
        groups = {}
        for result in valid_results:
            dt = result['datetime']
            # Create a key based on rounded datetime (to nearest hour)
            key = dt.replace(minute=0, second=0, microsecond=0)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        # Select best result from each group
        best_matches = []
        for group in groups.values():
            # Sort by confidence and method reliability
            method_priority = {
                'duckling': 5, 'parsedatetime': 4, 'datefinder': 3, 
                'spacy': 2, 'dateutil': 1, 'regex': 0
            }
            
            group.sort(key=lambda x: (
                x.get('confidence', 0),
                method_priority.get(x.get('method', ''), 0)
            ), reverse=True)
            
            best_matches.append(group[0])
        
        return best_matches[:3]  # Return top 3 matches

    def extract_simple(self, text: str) -> List[datetime]:
        """Simple interface that returns just the datetime objects"""
        results = self.extract_datetime(text)
        return [r['datetime'] for r in results['best_matches'] if 'datetime' in r]


def test_advanced_extractor():
    """Test the advanced extractor with various sentence formats"""
    
    extractor = AdvancedDateTimeExtractor(
        use_spacy=True, 
        use_duckling=False,  # Set to True if duckling is installed
        use_transformers=False  # Set to True if you want to use transformers
    )
    
    test_sentences = [
        "The meeting is scheduled for tomorrow at 3:30 PM",
        "Let's meet on December 25, 2024 at 9:00 AM",
        "The event starts at half past seven on Friday",
        "Deadline is 15/03/2024 by 11:59 PM",
        "Call me today at 2 o'clock",
        "The appointment is next week at quarter past 4",
        "Conference call on Mon at 10:30",
        "Party on 2024-12-31 at midnight",
        "Lunch meeting tomorrow 12pm",
        "The concert is on Saturday evening at 7:30",
        "Submit by Jan 15 before 5 PM",
        "Meeting at 14:30 on 25th March",
        "Let's catch up this Friday around 6:30 in the evening",
        "Dinner reservation for 8 people on the 15th at 7:30 PM",
        "Conference starts Monday morning at 9 AM",
        "Deadline is in 3 days at noon",
        "Call scheduled for next Tuesday at 2:15 PM",
        "Event happening on Christmas Day",
        "Meeting rescheduled to next Friday afternoon",
        "Delivery expected between 2-4 PM today"
    ]
    
    print("Advanced Date and Time Extraction Results:")
    print("=" * 60)
    
    for sentence in test_sentences:
        print(f"\n📝 Input: {sentence}")
        result = extractor.extract_datetime(sentence)
        
        print("\n🎯 Best Matches:")
        if result['best_matches']:
            for i, match in enumerate(result['best_matches'], 1):
                dt_str = match['datetime'].strftime('%Y-%m-%d %H:%M:%S')
                method = match.get('method', 'unknown')
                confidence = match.get('confidence', 0)
                text_match = match.get('text', match.get('text_chunk', ''))
                print(f"  {i}. {dt_str} (method: {method}, confidence: {confidence:.2f})")
                if text_match:
                    print(f"     Matched text: '{text_match}'")
        else:
            print("  No datetime information found")
        
        # Show method breakdown
        method_counts = {}
        for method, method_results in result.items():
            if method in ['original_text', 'combined', 'best_matches']:
                continue
            valid_results = [r for r in method_results if 'datetime' in r]
            if valid_results:
                method_counts[method] = len(valid_results)
        
        if method_counts:
            print(f"  📊 Method results: {dict(method_counts)}")
        
        print("-" * 60)

    # Demonstrate simple interface
    print("\n🚀 Simple Interface Demo:")
    print("-" * 30)
    simple_results = extractor.extract_simple("Meeting tomorrow at 3 PM and deadline next Friday")
    for dt in simple_results:
        print(f"Found: {dt.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    test_advanced_extractor()
