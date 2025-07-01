import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import calendar

class DateTimeExtractor:
    def __init__(self):
        # Month mappings
        self.months = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        # Day mappings
        self.days = {
            'monday': 0, 'mon': 0, 'tuesday': 1, 'tue': 1, 'tues': 1,
            'wednesday': 2, 'wed': 2, 'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
            'friday': 4, 'fri': 4, 'saturday': 5, 'sat': 5, 'sunday': 6, 'sun': 6
        }
        
        # Relative time mappings
        self.relative_times = {
            'today': 0, 'tomorrow': 1, 'yesterday': -1,
            'next week': 7, 'last week': -7, 'this week': 0,
            'next month': 30, 'last month': -30, 'this month': 0
        }
        
        # Time patterns
        self.time_patterns = [
            # 12-hour format with AM/PM
            r'(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?\s*(?P<ampm>[ap]\.?m\.?)',
            r'(?P<hour>\d{1,2})\s*(?P<ampm>[ap]\.?m\.?)',
            # 24-hour format
            r'(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?(?!\s*[ap]\.?m\.?)',
            # Informal time expressions
            r'(?P<hour>\d{1,2})\s*o\'?clock',
            r'half\s*past\s*(?P<hour>\d{1,2})',
            r'quarter\s*past\s*(?P<hour>\d{1,2})',
            r'quarter\s*to\s*(?P<hour>\d{1,2})',
        ]
        
        # Date patterns
        self.date_patterns = [
            # Standard formats
            r'(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{2,4})',
            r'(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>\d{2,4})',
            r'(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})',
            # Month name formats
            r'(?P<month>' + '|'.join(self.months.keys()) + r')\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})',
            r'(?P<day>\d{1,2})\s+(?P<month>' + '|'.join(self.months.keys()) + r')\s+(?P<year>\d{4})',
            r'(?P<month>' + '|'.join(self.months.keys()) + r')\s+(?P<day>\d{1,2})',
            r'(?P<day>\d{1,2})\s+(?P<month>' + '|'.join(self.months.keys()) + r')',
            # Day of week patterns
            r'(?P<day_name>' + '|'.join(self.days.keys()) + r')',
            # Relative date patterns
            r'(?P<relative>today|tomorrow|yesterday|next\s+week|last\s+week|this\s+week|next\s+month|last\s+month|this\s+month)',
        ]
    
    def extract_time(self, text: str) -> List[Dict]:
        """Extract time information from text"""
        text = text.lower().strip()
        times = []
        
        for pattern in self.time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                time_info = self._parse_time_match(match)
                if time_info:
                    times.append(time_info)
        
        return times
    
    def extract_date(self, text: str) -> List[Dict]:
        """Extract date information from text"""
        text = text.lower().strip()
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_info = self._parse_date_match(match)
                if date_info:
                    dates.append(date_info)
        
        return dates
    
    def extract_datetime(self, text: str) -> Dict:
        """Extract both date and time information from text"""
        times = self.extract_time(text)
        dates = self.extract_date(text)
        
        return {
            'original_text': text,
            'dates': dates,
            'times': times,
            'combined': self._combine_datetime(dates, times)
        }
    
    def _parse_time_match(self, match) -> Optional[Dict]:
        """Parse a time regex match into structured data"""
        groups = match.groupdict()
        
        try:
            if 'hour' in groups and groups['hour']:
                hour = int(groups['hour'])
                minute = int(groups.get('minute', 0)) if groups.get('minute') else 0
                second = int(groups.get('second', 0)) if groups.get('second') else 0
                
                # Handle AM/PM
                if 'ampm' in groups and groups['ampm']:
                    ampm = groups['ampm'].replace('.', '').lower()
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    elif ampm == 'am' and hour == 12:
                        hour = 0
                
                # Handle special cases
                if 'half past' in match.group():
                    minute = 30
                elif 'quarter past' in match.group():
                    minute = 15
                elif 'quarter to' in match.group():
                    hour = (hour + 1) % 24
                    minute = 45
                
                return {
                    'type': 'time',
                    'hour': hour,
                    'minute': minute,
                    'second': second,
                    'matched_text': match.group(),
                    'time_string': f"{hour:02d}:{minute:02d}:{second:02d}"
                }
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _parse_date_match(self, match) -> Optional[Dict]:
        """Parse a date regex match into structured data"""
        groups = match.groupdict()
        current_date = datetime.now()
        
        try:
            # Handle relative dates
            if 'relative' in groups and groups['relative']:
                relative = groups['relative'].strip()
                if relative in self.relative_times:
                    delta_days = self.relative_times[relative]
                    target_date = current_date + timedelta(days=delta_days)
                    return {
                        'type': 'relative_date',
                        'year': target_date.year,
                        'month': target_date.month,
                        'day': target_date.day,
                        'matched_text': match.group(),
                        'date_string': target_date.strftime('%Y-%m-%d')
                    }
            
            # Handle day names
            if 'day_name' in groups and groups['day_name']:
                day_name = groups['day_name'].strip()
                if day_name in self.days:
                    target_weekday = self.days[day_name]
                    current_weekday = current_date.weekday()
                    days_ahead = target_weekday - current_weekday
                    if days_ahead <= 0:  # Target day is today or has passed this week
                        days_ahead += 7  # Move to next week
                    target_date = current_date + timedelta(days=days_ahead)
                    return {
                        'type': 'day_name',
                        'year': target_date.year,
                        'month': target_date.month,
                        'day': target_date.day,
                        'day_name': day_name,
                        'matched_text': match.group(),
                        'date_string': target_date.strftime('%Y-%m-%d')
                    }
            
            # Handle explicit dates
            year = None
            month = None
            day = None
            
            if 'year' in groups and groups['year']:
                year = int(groups['year'])
                if year < 100:  # Handle 2-digit years
                    year += 2000 if year < 50 else 1900
            
            if 'month' in groups and groups['month']:
                if groups['month'].isdigit():
                    month = int(groups['month'])
                else:
                    month_name = groups['month'].lower()
                    month = self.months.get(month_name)
            
            if 'day' in groups and groups['day']:
                day = int(groups['day'])
            
            # Use current year if not specified
            if year is None:
                year = current_date.year
            
            if month and day and 1 <= month <= 12 and 1 <= day <= 31:
                try:
                    # Validate the date
                    datetime(year, month, day)
                    return {
                        'type': 'explicit_date',
                        'year': year,
                        'month': month,
                        'day': day,
                        'matched_text': match.group(),
                        'date_string': f"{year:04d}-{month:02d}-{day:02d}"
                    }
                except ValueError:
                    pass
        
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _combine_datetime(self, dates: List[Dict], times: List[Dict]) -> List[Dict]:
        """Combine date and time information"""
        combined = []
        
        if not dates and not times:
            return combined
        
        # If only times are present, use current date
        if times and not dates:
            current_date = datetime.now()
            for time_info in times:
                try:
                    dt = datetime(
                        current_date.year, current_date.month, current_date.day,
                        time_info['hour'], time_info['minute'], time_info['second']
                    )
                    combined.append({
                        'datetime': dt.isoformat(),
                        'date_part': current_date.strftime('%Y-%m-%d'),
                        'time_part': time_info['time_string'],
                        'source': 'time_only'
                    })
                except:
                    pass
        
        # If only dates are present
        elif dates and not times:
            for date_info in dates:
                combined.append({
                    'datetime': f"{date_info['date_string']}T00:00:00",
                    'date_part': date_info['date_string'],
                    'time_part': '00:00:00',
                    'source': 'date_only'
                })
        
        # If both dates and times are present, combine them
        else:
            for date_info in dates:
                for time_info in times:
                    try:
                        dt = datetime(
                            date_info['year'], date_info['month'], date_info['day'],
                            time_info['hour'], time_info['minute'], time_info['second']
                        )
                        combined.append({
                            'datetime': dt.isoformat(),
                            'date_part': date_info['date_string'],
                            'time_part': time_info['time_string'],
                            'source': 'combined'
                        })
                    except:
                        pass
        
        return combined

# Example usage and testing
def test_extractor():
    extractor = DateTimeExtractor()
    
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
        "Let's catch up this Friday around 6:30 in the evening"
    ]
    
    print("Date and Time Extraction Results:")
    print("=" * 50)
    
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        result = extractor.extract_datetime(sentence)
        
        if result['dates']:
            print("Dates found:")
            for date in result['dates']:
                print(f"  - {date['date_string']} ({date['type']}) - '{date['matched_text']}'")
        
        if result['times']:
            print("Times found:")
            for time in result['times']:
                print(f"  - {time['time_string']} - '{time['matched_text']}'")
        
        if result['combined']:
            print("Combined DateTime:")
            for dt in result['combined']:
                print(f"  - {dt['datetime']} (source: {dt['source']})")
        
        if not result['dates'] and not result['times']:
            print("  No date/time information found")
        
        print("-" * 50)

if __name__ == "__main__":
    test_extractor()
