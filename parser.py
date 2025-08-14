import spacy
import re
from dateutil import parser as date_parser
from datetime import datetime, timedelta
import calendar

class QuestionParser:
    """Parse natural language questions into structured analysis plans"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if spacy model not installed
            self.nlp = None
        
        # Keywords for different types of analysis
        self.metric_keywords = {
            'sum': ['sum', 'total', 'aggregate', 'add'],
            'mean': ['average', 'avg', 'mean'],
            'count': ['count', 'number', 'how many'],
            'min': ['minimum', 'min', 'lowest', 'smallest'],
            'max': ['maximum', 'max', 'highest', 'largest', 'biggest'],
            'std': ['std', 'standard deviation'],
            'median': ['median', 'middle']
        }
        
        self.comparison_operators = {
            '>=': ['greater than or equal', 'at least', '>=', 'minimum'],
            '<=': ['less than or equal', 'at most', '<=', 'maximum'],
            '>': ['greater than', 'more than', 'above', '>'],
            '<': ['less than', 'below', 'under', '<'],
            '==': ['equal', 'equals', '==', '=', 'is'],
            '!=': ['not equal', 'not', '!=', 'different']
        }
        
        self.time_patterns = {
            'quarter': r'Q[1-4]\s*\d{4}|quarter\s*[1-4]\s*\d{4}',
            'month': r'(january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}',
            'year': r'\d{4}',
            'last_days': r'last\s*(\d+)\s*days?',
            'last_months': r'last\s*(\d+)\s*months?',
            'ytd': r'year\s*to\s*date|ytd'
        }
    
    def parse(self, question):
        """Parse a natural language question into a structured plan"""
        question_lower = question.lower()
        
        # Initialize plan
        plan = {
            'intent': self._detect_intent(question_lower),
            'metric': self._extract_metric(question_lower),
            'columns': self._extract_columns(question),
            'groupby': self._extract_groupby(question_lower),
            'filters': self._extract_filters(question_lower),
            'time_filters': self._extract_time_filters(question_lower),
            'top_k': self._extract_top_k(question_lower),
            'sort_order': self._extract_sort_order(question_lower),
            'original_question': question
        }
        
        return plan
    
    def _detect_intent(self, question):
        """Detect the main intent of the question"""
        if any(word in question for word in ['top', 'best', 'highest', 'largest']):
            return 'ranking'
        elif any(word in question for word in ['bottom', 'worst', 'lowest', 'smallest']):
            return 'ranking'
        elif any(word in question for word in ['trend', 'over time', 'by month', 'by year']):
            return 'trend'
        elif any(word in question for word in ['compare', 'vs', 'versus', 'between']):
            return 'comparison'
        elif any(word in question for word in ['sum', 'total', 'count', 'average']):
            return 'aggregation'
        else:
            return 'general'
    
    def _extract_metric(self, question):
        """Extract the metric/aggregation type"""
        for metric, keywords in self.metric_keywords.items():
            if any(keyword in question for keyword in keywords):
                return metric
        
        # Default based on common words
        if any(word in question for word in ['revenue', 'sales', 'price', 'cost', 'amount']):
            return 'sum'
        elif any(word in question for word in ['count', 'number']):
            return 'count'
        else:
            return 'sum'  # Default
    
    def _extract_columns(self, question):
        """Extract potential column references from question"""
        # This is basic - the mapper will do the heavy lifting
        words = question.lower().split()
        
        # Common column name patterns
        potential_columns = []
        
        # Look for quoted strings
        quoted = re.findall(r'"([^"]*)"', question)
        potential_columns.extend(quoted)
        
        quoted = re.findall(r"'([^']*)'", question)
        potential_columns.extend(quoted)
        
        # Look for business terms that might be columns
        business_terms = [
            'product', 'customer', 'region', 'category', 'channel',
            'sales', 'revenue', 'price', 'quantity', 'date',
            'order', 'spend', 'cost', 'profit', 'margin'
        ]
        
        for term in business_terms:
            if term in question.lower():
                potential_columns.append(term)
        
        return list(set(potential_columns))
    
    def _extract_groupby(self, question):
        """Extract group-by dimensions"""
        groupby_indicators = ['by', 'per', 'for each', 'group by']
        potential_groupby = []
        
        words = question.split()
        for i, word in enumerate(words):
            if word.lower() in ['by', 'per'] and i < len(words) - 1:
                next_word = words[i + 1].lower()
                potential_groupby.append(next_word)
        
        # Common grouping dimensions
        common_dims = ['product', 'customer', 'region', 'category', 'channel', 'month', 'year']
        for dim in common_dims:
            if dim in question and dim not in potential_groupby:
                potential_groupby.append(dim)
        
        return potential_groupby
    
    def _extract_filters(self, question):
        """Extract filter conditions"""
        filters = []
        
        # Look for comparison patterns
        for op, keywords in self.comparison_operators.items():
            for keyword in keywords:
                if keyword in question:
                    # Try to extract the filter
                    pattern = f"{keyword}\\s+(\\w+)"
                    matches = re.findall(pattern, question, re.IGNORECASE)
                    for match in matches:
                        filters.append({
                            'operator': op,
                            'value': match,
                            'column': None  # Will be filled by mapper
                        })
        
        # Look for "in region X" patterns
        region_pattern = r'in\s+(region\s+)?(\w+)'
        matches = re.findall(region_pattern, question, re.IGNORECASE)
        for match in matches:
            filters.append({
                'operator': '==',
                'value': match[1],
                'column': 'region'
            })
        
        return filters
    
    def _extract_time_filters(self, question):
        """Extract time-based filters"""
        time_filters = []
        
        for time_type, pattern in self.time_patterns.items():
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if time_type == 'quarter':
                    # Parse Q1 2024 format
                    q_match = re.search(r'Q([1-4])\s*(\d{4})', match, re.IGNORECASE)
                    if q_match:
                        quarter, year = q_match.groups()
                        start_month = (int(quarter) - 1) * 3 + 1
                        end_month = int(quarter) * 3
                        time_filters.append({
                            'type': 'quarter',
                            'quarter': int(quarter),
                            'year': int(year),
                            'start_month': start_month,
                            'end_month': end_month
                        })
                
                elif time_type == 'last_days':
                    days = int(match)
                    time_filters.append({
                        'type': 'last_days',
                        'days': days
                    })
                
                elif time_type == 'year':
                    year = int(match)
                    time_filters.append({
                        'type': 'year',
                        'year': year
                    })
        
        return time_filters
    
    def _extract_top_k(self, question):
        """Extract top-K limit"""
        # Look for "top 5", "best 10", etc.
        top_pattern = r'(?:top|best|highest|largest|biggest)\s*(\d+)'
        matches = re.findall(top_pattern, question, re.IGNORECASE)
        if matches:
            return int(matches[0])
        
        bottom_pattern = r'(?:bottom|worst|lowest|smallest)\s*(\d+)'
        matches = re.findall(bottom_pattern, question, re.IGNORECASE)
        if matches:
            return int(matches[0])
        
        # Default
        return 10
    
    def _extract_sort_order(self, question):
        """Extract sort order preference"""
        if any(word in question.lower() for word in ['top', 'best', 'highest', 'largest', 'desc']):
            return 'desc'
        elif any(word in question.lower() for word in ['bottom', 'worst', 'lowest', 'smallest', 'asc']):
            return 'asc'
        else:
            return 'desc'  # Default for rankings