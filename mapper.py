from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class ColumnMapper:
    """Map natural language terms to dataset columns using embeddings and fuzzy matching"""
    
    def __init__(self, schemas):
        self.schemas = schemas
        self.embedding_model = None
        self._load_embedding_model()
        self._build_column_index()
        
        # Enhanced business term mappings for better intelligence
        self.business_term_mappings = {
            # Revenue/Sales terms
            'revenue': ['revenue', 'sales', 'income', 'earnings', 'proceeds', 'turnover'],
            'sales': ['sales', 'revenue', 'amount', 'total', 'value', 'deal_value'],
            'profit': ['profit', 'margin', 'earnings', 'net', 'gross'],
            
            # Customer terms
            'customer': ['customer', 'client', 'account', 'user', 'buyer'],
            'user': ['user', 'customer', 'member', 'subscriber'],
            
            # Product terms
            'product': ['product', 'item', 'sku', 'service', 'offering'],
            'category': ['category', 'type', 'class', 'segment', 'group'],
            
            # Geography terms
            'region': ['region', 'area', 'territory', 'location', 'zone'],
            'country': ['country', 'nation', 'market'],
            'city': ['city', 'location', 'place'],
            
            # Time terms
            'date': ['date', 'time', 'timestamp', 'created', 'updated', 'when'],
            'month': ['month', 'monthly', 'period'],
            'quarter': ['quarter', 'q1', 'q2', 'q3', 'q4'],
            
            # Marketing terms
            'channel': ['channel', 'source', 'medium', 'campaign'],
            'spend': ['spend', 'cost', 'investment', 'budget'],
            'conversion': ['conversion', 'convert', 'success', 'rate'],
            
            # Sales terms
            'rep': ['rep', 'representative', 'salesperson', 'sales_rep'],
            'deal': ['deal', 'opportunity', 'sale', 'transaction'],
            
            # Metrics
            'count': ['count', 'number', 'quantity', 'total', 'sum'],
            'average': ['average', 'avg', 'mean'],
            'rate': ['rate', 'percentage', 'percent', 'ratio']
        }
    
    def _load_embedding_model(self):
        """Load sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            # Will fallback to fuzzy matching only
    
    def _build_column_index(self):
        """Build an index of all columns across datasets"""
        self.column_index = {}
        self.all_columns = []
        
        for dataset_name, schema in self.schemas.items():
            for col_name in schema['columns'].keys():
                self.column_index[col_name.lower()] = {
                    'dataset': dataset_name,
                    'original_name': col_name,
                    'dtype': schema['columns'][col_name]['dtype']
                }
                self.all_columns.append(col_name.lower())
        
        # Pre-compute embeddings if model is available
        if self.embedding_model and self.all_columns:
            try:
                self.column_embeddings = self.embedding_model.encode(self.all_columns)
            except Exception as e:
                print(f"Warning: Could not compute embeddings: {e}")
                self.column_embeddings = None
        else:
            self.column_embeddings = None
    
    def map_columns(self, plan):
        """Map column references in the plan to actual dataset columns with enhanced intelligence"""
        mapped_plan = plan.copy()
        mapping_confidence = {}
        
        # Map columns mentioned in the question
        mapped_columns = []
        for col_ref in plan.get('columns', []):
            mapped_col, confidence = self._find_best_column_match_with_confidence(col_ref)
            if mapped_col:
                mapped_columns.append(mapped_col)
                mapping_confidence[col_ref] = confidence
        mapped_plan['columns'] = mapped_columns
        
        # Map group-by columns
        mapped_groupby = []
        for groupby_ref in plan.get('groupby', []):
            mapped_col, confidence = self._find_best_column_match_with_confidence(groupby_ref)
            if mapped_col:
                mapped_groupby.append(mapped_col)
                mapping_confidence[groupby_ref] = confidence
        mapped_plan['groupby'] = mapped_groupby
        
        # Map filter columns
        mapped_filters = []
        for filter_def in plan.get('filters', []):
            if filter_def.get('column'):
                mapped_col, confidence = self._find_best_column_match_with_confidence(filter_def['column'])
                if mapped_col:
                    filter_def['column'] = mapped_col['original_name']
                    filter_def['dataset'] = mapped_col['dataset']
                    mapped_filters.append(filter_def)
        mapped_plan['filters'] = mapped_filters
        
        # Add mapping warnings for low confidence
        low_confidence_mappings = [k for k, v in mapping_confidence.items() if v < 0.6]
        if low_confidence_mappings:
            mapped_plan['mapping_warnings'] = [
                f"Low confidence mapping for: {', '.join(low_confidence_mappings)}"
            ]
            mapped_plan['suggested_alternatives'] = self._suggest_alternatives(low_confidence_mappings)
        
        # Smart inference for missing mappings
        mapped_plan = self._infer_missing_mappings(mapped_plan)
        
        return mapped_plan
    
    def _find_best_column_match_with_confidence(self, term):
        """Find the best matching column for a term with confidence score"""
        if not term or not isinstance(term, str):
            return None, 0
            
        term_lower = term.lower()
        
        # First check business term mappings for enhanced intelligence
        for business_term, synonyms in self.business_term_mappings.items():
            if term_lower in synonyms:
                # Look for columns matching this business concept
                for col_name in self.all_columns:
                    if any(syn in col_name for syn in synonyms):
                        return self.column_index[col_name], 0.95
        
        # Exact match
        if term_lower in self.column_index:
            return self.column_index[term_lower], 1.0
        
        # Try fuzzy matching with business context weighting
        if self.all_columns:
            fuzzy_matches = process.extract(
                term_lower, 
                self.all_columns, 
                scorer=fuzz.ratio,
                limit=5
            )
            
            # Weight matches by business relevance
            weighted_matches = []
            for match, score in fuzzy_matches:
                weight = 1.0
                
                # Boost score for business-relevant columns
                for business_term in self.business_term_mappings.keys():
                    if business_term in match and business_term in term_lower:
                        weight = 1.2
                        break
                
                weighted_score = min(score * weight, 100)
                weighted_matches.append((match, weighted_score))
            
            best_fuzzy = max(weighted_matches, key=lambda x: x[1])
            fuzzy_confidence = best_fuzzy[1] / 100
        else:
            fuzzy_confidence = 0
            best_fuzzy = (None, 0)
        
        # Semantic matching if available
        semantic_confidence = 0
        best_semantic_match = None
        
        if (self.embedding_model and 
            self.column_embeddings is not None and 
            len(self.all_columns) > 0):
            try:
                term_embedding = self.embedding_model.encode([term_lower])
                similarities = np.dot(self.column_embeddings, term_embedding.T).flatten()
                best_idx = np.argmax(similarities)
                semantic_confidence = similarities[best_idx]
                best_semantic_match = self.all_columns[best_idx]
            except Exception as e:
                semantic_confidence = 0
        
        # Choose best match
        if semantic_confidence > fuzzy_confidence and semantic_confidence > 0.7:
            return self.column_index[best_semantic_match], semantic_confidence
        elif fuzzy_confidence > 0.6:
            return self.column_index[best_fuzzy[0]], fuzzy_confidence
        
        return None, 0
    
    def _suggest_alternatives(self, failed_terms):
        """Suggest alternative column mappings for failed terms"""
        suggestions = {}
        
        for term in failed_terms:
            term_lower = term.lower()
            alternatives = []
            
            # Look for partial matches in column names
            for col_name in self.all_columns:
                if any(word in col_name for word in term_lower.split()):
                    alternatives.append(self.column_index[col_name]['original_name'])
            
            # Business context suggestions
            for business_term, synonyms in self.business_term_mappings.items():
                if term_lower in synonyms:
                    for col_name in self.all_columns:
                        if any(syn in col_name for syn in synonyms):
                            alternatives.append(self.column_index[col_name]['original_name'])
            
            suggestions[term] = list(set(alternatives))[:5]  # Top 5 unique suggestions
        
        return suggestions
    
    def _infer_missing_mappings(self, plan):
        """Infer missing column mappings based on context and data types"""
        # If no specific columns mapped but we have a metric, try to infer target column
        if not plan.get('columns') and plan.get('metric'):
            target_col = self._infer_metric_column(plan['metric'])
            if target_col:
                plan['columns'] = [target_col]
        
        # If no group-by specified but question seems to want grouping, infer it
        if not plan.get('groupby') and plan.get('intent') == 'ranking':
            groupby_col = self._infer_groupby_column(plan)
            metric_col = self._infer_metric_column(plan.get('metric', 'sum'))
            
            # Build groupby list with both categorical and metric columns
            groupby_list = []
            if groupby_col:
                groupby_list.append(groupby_col)
            if metric_col and metric_col != groupby_col:
                groupby_list.append(metric_col)
            
            if groupby_list:
                plan['groupby'] = groupby_list
        
        # Infer date column for time filters
        for time_filter in plan.get('time_filters', []):
            if 'column' not in time_filter:
                date_col = self._find_date_column()
                if date_col:
                    time_filter['column'] = date_col['original_name']
                    time_filter['dataset'] = date_col['dataset']
        
        return plan
    
    def _infer_metric_column(self, metric):
        """Infer which column to apply the metric to"""
        # Look for numeric columns that match common business metrics
        numeric_preferences = {
            'sum': ['sales', 'revenue', 'amount', 'price', 'cost', 'spend', 'value', 'total', 'deal'],
            'count': ['id', 'order', 'customer', 'transaction'],
            'mean': ['price', 'amount', 'score', 'rating'],
            'min': ['price', 'date', 'amount'],
            'max': ['price', 'date', 'amount']
        }
        
        preferences = numeric_preferences.get(metric, ['amount', 'value', 'price', 'total', 'sales', 'deal'])
        
        for pref in preferences:
            match, confidence = self._find_best_column_match_with_confidence(pref)
            if match and self._is_numeric_column(match):
                return match
        
        # Fall back to first numeric column
        for dataset_name, schema in self.schemas.items():
            for col_name, col_info in schema['columns'].items():
                if self._is_numeric_dtype(col_info['dtype']):
                    return {
                        'dataset': dataset_name,
                        'original_name': col_name,
                        'dtype': col_info['dtype']
                    }
        
        return None
    
    def _infer_groupby_column(self, plan):
        """Infer which column to group by"""
        # Common grouping dimensions in order of preference
        groupby_preferences = [
            'product', 'customer', 'region', 'category', 'channel', 
            'type', 'status', 'segment', 'rep', 'sales_rep'
        ]
        
        for pref in groupby_preferences:
            match, confidence = self._find_best_column_match_with_confidence(pref)
            if match and self._is_categorical_column(match):
                return match
        
        # Fall back to first categorical column with reasonable cardinality
        for dataset_name, schema in self.schemas.items():
            for col_name, col_info in schema['columns'].items():
                if (self._is_categorical_dtype(col_info['dtype']) 
                    and col_info['unique_pct'] < 50  # Not too unique
                    and col_info['unique_pct'] > 5):  # Not too few categories
                    return {
                        'dataset': dataset_name,
                        'original_name': col_name,
                        'dtype': col_info['dtype']
                    }
        
        return None
    
    def _find_date_column(self):
        """Find the best date column across all datasets"""
        date_keywords = ['date', 'time', 'created', 'updated', 'timestamp']
        
        # Look for columns with date keywords
        for keyword in date_keywords:
            match, confidence = self._find_best_column_match_with_confidence(keyword)
            if match and self._is_date_column(match):
                return match
        
        # Look for datetime columns
        for dataset_name, schema in self.schemas.items():
            for col_name, col_info in schema['columns'].items():
                if 'datetime' in str(col_info['dtype']).lower() or 'date' in col_name.lower():
                    return {
                        'dataset': dataset_name,
                        'original_name': col_name,
                        'dtype': col_info['dtype']
                    }
        
        return None
    
    def _is_numeric_column(self, col_info):
        """Check if a column is numeric"""
        return self._is_numeric_dtype(col_info.get('dtype', ''))
    
    def _is_numeric_dtype(self, dtype):
        """Check if a dtype is numeric"""
        dtype_str = str(dtype).lower()
        return any(num_type in dtype_str for num_type in ['int', 'float', 'number'])
    
    def _is_categorical_column(self, col_info):
        """Check if a column is categorical"""
        return self._is_categorical_dtype(col_info.get('dtype', ''))
    
    def _is_categorical_dtype(self, dtype):
        """Check if a dtype is categorical"""
        dtype_str = str(dtype).lower()
        return 'object' in dtype_str or 'string' in dtype_str or 'category' in dtype_str
    
    def _is_date_column(self, col_info):
        """Check if a column is a date column"""
        dtype_str = str(col_info.get('dtype', '')).lower()
        return any(date_type in dtype_str for date_type in ['datetime', 'date', 'timestamp'])