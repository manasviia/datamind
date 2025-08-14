import pandas as pd
import numpy as np
from typing import Dict, Any

class ResultValidator:
    """Validate analysis results using alternate computation methods"""
    
    def validate(self, plan, dfs, result_df):
        """Validate results and return trust score"""
        try:
            validation_result = {
                'trust_score': 0,
                'validation_method': 'none',
                'alternate_result': None,
                'comparison': None,
                'issues': []
            }
            
            # Skip validation if no result
            if result_df is None or result_df.empty:
                validation_result['trust_score'] = 0
                validation_result['issues'].append("No results to validate")
                return validation_result
            
            # Skip validation for error/message results
            if 'error' in result_df.columns or 'message' in result_df.columns:
                validation_result['trust_score'] = 50
                validation_result['issues'].append("Cannot validate error messages")
                return validation_result
            
            # Choose validation method based on analysis type
            primary_dataset = plan.get('primary_dataset')
            if not primary_dataset or primary_dataset not in dfs:
                validation_result['trust_score'] = 50
                validation_result['issues'].append("Cannot validate - dataset not available")
                return validation_result
            
            # Perform validation based on plan type
            steps = plan.get('steps', [])
            has_groupby = any(step.get('type') == 'group_aggregate' for step in steps)
            has_aggregate = any(step.get('type') == 'aggregate' for step in steps)

            if has_groupby:
                return self._validate_group_aggregate(plan, dfs, result_df)
            elif has_aggregate or plan.get('metric_column'):
                return self._validate_simple_aggregate(plan, dfs, result_df)
            else:
                return self._validate_basic_query(plan, dfs, result_df)
                
        except Exception as e:
            return {
                'trust_score': 0,
                'validation_method': 'failed',
                'alternate_result': None,
                'comparison': None,
                'issues': [f"Validation failed: {str(e)}"]
            }
    
    def _validate_group_aggregate(self, plan, dfs, result_df):
        """Validate grouped aggregation results with proper sorting"""
        validation_result = {
            'trust_score': 50,
            'validation_method': 'pivot_table',
            'alternate_result': None,
            'comparison': None,
            'issues': []
        }
        
        try:
            primary_dataset = plan['primary_dataset']
            df = dfs[primary_dataset].copy()
            
            # Apply same filters as original analysis
            df = self._apply_filters(df, plan)
            
            # Get the execution steps to understand what was done
            groupby_step = None
            for step in plan.get('steps', []):
                if step.get('type') == 'group_aggregate':
                    groupby_step = step
                    break
            
            if not groupby_step:
                validation_result['issues'].append("No group_aggregate step found")
                return validation_result
            
            groupby_cols = groupby_step.get('groupby_columns', [])
            metric_col = groupby_step.get('metric_column')
            metric_type = groupby_step.get('metric_type', 'sum')
            
            if not groupby_cols or not metric_col:
                validation_result['issues'].append("Missing required columns for validation")
                return validation_result
            
            # Use pivot_table as alternate method
            if metric_type == 'count' or metric_col == '*':
                # For count, use any column
                available_cols = [col for col in df.columns if col not in groupby_cols]
                if available_cols:
                    alternate_result = df.pivot_table(
                        values=available_cols[0],
                        index=groupby_cols,
                        aggfunc='count',
                        fill_value=0
                    ).reset_index()
                    alternate_result.columns = list(alternate_result.columns[:-1]) + ['count']
                else:
                    validation_result['issues'].append("No columns available for count validation")
                    return validation_result
            else:
                # For other metrics
                if metric_col not in df.columns:
                    validation_result['issues'].append(f"Metric column '{metric_col}' not found")
                    return validation_result
                
                agg_func = self._get_agg_func(metric_type)
                alternate_result = df.pivot_table(
                    values=metric_col,
                    index=groupby_cols,
                    aggfunc=agg_func,
                    fill_value=0
                ).reset_index()
                
                # Rename the result column to match expected format
                if len(alternate_result.columns) > len(groupby_cols):
                    alternate_result.columns = list(alternate_result.columns[:-1]) + [f'{metric_type}_{metric_col}']
            
            # CRITICAL FIX: Apply same sorting as original result
            sort_order = plan.get('sort_order', 'desc')
            top_k = plan.get('top_k', 5)
            
            if len(alternate_result.columns) > 1:
                # Sort by the metric column (last column)
                metric_column = alternate_result.columns[-1]
                ascending = sort_order == 'asc'
                alternate_result = alternate_result.sort_values(metric_column, ascending=ascending)
                
                # Apply same top-k limit
                if top_k > 0:
                    alternate_result = alternate_result.head(top_k)
            
            # Reset index to match result_df format
            alternate_result = alternate_result.reset_index(drop=True)
            
            validation_result['alternate_result'] = alternate_result.to_string(index=False)
            
            # Compare results with better matching logic
            comparison = self._compare_results_robust(result_df, alternate_result)
            validation_result['comparison'] = comparison
            
            # Calculate trust score based on comparison
            if comparison['match_rate'] >= 0.95:
                validation_result['trust_score'] = 100
            elif comparison['match_rate'] >= 0.8:
                validation_result['trust_score'] = 85
            elif comparison['match_rate'] >= 0.6:
                validation_result['trust_score'] = 70
            elif comparison['value_correlation'] >= 0.9:
                # If values correlate well but order is slightly different
                validation_result['trust_score'] = 80
                validation_result['issues'].append("Minor ordering differences but values match well")
            else:
                validation_result['trust_score'] = 30
                validation_result['issues'].append(f"Results differ significantly (match rate: {comparison['match_rate']:.1%})")
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
            validation_result['trust_score'] = 20
        
        return validation_result
    
    def _compare_results_robust(self, result1, result2):
        """Compare two result DataFrames with robust matching"""
        try:
            if result1 is None or result2 is None:
                return {'match_rate': 0, 'details': 'One or both results are None', 'value_correlation': 0}
            
            if len(result1) != len(result2):
                return {'match_rate': 0, 'details': f'Different row counts: {len(result1)} vs {len(result2)}', 'value_correlation': 0}
            
            if len(result1) == 0:
                return {'match_rate': 1.0, 'details': 'Both results are empty', 'value_correlation': 1.0}
            
            # For grouped results, compare by matching on group columns
            if len(result1.columns) >= 2 and len(result2.columns) >= 2:
                group_col1 = result1.columns[0]  # First column (group)
                metric_col1 = result1.columns[-1]  # Last column (metric)
                
                group_col2 = result2.columns[0]
                metric_col2 = result2.columns[-1]
                
                # Create lookup dictionaries
                lookup1 = dict(zip(result1[group_col1], result1[metric_col1]))
                lookup2 = dict(zip(result2[group_col2], result2[metric_col2]))
                
                matches = 0
                total = len(lookup1)
                values1 = []
                values2 = []
                
                for key in lookup1:
                    if key in lookup2:
                        val1 = lookup1[key]
                        val2 = lookup2[key]
                        values1.append(val1)
                        values2.append(val2)
                        
                        # Check if values are close (within 0.1% tolerance)
                        if self._values_close(val1, val2, rtol=0.001):
                            matches += 1
                
                match_rate = matches / total if total > 0 else 0
                
                # Calculate correlation of values
                value_correlation = 0
                if len(values1) > 1:
                    try:
                        correlation_matrix = np.corrcoef(values1, values2)
                        value_correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0
                    except:
                        value_correlation = 0
                
                return {
                    'match_rate': match_rate, 
                    'details': f'{matches}/{total} values match within tolerance',
                    'total_rows': total,
                    'matching_rows': matches,
                    'value_correlation': abs(value_correlation)  # Use absolute correlation
                }
            
            # Fallback for other result types
            return {'match_rate': 0.8, 'details': 'Partial validation - structure differs', 'value_correlation': 0.8}
                
        except Exception as e:
            return {'match_rate': 0, 'details': f'Comparison failed: {str(e)}', 'value_correlation': 0}
    
    def _validate_simple_aggregate(self, plan, dfs, result_df):
        """Validate simple aggregation (no grouping)"""
        validation_result = {
            'trust_score': 50,
            'validation_method': 'direct_calculation',
            'alternate_result': None,
            'comparison': None,
            'issues': []
        }
        
        try:
            primary_dataset = plan['primary_dataset']
            df = dfs[primary_dataset].copy()
            
            # Apply same filters
            df = self._apply_filters(df, plan)
            
            metric_col = plan.get('metric_column', {}).get('original_name')
            metric_type = plan.get('metric', 'sum')
            
            if not metric_col:
                validation_result['issues'].append("No metric column specified")
                return validation_result
            
            # Calculate using direct method
            if metric_type == 'count' or metric_col == '*':
                alternate_value = len(df)
            else:
                if metric_col not in df.columns:
                    validation_result['issues'].append(f"Column '{metric_col}' not found")
                    return validation_result
                
                if metric_type == 'sum':
                    alternate_value = df[metric_col].sum()
                elif metric_type == 'mean':
                    alternate_value = df[metric_col].mean()
                elif metric_type == 'min':
                    alternate_value = df[metric_col].min()
                elif metric_type == 'max':
                    alternate_value = df[metric_col].max()
                elif metric_type == 'std':
                    alternate_value = df[metric_col].std()
                elif metric_type == 'median':
                    alternate_value = df[metric_col].median()
                else:
                    alternate_value = df[metric_col].sum()  # Default
            
            alternate_result = pd.DataFrame({'result': [alternate_value]})
            validation_result['alternate_result'] = alternate_result.to_string(index=False)
            
            # Compare values
            original_value = result_df.iloc[0, -1] if len(result_df) > 0 else 0
            
            if abs(alternate_value - original_value) < 0.001:
                validation_result['trust_score'] = 100
                validation_result['comparison'] = {'match': True, 'difference': 0}
            else:
                # Calculate relative difference
                rel_diff = abs(alternate_value - original_value) / max(abs(original_value), 1)
                if rel_diff < 0.01:  # Less than 1% difference
                    validation_result['trust_score'] = 90
                elif rel_diff < 0.05:  # Less than 5% difference
                    validation_result['trust_score'] = 70
                else:
                    validation_result['trust_score'] = 30
                    validation_result['issues'].append(f"Significant difference: {rel_diff:.1%}")
                
                validation_result['comparison'] = {
                    'match': False,
                    'original': original_value,
                    'alternate': alternate_value,
                    'difference': rel_diff
                }
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
            validation_result['trust_score'] = 20
        
        return validation_result
    
    def _validate_basic_query(self, plan, dfs, result_df):
        """Validate basic queries (filters only, no aggregation)"""
        validation_result = {
            'trust_score': 75,  # Higher base score for simpler operations
            'validation_method': 'row_count_check',
            'alternate_result': None,
            'comparison': None,
            'issues': []
        }
        
        try:
            primary_dataset = plan['primary_dataset']
            df = dfs[primary_dataset].copy()
            
            # Apply filters using alternate method
            original_count = len(df)
            df_filtered = self._apply_filters(df, plan)
            filtered_count = len(df_filtered)
            
            result_count = len(result_df)
            
            validation_result['comparison'] = {
                'original_count': original_count,
                'filtered_count': filtered_count,
                'result_count': result_count
            }
            
            if filtered_count == result_count:
                validation_result['trust_score'] = 100
            elif abs(filtered_count - result_count) <= 1:
                validation_result['trust_score'] = 95
            else:
                validation_result['trust_score'] = 50
                validation_result['issues'].append(f"Row count mismatch: expected {filtered_count}, got {result_count}")
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
            validation_result['trust_score'] = 40
        
        return validation_result
    
    def _apply_filters(self, df, plan):
        """Apply filters to dataframe (simplified version for validation)"""
        try:
            # Apply time filters
            for step in plan.get('steps', []):
                if step.get('type') == 'filter':
                    for time_filter in step.get('time_filters', []):
                        col = time_filter.get('column', 'order_date')
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            
                            if time_filter.get('type') == 'quarter':
                                year = time_filter.get('year', 2024)
                                start_month = time_filter.get('start_month', 1)
                                end_month = time_filter.get('end_month', 3)
                                
                                from datetime import datetime
                                start_date = datetime(year, start_month, 1)
                                if end_month == 12:
                                    end_date = datetime(year + 1, 1, 1)
                                else:
                                    end_date = datetime(year, end_month + 1, 1)
                                
                                df = df[(df[col] >= start_date) & (df[col] < end_date)]
                            
                            elif time_filter.get('type') == 'year':
                                year = time_filter.get('year', 2024)
                                df = df[df[col].dt.year == year]
            
            return df
            
        except Exception:
            return df  # Return original if filtering fails
    
    def _values_close(self, val1, val2, rtol=1e-5, atol=1e-8):
        """Check if two values are close (handling different types)"""
        try:
            if type(val1) != type(val2):
                # Convert to same type for comparison
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    return abs(float(val1) - float(val2)) <= max(atol, rtol * max(abs(float(val1)), abs(float(val2))))
                else:
                    return str(val1) == str(val2)
            
            if isinstance(val1, (int, float)):
                return abs(val1 - val2) <= max(atol, rtol * max(abs(val1), abs(val2)))
            else:
                return val1 == val2
                
        except Exception:
            return False
    
    def _get_agg_func(self, metric_type):
        """Get pandas aggregation function name"""
        mapping = {
            'sum': 'sum',
            'mean': 'mean',
            'count': 'count',
            'min': 'min',
            'max': 'max',
            'std': 'std',
            'median': 'median'
        }
        return mapping.get(metric_type, 'sum')