import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class CodeGenerator:
    """Generate and execute pandas code from execution plans"""
    
    def __init__(self):
        # Whitelist of safe functions for exec
        self.safe_globals = {
            'pd': pd,
            'np': np,
            'datetime': datetime,
            'timedelta': timedelta,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round
        }
    
    def generate_and_execute(self, plan, dfs):
        """Generate pandas code from plan and execute it safely"""
        try:
            # Generate code
            code = self._generate_code(plan, dfs)
            
            # Execute code safely
            result_df = self._execute_code(code, dfs)
            
            return code, result_df
            
        except Exception as e:
            raise Exception(f"Code generation/execution failed: {str(e)}")
    
    def _generate_code(self, plan, dfs):
        """Generate pandas code from execution plan"""
        lines = []
        
        # Start with primary dataset
        primary_dataset = plan['primary_dataset']
        if primary_dataset not in dfs:
            raise Exception(f"Dataset {primary_dataset} not found")
        
        lines.append(f"# Analysis generated from: {plan['original_question']}")
        lines.append(f"df = dfs['{primary_dataset}'].copy()")
        lines.append("print('Loaded dataset with', len(df), 'rows')")
        lines.append("")
        
        # Add data exploration for date ranges
        lines.append("# Explore date ranges in the data")
        lines.append("date_cols = [col for col in df.columns if 'date' in col.lower()]")
        lines.append("for date_col in date_cols:")
        lines.append("    try:")
        lines.append("        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')")
        lines.append("        print(f'Date column {date_col}: {df[date_col].min()} to {df[date_col].max()}')")
        lines.append("    except:")
        lines.append("        pass")
        lines.append("")
        
        # Process each step
        for i, step in enumerate(plan['steps']):
            lines.append(f"# Step {i+1}: {step.get('description', step.get('type'))}")
            step_code = self._generate_step_code(step, plan)
            if step_code:
                lines.extend(step_code)
                lines.append(f"print('After step {i+1}:', len(df), 'rows')")
                lines.append("")
        
        # Handle empty results gracefully
        lines.append("# Handle empty results")
        lines.append("if len(df) == 0:")
        lines.append("    print('No data matches the specified criteria.')")
        lines.append("    print('Creating empty result with helpful message')")
        lines.append("    df = pd.DataFrame({'message': ['No data found for the specified criteria']})")
        lines.append("")
        
        # Final result assignment
        lines.append("result = df.copy()")
        lines.append("print('Final result:', len(result), 'rows,', len(result.columns), 'columns')")
        
        return "\n".join(lines)
    
    def _generate_step_code(self, step, plan):
        """Generate code for a specific step"""
        step_type = step['type']
        
        if step_type == 'load_data':
            return []  # Already handled in main function
        
        elif step_type == 'filter':
            return self._generate_filter_code(step, plan)
        
        elif step_type == 'group_aggregate':
            return self._generate_group_aggregate_code(step, plan)
        
        elif step_type == 'aggregate':
            return self._generate_aggregate_code(step, plan)
        
        elif step_type == 'sort_limit':
            return self._generate_sort_limit_code(step, plan)
        
        else:
            return [f"# Unknown step type: {step_type}"]
    
    def _generate_filter_code(self, step, plan):
        """Generate filtering code with better error handling"""
        lines = []
        
        # Time filters with robust date handling
        for time_filter in step.get('time_filters', []):
            time_code = self._generate_time_filter_code(time_filter, plan)
            if time_code:
                lines.extend(time_code)
        
        # Regular filters (only for same dataset)
        primary_dataset = plan.get('primary_dataset')
        for filter_def in step.get('filters', []):
            col = filter_def.get('column')
            dataset = filter_def.get('dataset')
            op = filter_def.get('operator', '==')
            value = filter_def.get('value')
            
            # Only apply filters for the same dataset
            if col and value and dataset == primary_dataset:
                safe_col = self._sanitize_column_name(col)
                
                lines.append(f"if '{safe_col}' in df.columns:")
                if op == '==':
                    if isinstance(value, str):
                        lines.append(f"    df = df[df['{safe_col}'].astype(str).str.contains('{value}', case=False, na=False)]")
                    else:
                        lines.append(f"    df = df[df['{safe_col}'] == {repr(value)}]")
                else:
                    lines.append(f"    df = df[df['{safe_col}'] {op} {repr(value)}]")
                lines.append(f"else:")
                lines.append(f"    print('Warning: Column {safe_col} not found for filtering')")
        
        return lines
    
    def _generate_time_filter_code(self, time_filter, plan):
        """Generate robust time-based filtering code"""
        lines = []
        filter_type = time_filter.get('type')
        col = time_filter.get('column', 'order_date')
        
        safe_col = self._sanitize_column_name(col)
        
        lines.append(f"# Time filter: {filter_type}")
        lines.append(f"if '{safe_col}' in df.columns:")
        lines.append(f"    print('Applying {filter_type} filter to {safe_col}')")
        lines.append(f"    # Convert to datetime")
        lines.append(f"    df['{safe_col}'] = pd.to_datetime(df['{safe_col}'], errors='coerce')")
        lines.append(f"    print('Data date range:', df['{safe_col}'].min(), 'to', df['{safe_col}'].max())")
        
        if filter_type == 'quarter':
            year = time_filter.get('year', 2024)
            quarter = time_filter.get('quarter', 1)
            start_month = time_filter.get('start_month', 1)
            end_month = time_filter.get('end_month', 3)
            
            lines.append(f"    # Filter for Q{quarter} {year} ({start_month}/1/{year} to {end_month}/30/{year})")
            lines.append(f"    start_date = datetime({year}, {start_month}, 1)")
            if end_month == 12:
                lines.append(f"    end_date = datetime({year + 1}, 1, 1)")
            else:
                lines.append(f"    end_date = datetime({year}, {end_month + 1}, 1)")
            
            lines.append(f"    print('Filtering for period:', start_date, 'to', end_date)")
            lines.append(f"    rows_before = len(df)")
            lines.append(f"    df = df[(df['{safe_col}'] >= start_date) & (df['{safe_col}'] < end_date)]")
            lines.append(f"    rows_after = len(df)")
            lines.append(f"    print(f'Time filter removed {{rows_before - rows_after}} rows, {{rows_after}} remain')")
            
            # Fallback: if no data in specific quarter, expand to full year
            lines.append(f"    if len(df) == 0:")
            lines.append(f"        print('No data found in Q{quarter} {year}. Trying full year {year}...')")
            lines.append(f"        df = dfs['{plan['primary_dataset']}'].copy()")
            lines.append(f"        df['{safe_col}'] = pd.to_datetime(df['{safe_col}'], errors='coerce')")
            lines.append(f"        df = df[df['{safe_col}'].dt.year == {year}]")
            lines.append(f"        if len(df) > 0:")
            lines.append(f"            print(f'Found {{len(df)}} rows in {year} (expanded from Q{quarter})')")
        
        elif filter_type == 'year':
            year = time_filter.get('year', 2024)
            lines.append(f"    # Filter for year {year}")
            lines.append(f"    rows_before = len(df)")
            lines.append(f"    df = df[df['{safe_col}'].dt.year == {year}]")
            lines.append(f"    rows_after = len(df)")
            lines.append(f"    print(f'Year filter: {{rows_after}} rows in {year} (removed {{rows_before - rows_after}})')")
        
        elif filter_type == 'last_days':
            days = time_filter.get('days', 30)
            lines.append(f"    # Filter for last {days} days")
            lines.append(f"    cutoff_date = datetime.now() - timedelta(days={days})")
            lines.append(f"    print('Filtering for dates after:', cutoff_date)")
            lines.append(f"    df = df[df['{safe_col}'] >= cutoff_date]")
        
        lines.append(f"else:")
        lines.append(f"    print('Warning: Date column {safe_col} not found in dataset')")
        lines.append(f"    available_cols = [col for col in df.columns if 'date' in col.lower()]")
        lines.append(f"    if available_cols:")
        lines.append(f"        print('Available date columns:', available_cols)")
        lines.append(f"    else:")
        lines.append(f"        print('No date columns found in dataset')")
        
        return lines
    
    def _generate_group_aggregate_code(self, step, plan):
        """Generate group-by and aggregation code with duplicate column handling"""
        lines = []
        
        groupby_cols = step.get('groupby_columns', [])
        metric_col = step.get('metric_column')
        metric_type = step.get('metric_type', 'sum')
        
        if not groupby_cols:
            lines.append("# No groupby columns specified")
            return lines
        
        # Clean and validate groupby columns
        lines.append("# Group by and aggregate")
        lines.append("groupby_cols = []")
        
        for col in groupby_cols:
            safe_col = self._sanitize_column_name(col)
            lines.append(f"if '{safe_col}' in df.columns:")
            lines.append(f"    groupby_cols.append('{safe_col}')")
            lines.append(f"else:")
            lines.append(f"    print('Warning: Groupby column {safe_col} not found')")
        
        lines.append("")
        lines.append("if groupby_cols and len(df) > 0:")
        
        if metric_col == '*' or metric_type == 'count' or not metric_col:
            # Count rows
            lines.append("    # Count rows by group")
            lines.append("    df_grouped = df.groupby(groupby_cols).size().reset_index(name='count')")
            lines.append("    df = df_grouped")
        else:
            # Aggregate specific column
            safe_metric = self._sanitize_column_name(metric_col)
            agg_func = self._get_pandas_agg_func(metric_type)
            
            lines.append(f"    if '{safe_metric}' in df.columns:")
            lines.append(f"        # Calculate {metric_type} of {safe_metric}")
            lines.append(f"        df_grouped = df.groupby(groupby_cols)['{safe_metric}'].{agg_func}().reset_index()")
            lines.append(f"        # Rename result column for clarity")
            lines.append(f"        result_col_name = '{metric_type}_{safe_metric}'")
            lines.append(f"        # Handle potential duplicate column names")
            lines.append(f"        if result_col_name in df_grouped.columns:")
            lines.append(f"            result_col_name = '{metric_type}_{safe_metric}_result'")
            lines.append(f"        df_grouped.rename(columns={{'{safe_metric}': result_col_name}}, inplace=True)")
            lines.append(f"        df = df_grouped")
            lines.append(f"    else:")
            lines.append(f"        print('Warning: Metric column {safe_metric} not found, using count instead')")
            lines.append(f"        df_grouped = df.groupby(groupby_cols).size().reset_index(name='count')")
            lines.append(f"        df = df_grouped")
        
        lines.append("elif len(df) == 0:")
        lines.append("    print('No data to group - dataset is empty after filtering')")
        lines.append("else:")
        lines.append("    print('Error: No valid groupby columns found')")
        lines.append("    df = pd.DataFrame({'error': ['No valid groupby columns']})")
        
        return lines
    
    def _generate_aggregate_code(self, step, plan):
        """Generate simple aggregation code (no grouping)"""
        lines = []
        
        metric_col = step.get('metric_column')
        metric_type = step.get('metric_type', 'sum')
        
        lines.append("# Simple aggregation (no grouping)")
        lines.append("if len(df) > 0:")
        
        if metric_col == '*' or metric_type == 'count' or not metric_col:
            lines.append("    result_value = len(df)")
            lines.append("    result_name = 'count'")
        else:
            safe_metric = self._sanitize_column_name(metric_col)
            agg_func = self._get_pandas_agg_func(metric_type)
            
            lines.append(f"    if '{safe_metric}' in df.columns:")
            lines.append(f"        result_value = df['{safe_metric}'].{agg_func}()")
            lines.append(f"        result_name = '{metric_type}_{safe_metric}'")
            lines.append("    else:")
            lines.append("        result_value = len(df)")
            lines.append("        result_name = 'count'")
        
        lines.append("    df = pd.DataFrame({result_name: [result_value]})")
        lines.append("else:")
        lines.append("    df = pd.DataFrame({'message': ['No data to aggregate']})")
        
        return lines
    
    def _generate_sort_limit_code(self, step, plan):
        """Generate sorting and limiting code"""
        lines = []
        
        sort_col = step.get('sort_column')
        sort_order = step.get('sort_order', 'desc')
        limit = step.get('limit', 10)
        
        lines.append("# Sort and limit results")
        lines.append("if len(df) > 0 and 'message' not in df.columns and 'error' not in df.columns:")
        
        if sort_col:
            safe_sort_col = self._sanitize_column_name(sort_col)
            ascending = sort_order == 'asc'
            
            lines.append("    # Find the best column to sort by")
            lines.append("    sort_column = None")
            lines.append("    ")
            lines.append("    # Look for exact match first")
            lines.append(f"    if '{safe_sort_col}' in df.columns:")
            lines.append(f"        sort_column = '{safe_sort_col}'")
            lines.append("    else:")
            lines.append("        # Look for columns containing the sort term")
            lines.append("        for col in df.columns:")
            lines.append(f"            if '{safe_sort_col}'.lower() in col.lower():")
            lines.append("                sort_column = col")
            lines.append("                break")
            lines.append("    ")
            lines.append("    if sort_column:")
            lines.append(f"        df = df.sort_values(sort_column, ascending={ascending})")
            lines.append(f"        print('Sorted by', sort_column, '({sort_order})')")
            lines.append("    else:")
            lines.append("        # Fall back to sorting by last column (usually the metric)")
            lines.append("        if len(df.columns) > 0:")
            lines.append(f"            df = df.sort_values(df.columns[-1], ascending={ascending})")
            lines.append(f"            print('Sorted by', df.columns[-1], '({sort_order}) - fallback')")
        
        if limit > 0:
            lines.append(f"    # Take top {limit} results")
            lines.append(f"    df = df.head({limit})")
        
        lines.append("else:")
        lines.append("    print('Skipping sort/limit - no valid data to sort')")
        
        return lines
    
    def _sanitize_column_name(self, col_name):
        """Sanitize column name to prevent code injection"""
        if not isinstance(col_name, str):
            return str(col_name)
        
        # Remove dangerous characters, keep alphanumeric, spaces, underscores, hyphens
        sanitized = re.sub(r'[^\w\s-]', '', str(col_name))
        return sanitized.strip()
    
    def _get_pandas_agg_func(self, metric_type):
        """Map metric type to pandas aggregation function"""
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
    
    def _execute_code(self, code, dfs):
        """Safely execute generated pandas code"""
        try:
            # Create execution environment
            exec_globals = self.safe_globals.copy()
            exec_globals['dfs'] = dfs
            
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Return result
            if 'result' in exec_locals:
                result = exec_locals['result']
                if isinstance(result, pd.DataFrame):
                    return result
                else:
                    # Convert single values to DataFrame
                    return pd.DataFrame({'result': [result]})
            else:
                return pd.DataFrame({'error': ['No result generated']})
                
        except Exception as e:
            print(f"EXECUTION ERROR: {e}")
            print("Code that failed:")
            print(code)
            raise Exception(f"Code execution failed: {str(e)}")
