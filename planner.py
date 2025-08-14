from typing import Dict, List, Any
import pandas as pd

class ExecutionPlanner:
    """Create execution plans from mapped analysis requests"""
    
    def create_plan(self, mapped_plan, top_k_limit=10):
        """Create a detailed execution plan"""
        plan = {
            'steps': [],
            'primary_dataset': self._select_primary_dataset(mapped_plan),
            'metric': mapped_plan.get('metric', 'sum'),
            'metric_column': None,
            'groupby': mapped_plan.get('groupby', []),
            'filters': mapped_plan.get('filters', []),
            'time_filters': mapped_plan.get('time_filters', []),
            'top_k': min(mapped_plan.get('top_k', top_k_limit), top_k_limit),
            'sort_order': mapped_plan.get('sort_order', 'desc'),
            'intent': mapped_plan.get('intent', 'general'),
            'original_question': mapped_plan.get('original_question', '')
        }
        
        # Determine metric column
        plan['metric_column'] = self._determine_metric_column(mapped_plan)
        
        # Build execution steps
        steps = []
        
        # Step 1: Data loading
        steps.append({
            'type': 'load_data',
            'dataset': plan['primary_dataset'],
            'description': f"Load dataset: {plan['primary_dataset']}"
        })
        
        # Step 2: Apply filters
        valid_filters = []
        valid_time_filters = []
        
        # Only include filters for the primary dataset
        primary_dataset = plan['primary_dataset']
        for filter_def in plan['filters']:
            if filter_def.get('dataset') == primary_dataset:
                valid_filters.append(filter_def)
        
        # Include time filters
        for time_filter in plan['time_filters']:
            valid_time_filters.append(time_filter)
        
        if valid_filters or valid_time_filters:
            steps.append({
                'type': 'filter',
                'filters': valid_filters,
                'time_filters': valid_time_filters,
                'description': self._describe_filters(valid_filters, valid_time_filters)
            })
        
        # Step 3: Group and aggregate
        # FIXED: Separate groupby columns from metric columns properly and remove duplicates
        groupby_cols = []
        metric_col = None
        
        # Get metric column
        if plan['metric_column']:
            metric_col = plan['metric_column']['original_name']
        
        # Get groupby columns (exclude the metric column and remove duplicates)
        seen_cols = set()
        for col in plan['groupby']:
            if col and col.get('original_name'):
                col_name = col['original_name']
                # Don't group by the metric column itself and avoid duplicates
                if metric_col != col_name and col_name not in seen_cols:
                    groupby_cols.append(col_name)
                    seen_cols.add(col_name)
        
        # If we have groupby columns, create group-aggregate step
        if groupby_cols:
            if metric_col:
                steps.append({
                    'type': 'group_aggregate',
                    'groupby_columns': groupby_cols,
                    'metric_column': metric_col,
                    'metric_type': plan['metric'],
                    'description': f"Group by {', '.join(groupby_cols)} and calculate {plan['metric']} of {metric_col}"
                })
            else:
                # Group by with count
                steps.append({
                    'type': 'group_aggregate',
                    'groupby_columns': groupby_cols,
                    'metric_column': '*',
                    'metric_type': 'count',
                    'description': f"Group by {', '.join(groupby_cols)} and count rows"
                })
        elif metric_col:
            # Simple aggregation without grouping
            steps.append({
                'type': 'aggregate',
                'metric_column': metric_col,
                'metric_type': plan['metric'],
                'description': f"Calculate {plan['metric']} of {metric_col}"
            })
        
        # Step 4: Sort and limit (only if we have grouping)
        if groupby_cols and plan['top_k'] > 0:
            # Sort by the metric column or first groupby column
            sort_col = metric_col if metric_col else groupby_cols[0]
            steps.append({
                'type': 'sort_limit',
                'sort_column': sort_col,
                'sort_order': plan['sort_order'],
                'limit': plan['top_k'],
                'description': f"Sort by {sort_col} {plan['sort_order']} and take top {plan['top_k']} results"
            })
        
        plan['steps'] = steps
        return plan
    
    def _select_primary_dataset(self, mapped_plan):
        """Select the primary dataset to work with"""
        # Count column references per dataset
        dataset_counts = {}
        
        # Count columns
        for col in mapped_plan.get('columns', []):
            if col and col.get('dataset'):
                dataset = col['dataset']
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        # Count groupby columns
        for col in mapped_plan.get('groupby', []):
            if col and col.get('dataset'):
                dataset = col['dataset']
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        # Return dataset with most references
        if dataset_counts:
            return max(dataset_counts.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to first available dataset
            return None
    
    def _determine_metric_column(self, mapped_plan):
        """Determine which column to apply the metric to"""
        metric = mapped_plan.get('metric', 'sum')
        
        # For count operations, we don't need a specific column
        if metric == 'count':
            return {'original_name': '*', 'dataset': None, 'dtype': 'count'}
        
        # Look for numeric columns in the mapped columns
        for col in mapped_plan.get('columns', []):
            if col and self._is_numeric_column(col):
                return col
        
        # Look in groupby for numeric columns (these might be metrics)
        numeric_cols = []
        categorical_cols = []
        
        for col in mapped_plan.get('groupby', []):
            if col:
                if self._is_numeric_column(col):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
        
        # Prefer columns that sound like metrics
        for col in numeric_cols:
            col_name = col.get('original_name', '').lower()
            if any(metric_word in col_name for metric_word in 
                   ['sales', 'revenue', 'amount', 'price', 'total', 'value', 'cost']):
                return col
        
        # Return first numeric column
        if numeric_cols:
            return numeric_cols[0]
        
        # If no numeric columns found, return None (will use count)
        return None
    
    def _is_numeric_column(self, col_info):
        """Check if column is numeric"""
        if not col_info:
            return False
        dtype_str = str(col_info.get('dtype', '')).lower()
        return any(num_type in dtype_str for num_type in ['int', 'float', 'number'])
    
    def _describe_filters(self, filters, time_filters):
        """Create a human-readable description of filters"""
        descriptions = []
        
        for filter_def in filters:
            col = filter_def.get('column', 'unknown')
            op = filter_def.get('operator', '==')
            val = filter_def.get('value', '')
            
            op_text = {
                '==': 'equals',
                '!=': 'not equals',
                '>': 'greater than',
                '>=': 'greater than or equal to',
                '<': 'less than',
                '<=': 'less than or equal to'
            }.get(op, op)
            
            descriptions.append(f"{col} {op_text} {val}")
        
        for time_filter in time_filters:
            if time_filter.get('type') == 'quarter':
                descriptions.append(f"Q{time_filter['quarter']} {time_filter['year']}")
            elif time_filter.get('type') == 'last_days':
                descriptions.append(f"last {time_filter['days']} days")
            elif time_filter.get('type') == 'year':
                descriptions.append(f"year {time_filter['year']}")
        
        return f"Filter data where {', '.join(descriptions)}" if descriptions else "Apply filters"