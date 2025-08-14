import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_sample_data():
    """Load sample datasets for demo purposes"""
    dfs = {}
    schemas = {}
    
    # Sample ecommerce orders data
    np.random.seed(42)  # For consistent demo data
    
    ecommerce_data = {
        'order_id': [f'ORD_{i:05d}' for i in range(1, 101)],
        'customer_id': [f'CUST_{np.random.randint(1, 51):03d}' for _ in range(100)],
        'order_date': pd.date_range('2024-01-01', periods=100, freq='3D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'product': np.random.choice(['Widget A', 'Widget B', 'Widget C', 'Gadget X', 'Gadget Y'], 100),
        'category': np.random.choice(['Electronics', 'Home', 'Office', 'Sports'], 100),
        'qty': np.random.randint(1, 10, 100),
        'price': np.round(np.random.uniform(10, 500, 100), 2),
        'refund': np.random.choice([0, 0, 0, 0, 1], 100)  # 20% refund rate
    }
    
    ecommerce_df = pd.DataFrame(ecommerce_data)
    ecommerce_df['total_sales'] = ecommerce_df['qty'] * ecommerce_df['price']
    
    # Sample marketing spend data
    marketing_data = {
        'date': pd.date_range('2024-01-01', periods=90, freq='D'),
        'channel': np.random.choice(['Google Ads', 'Facebook', 'Email', 'Direct Mail'], 90),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 90),
        'spend': np.round(np.random.uniform(100, 2000, 90), 2)
    }
    
    marketing_df = pd.DataFrame(marketing_data)
    
    # Store dataframes
    dfs['ecommerce_orders.csv'] = ecommerce_df
    dfs['marketing_spend.csv'] = marketing_df
    
    # Generate schemas
    schemas['ecommerce_orders.csv'] = infer_schema(ecommerce_df)
    schemas['marketing_spend.csv'] = infer_schema(marketing_df)
    
    return dfs, schemas

def infer_schema(df):
    """Infer schema information from a DataFrame"""
    schema = {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': {}
    }
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'null_pct': float(df[col].isnull().sum() / len(df) * 100),
            'unique_count': int(df[col].nunique()),
            'unique_pct': float(df[col].nunique() / len(df) * 100),
            'sample_values': []
        }
        
        # Get sample non-null values
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            sample_size = min(5, len(non_null_values))
            col_info['sample_values'] = non_null_values.sample(sample_size).tolist()
        
        # Additional stats for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'min': float(df[col].min()) if not df[col].isnull().all() else None,
                'max': float(df[col].max()) if not df[col].isnull().all() else None,
                'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                'std': float(df[col].std()) if not df[col].isnull().all() else None
            })
        
        schema['columns'][col] = col_info
    
    return schema

def format_number(num):
    """Format numbers for display"""
    if pd.isna(num):
        return 'N/A'
    
    if isinstance(num, (int, float)):
        if abs(num) >= 1_000_000:
            return f'{num/1_000_000:.1f}M'
        elif abs(num) >= 1_000:
            return f'{num/1_000:.1f}K'
        elif abs(num) < 1 and num != 0:
            return f'{num:.3f}'
        else:
            return f'{num:,.0f}'
    
    return str(num)

def safe_column_name(name):
    """Create safe column name for code generation"""
    import re
    if not isinstance(name, str):
        name = str(name)
    
    # Replace spaces and special chars with underscores
    safe_name = re.sub(r'[^\w]', '_', name)
    
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    # Ensure it starts with letter or underscore
    if safe_name and safe_name[0].isdigit():
        safe_name = f'col_{safe_name}'
    
    return safe_name or 'unnamed_column'

def detect_date_columns(df):
    """Detect potential date columns in a DataFrame"""
    date_columns = []
    
    for col in df.columns:
        # Check dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
        
        # Check column name patterns
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
            # Try to parse a sample of values
            try:
                sample = df[col].dropna().head(10)
                parsed = pd.to_datetime(sample, errors='coerce')
                if not parsed.isnull().all():
                    date_columns.append(col)
            except:
                pass
    
    return date_columns

def get_numeric_columns(df):
    """Get list of numeric columns"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    """Get list of categorical/text columns"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def truncate_text(text, max_length=50):
    """Truncate text to specified length"""
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + '...'

def validate_dataframe(df):
    """Basic validation of DataFrame"""
    issues = []
    
    if df is None:
        issues.append("DataFrame is None")
        return issues
    
    if df.empty:
        issues.append("DataFrame is empty")
        return issues
    
    # Check for completely null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"Columns with all null values: {null_cols}")
    
    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        issues.append(f"Duplicate column names: {duplicate_cols}")
    
    # Check memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if memory_mb > 100:
        issues.append(f"Large dataset: {memory_mb:.1f} MB")
    
    return issues

def load_sample_data():
    """Load sample datasets for demo purposes"""
    dfs = {}
    schemas = {}
    
    # Sample ecommerce orders data
    np.random.seed(42)  # For consistent demo data
    
    ecommerce_data = {
        'order_id': [f'ORD_{i:05d}' for i in range(1, 101)],
        'customer_id': [f'CUST_{np.random.randint(1, 51):03d}' for _ in range(100)],
        'order_date': pd.date_range('2024-01-01', periods=100, freq='3D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'product': np.random.choice(['Widget A', 'Widget B', 'Widget C', 'Gadget X', 'Gadget Y'], 100),
        'category': np.random.choice(['Electronics', 'Home', 'Office', 'Sports'], 100),
        'qty': np.random.randint(1, 10, 100),
        'price': np.round(np.random.uniform(10, 500, 100), 2),
        'refund': np.random.choice([0, 0, 0, 0, 1], 100)
    }
    
    ecommerce_df = pd.DataFrame(ecommerce_data)
    ecommerce_df['total_sales'] = ecommerce_df['qty'] * ecommerce_df['price']
    
    # Store dataframes
    dfs['ecommerce_orders.csv'] = ecommerce_df
    
    # Generate schemas
    schemas['ecommerce_orders.csv'] = infer_schema(ecommerce_df)
    
    return dfs, schemas