import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Optional

class ChartGenerator:
    """Generate matplotlib charts from analysis results"""
    
    def __init__(self):
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_chart(self, result_df, plan, chart_type='auto'):
        """Create appropriate chart based on data and plan"""
        if result_df is None or result_df.empty:
            return None
            
        try:
            # Determine chart type if auto
            if chart_type == 'auto':
                chart_type = self._auto_select_chart_type(result_df, plan)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate chart based on type
            if chart_type == 'bar':
                self._create_bar_chart(ax, result_df, plan)
            elif chart_type == 'line':
                self._create_line_chart(ax, result_df, plan)
            elif chart_type == 'pie':
                self._create_pie_chart(ax, result_df, plan)
            elif chart_type == 'scatter':
                self._create_scatter_chart(ax, result_df, plan)
            else:
                self._create_bar_chart(ax, result_df, plan)  # Default
            
            # Styling
            self._style_chart(ax, plan)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Chart generation failed: {str(e)}")
            return None
    
    def _auto_select_chart_type(self, result_df, plan):
        """Automatically select appropriate chart type"""
        num_rows = len(result_df)
        num_cols = len(result_df.columns)
        
        # Check for time series data
        date_cols = [col for col in result_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            return 'line'
        
        # Check for categorical data with small number of categories (good for pie)
        if num_rows <= 8 and plan.get('intent') == 'ranking':
            return 'pie'
        
        # Default to bar for most cases
        if plan.get('groupby') or plan.get('intent') == 'ranking':
            return 'bar'
        
        # Scatter for comparison of two numeric variables
        if num_cols >= 2 and plan.get('intent') == 'comparison':
            return 'scatter'
        
        return 'bar'
    
    def _create_bar_chart(self, ax, result_df, plan):
        """Create bar chart"""
        if len(result_df.columns) < 2:
            # Single column - create simple bar
            values = result_df.iloc[:, 0].values
            labels = [f'Item {i+1}' for i in range(len(values))]
            ax.bar(labels, values)
            ax.set_ylabel(result_df.columns[0])
        else:
            # Multiple columns - use first as labels, last as values
            labels = result_df.iloc[:, 0].astype(str).values
            values = result_df.iloc[:, -1].values
            
            # Truncate long labels
            labels = [label[:20] + '...' if len(str(label)) > 20 else str(label) for label in labels]
            
            bars = ax.bar(labels, values)
            ax.set_xlabel(result_df.columns[0])
            ax.set_ylabel(result_df.columns[-1])
            
            # Rotate labels if too many or too long
            if len(labels) > 5 or any(len(str(label)) > 10 for label in labels):
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if pd.notna(value):
                    ax.annotate(f'{value:,.0f}' if abs(value) > 1 else f'{value:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 3),
                               textcoords='offset points',
                               ha='center', va='bottom', fontsize=9)
    
    def _create_line_chart(self, ax, result_df, plan):
        """Create line chart"""
        if len(result_df.columns) < 2:
            # Single column - use index as x
            ax.plot(result_df.index, result_df.iloc[:, 0])
            ax.set_ylabel(result_df.columns[0])
        else:
            # Multiple columns
            x_vals = result_df.iloc[:, 0]
            y_vals = result_df.iloc[:, -1]
            
            ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel(result_df.columns[0])
            ax.set_ylabel(result_df.columns[-1])
            
            # Format x-axis if it's dates
            try:
                x_vals_datetime = pd.to_datetime(x_vals, errors='coerce')
                if not x_vals_datetime.isna().all():
                    ax.tick_params(axis='x', rotation=45)
            except:
                pass
        
        ax.grid(True, alpha=0.3)
    
    def _create_pie_chart(self, ax, result_df, plan):
        """Create pie chart"""
        if len(result_df.columns) < 2:
            return
        
        labels = result_df.iloc[:, 0].astype(str).values
        values = result_df.iloc[:, -1].values
        
        # Filter out zero/negative values
        mask = values > 0
        labels = labels[mask]
        values = values[mask]
        
        if len(values) == 0:
            ax.text(0.5, 0.5, 'No positive values to display', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Truncate labels
        labels = [label[:15] + '...' if len(label) > 15 else label for label in labels]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        
        # Improve text formatting
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _create_scatter_chart(self, ax, result_df, plan):
        """Create scatter plot"""
        if len(result_df.columns) < 2:
            return
        
        x_vals = result_df.iloc[:, 0]
        y_vals = result_df.iloc[:, -1]
        
        ax.scatter(x_vals, y_vals, alpha=0.7, s=60)
        ax.set_xlabel(result_df.columns[0])
        ax.set_ylabel(result_df.columns[-1])
        
        # Add trend line if both columns are numeric
        try:
            x_numeric = pd.to_numeric(x_vals, errors='coerce')
            y_numeric = pd.to_numeric(y_vals, errors='coerce')
            
            if not x_numeric.isna().any() and not y_numeric.isna().any():
                z = np.polyfit(x_numeric, y_numeric, 1)
                p = np.poly1d(z)
                ax.plot(x_numeric, p(x_numeric), "r--", alpha=0.8, linewidth=1)
        except:
            pass
        
        ax.grid(True, alpha=0.3)
    
    def _style_chart(self, ax, plan):
        """Apply consistent styling to charts"""
        # Set title
        question = plan.get('original_question', 'Analysis Results')
        title = question[:60] + '...' if len(question) > 60 else question
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Style axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Format numbers on axes
        try:
            # Y-axis formatting for large numbers
            if hasattr(ax, 'get_ylim'):
                y_max = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
                if y_max > 1000000:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
                elif y_max > 1000:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
        except:
            pass
        
        # Color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        # Apply colors to bars if it's a bar chart
        try:
            if hasattr(ax, 'patches') and ax.patches:
                for i, patch in enumerate(ax.patches):
                    patch.set_facecolor(colors[i % len(colors)])
                    patch.set_edgecolor('white')
                    patch.set_linewidth(0.5)
        except:
            pass