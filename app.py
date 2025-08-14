import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import os
from datetime import datetime

from parser import QuestionParser
from mapper import ColumnMapper
from planner import ExecutionPlanner
from codegen import CodeGenerator
from validator import ResultValidator
from viz import ChartGenerator
from report import PDFReporter
from utils import load_sample_data, infer_schema

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="DataMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.html("""
<style>
    .main {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #0ea5e9;
    }
    
    .hero-title {
        margin: 0;
        color: #1e40af;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .hero-tagline {
        margin: 0.5rem 0 0 0;
        color: #1e40af;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    .trust-high {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .trust-medium {
        background: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .trust-low {
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .insight-box {
        background: #eff6ff;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .insight-title {
        margin: 0 0 0.5rem 0;
        color: #1e40af;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .insight-text {
        margin: 0;
        color: #1e40af;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .step-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 1rem 0;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    
    .step-number {
        background: #3b82f6;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    
    .step-text {
        color: #374151;
        font-weight: 500;
    }
    
    .exec-summary {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .exec-summary ul {
        margin: 0;
        padding-left: 1rem;
    }
    
    .exec-summary li {
        margin: 0.5rem 0;
        color: #374151;
    }
    
    .warning-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #92400e;
    }
    
    .error-box {
        background: #fee2e2;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #dc2626;
    }
    
    .how-it-works {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .how-it-works h3 {
        color: #1f2937;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""")

def show_hero():
    """Hero section using proper HTML rendering"""
    st.html("""
    <div class="hero-section">
        <h1 class="hero-title">Hi, I'm DataMind 🧠</h1>
        <p class="hero-tagline">Your AI data analyst that speaks human</p>
    </div>
    """)

def show_how_it_works():
    """How it works section with proper HTML"""
    st.html("""
    <div class="how-it-works">
        <h3>How it works</h3>
        
        <div class="step-container">
            <div class="step-number">1</div>
            <div class="step-text">Upload your data (CSV or Excel)</div>
        </div>
        
        <div class="step-container">
            <div class="step-number">2</div>
            <div class="step-text">Ask questions in plain English</div>
        </div>
        
        <div class="step-container">
            <div class="step-number">3</div>
            <div class="step-text">Get validated insights with charts & reports</div>
        </div>
    </div>
    """)

def show_trust_score(trust_score):
    """Trust score with proper HTML rendering"""
    if trust_score >= 85:
        badge_class = "trust-high"
        text = f"✅ Verified ({trust_score}%)"
        desc = "High confidence - results validated through multiple methods"
    elif trust_score >= 60:
        badge_class = "trust-medium"
        text = f"⚠️ Uncertain ({trust_score}%)"
        desc = "Medium confidence - minor validation concerns"
    else:
        badge_class = "trust-low"
        text = f"❌ Review Needed ({trust_score}%)"
        desc = "Low confidence - significant validation issues"
    
    st.html(f'<div class="{badge_class}">{text}</div>')
    st.caption(desc)

def show_key_insight(insight_text, executive_summary=None):
    """Key insight with proper HTML rendering"""
    st.html(f"""
    <div class="insight-box">
        <div class="insight-title">💡 Key Insight</div>
        <div class="insight-text">{insight_text}</div>
    </div>
    """)
    
    if executive_summary:
        summary_html = '<div class="exec-summary"><strong>📋 Executive Summary:</strong><ul>'
        for point in executive_summary:
            summary_html += f'<li>{point}</li>'
        summary_html += '</ul></div>'
        st.html(summary_html)

def show_warning(message):
    """Warning message with proper HTML"""
    st.html(f'<div class="warning-box"><strong>⚠️ {message}</strong></div>')

def show_error(message):
    """Error message with proper HTML"""
    st.html(f'<div class="error-box"><strong>❌ {message}</strong></div>')

def get_question_examples(schemas):
    """Generate smart question examples based on data"""
    if not schemas:
        return [
            "What are the top 5 products by total sales?",
            "Show me average revenue by region",
            "What's the trend of sales over time?",
            "Compare performance across channels"
        ]
    
    examples = []
    for filename, schema in schemas.items():
        columns = list(schema['columns'].keys())
        
        # Find business-relevant columns
        numeric_cols = [col for col in columns if any(term in col.lower() 
                       for term in ['sales', 'revenue', 'amount', 'price', 'value', 'deal'])]
        categorical_cols = [col for col in columns if any(term in col.lower() 
                           for term in ['product', 'region', 'category', 'customer', 'rep', 'channel'])]
        
        if numeric_cols and categorical_cols:
            examples.extend([
                f"Top 5 {categorical_cols[0]} by {numeric_cols[0]}",
                f"Average {numeric_cols[0]} by {categorical_cols[0]}"
            ])
        
        # Look for time-based questions
        date_cols = [col for col in columns if any(term in col.lower() 
                    for term in ['date', 'time', 'quarter'])]
        if date_cols and numeric_cols:
            examples.append(f"Monthly trend of {numeric_cols[0]}")
    
    return examples[:4] if examples else [
        "Show me a data summary",
        "What are the main trends?",
        "Top 10 records by value"
    ]

def create_chart(result_df, plan, chart_type='auto'):
    """Create chart with Plotly or matplotlib fallback"""
    if result_df is None or result_df.empty or len(result_df.columns) < 2:
        return None, None
    
    try:
        x_col = result_df.columns[0]
        y_col = result_df.columns[-1]
        title = plan.get('original_question', 'Analysis Results')
        
        if PLOTLY_AVAILABLE:
            # Interactive Plotly chart
            if chart_type == 'line' or 'date' in x_col.lower() or 'time' in x_col.lower():
                fig = px.line(result_df, x=x_col, y=y_col, title=title, markers=True)
            elif chart_type == 'pie' and len(result_df) <= 8:
                fig = px.pie(result_df, values=y_col, names=x_col, title=title)
            else:
                fig = px.bar(result_df, x=x_col, y=y_col, title=title)
            
            fig.update_layout(
                height=400, 
                template='plotly_white',
                font_family="system-ui, -apple-system, sans-serif",
                title_font_size=16
            )
            return fig, 'plotly'
        
        else:
            # Matplotlib fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == 'line' or 'date' in x_col.lower():
                ax.plot(result_df[x_col], result_df[y_col], marker='o', linewidth=2)
                ax.grid(True, alpha=0.3)
            elif chart_type == 'pie' and len(result_df) <= 8:
                ax.pie(result_df[y_col], labels=result_df[x_col], autopct='%1.1f%%')
            else:
                bars = ax.bar(result_df[x_col], result_df[y_col])
                # Add value labels
                for bar, value in zip(bars, result_df[y_col]):
                    if pd.notna(value):
                        ax.annotate(f'{value:,.0f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                   xytext=(0, 3), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=9)
            
            ax.set_title(title, fontsize=14, pad=20)
            if chart_type != 'pie':
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            return fig, 'matplotlib'
            
    except Exception as e:
        st.error(f"Chart creation failed: {str(e)}")
        return None, None

def validate_uploaded_file(file, max_size_mb=100):
    """Validate uploaded file"""
    errors = []
    warnings = []
    
    # Check file size
    file_size_mb = len(file.getvalue()) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        errors.append(f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")
    
    # Check file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    if not any(file.name.lower().endswith(ext) for ext in allowed_extensions):
        errors.append(f"File type not supported. Use: {', '.join(allowed_extensions)}")
    
    # Reset file pointer
    file.seek(0)
    
    # Try to read preview
    try:
        if file.name.lower().endswith('.csv'):
            preview_df = pd.read_csv(file, nrows=5)
        else:
            preview_df = pd.read_excel(file, nrows=5)
        
        if preview_df.empty:
            errors.append("File appears to be empty")
        
        if len(preview_df.columns) == 0:
            errors.append("No columns detected")
        
        # Check for unnamed columns
        unnamed_cols = [col for col in preview_df.columns if 'Unnamed:' in str(col)]
        if unnamed_cols:
            warnings.append(f"Found {len(unnamed_cols)} unnamed columns - may indicate missing headers")
            
    except Exception as e:
        errors.append(f"Cannot read file: {str(e)}")
    
    file.seek(0)
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'file_size_mb': file_size_mb
    }

def process_files_enhanced(uploaded_files):
    """Enhanced file processing with validation"""
    if len(uploaded_files) > 3:
        show_error("Maximum 3 files allowed")
        return
    
    dfs = {}
    schemas = {}
    
    progress = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            # Validate file
            validation = validate_uploaded_file(file)
            
            if not validation['valid']:
                show_error(f"{file.name}: {', '.join(validation['errors'])}")
                continue
            
            if validation['warnings']:
                show_warning(f"{file.name}: {', '.join(validation['warnings'])}")
            
            progress.progress((i + 0.5) / len(uploaded_files))
            
            # Read file
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                continue
            
            # Basic validation
            if df.empty:
                show_error(f"{file.name}: File is empty")
                continue
            
            dfs[file.name] = df
            schemas[file.name] = infer_schema(df)
            
            progress.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            show_error(f"Failed to read {file.name}: {str(e)}")
    
    progress.empty()
    
    if dfs:
        st.session_state.dfs = dfs
        st.session_state.schemas = schemas
        st.success(f"✅ Successfully loaded {len(dfs)} files")
        
        # Show preview
        for filename, df in dfs.items():
            with st.expander(f"📊 {filename} ({len(df)} rows, {len(df.columns)} columns)"):
                st.dataframe(df.head(3))

# Replace the generate_enhanced_answer function in your app.py

def generate_enhanced_answer(question, result_df, plan, validation):
    """Generate business-friendly answer with insights that handles ties"""
    if result_df is None or result_df.empty:
        return "No data matches your criteria. Try adjusting your filters or rephrasing your question."
    
    if 'error' in result_df.columns or 'message' in result_df.columns:
        return result_df.iloc[0, -1] if len(result_df) > 0 else "Analysis encountered an issue."
    
    # Enhanced narrative generation
    if len(result_df) == 1:
        value = result_df.iloc[0, -1]
        if isinstance(value, (int, float)):
            formatted = f"${value:,.0f}" if any(term in str(result_df.columns[-1]).lower() 
                                              for term in ['sales', 'revenue', 'amount', 'deal', 'value']) else f"{value:,.0f}"
            return f"The result is **{formatted}**."
        return f"The result is **{value}**."
    
    elif len(result_df) > 1:
        top_item = result_df.iloc[0]
        metric_col = result_df.columns[-1]
        group_col = result_df.columns[0]
        
        if pd.api.types.is_numeric_dtype(result_df[metric_col]):
            top_value = top_item[metric_col]
            total = result_df[metric_col].sum()
            
            # Check for ties (same value)
            top_value_count = (result_df[metric_col] == top_value).sum()
            
            # Format based on column type
            if any(term in metric_col.lower() for term in ['sales', 'revenue', 'amount', 'deal', 'value']):
                formatted_top = f"${top_value:,.0f}"
                formatted_total = f"${total:,.0f}"
            else:
                formatted_top = f"{top_value:,.0f}"
                formatted_total = f"{total:,.0f}"
            
            # Handle ties intelligently
            if top_value_count > 1:
                # Multiple items tied for top
                tied_items = result_df[result_df[metric_col] == top_value][group_col].tolist()
                if top_value_count == len(result_df):
                    # All items have same value
                    narrative = f"**All {len(tied_items)} items are tied** with {formatted_top} each"
                else:
                    # Some items tied for top
                    tied_names = ", ".join([f"**{item}**" for item in tied_items[:3]])
                    if len(tied_items) > 3:
                        tied_names += f" and {len(tied_items) - 3} others"
                    narrative = f"{tied_names} are tied for the lead with {formatted_top} each"
            else:
                # Clear leader
                percentage = (top_value / total * 100) if total > 0 else 0
                narrative = f"**{top_item[group_col]}** leads with {formatted_top}"
                
                if len(result_df) > 1:
                    narrative += f", representing {percentage:.1f}% of the total {formatted_total}"
            
            # Add context about variation only if there's actual variation
            if len(result_df) >= 3 and result_df[metric_col].nunique() > 1:
                bottom_value = result_df.iloc[-1][metric_col]
                if top_value > 0 and bottom_value > 0:
                    ratio = top_value / bottom_value
                    if ratio > 2:
                        narrative += f". Significant variation: top performer is {ratio:.1f}x higher than lowest."
                    elif ratio == 1:
                        narrative += f". All items have identical values."
            elif result_df[metric_col].nunique() == 1:
                narrative += f". All items have identical values."
            
            return narrative + "."
        
        return f"Top result: **{top_item[group_col]}** with {top_item[metric_col]}."
    
    return f"Found {len(result_df)} results."

def generate_executive_summary(result_df, plan, validation):
    """Generate executive summary points with tie awareness"""
    summary_points = []
    
    if result_df is None or result_df.empty:
        return ["No actionable data found for analysis"]
    
    # Trust/confidence point
    trust_score = validation.get('trust_score', 50)
    if trust_score >= 85:
        summary_points.append("✅ High confidence results - validated through multiple methods")
    elif trust_score >= 60:
        summary_points.append("⚠️ Medium confidence results - minor validation concerns")
    else:
        summary_points.append("❌ Low confidence results - data quality issues detected")
    
    # Scale point
    if len(result_df) > 1 and pd.api.types.is_numeric_dtype(result_df.iloc[:, -1]):
        total_records = len(result_df)
        metric_col = result_df.columns[-1]
        total_value = result_df[metric_col].sum()
        unique_values = result_df[metric_col].nunique()
        
        if any(term in metric_col.lower() for term in ['sales', 'revenue', 'amount', 'deal']):
            summary_points.append(f"📊 Analysis covers {total_records} categories with ${total_value:,.0f} total value")
        else:
            summary_points.append(f"📊 Analysis covers {total_records} categories with {total_value:,.0f} total")
        
        # Add variation insight
        if unique_values == 1:
            summary_points.append("🔄 All items show identical performance - no clear leader")
        elif unique_values < total_records / 2:
            summary_points.append("🔄 Limited variation in performance - several ties detected")
    
    # Performance insight with tie awareness
    if len(result_df) > 1:
        top_item = result_df.iloc[0]
        metric_col = result_df.columns[-1]
        
        if pd.api.types.is_numeric_dtype(result_df[metric_col]):
            top_value = top_item[metric_col]
            top_value_count = (result_df[metric_col] == top_value).sum()
            
            if top_value_count > 1:
                summary_points.append(f"🤝 Multiple leaders: {top_value_count} items tied for top performance")
            else:
                summary_points.append(f"🏆 Clear leader: {top_item.iloc[0]} outperforms all others")
        else:
            summary_points.append(f"🏆 Top result: {top_item.iloc[0]} leads the analysis")
    
    # Actionability with tie consideration
    intent = plan.get('intent', '')
    if intent == 'ranking':
        if len(result_df) > 1 and pd.api.types.is_numeric_dtype(result_df.iloc[:, -1]):
            if result_df.iloc[:, -1].nunique() == 1:
                summary_points.append("🎯 Equal performance across all items - consider additional criteria")
            else:
                summary_points.append("🎯 Ranking analysis complete - focus on top performers for maximum impact")
        else:
            summary_points.append("🎯 Ranking analysis complete - focus on top performers for maximum impact")
    elif intent == 'trend':
        summary_points.append("📈 Trend analysis reveals patterns for strategic planning")
    else:
        summary_points.append("💡 Analysis provides actionable insights for decision making")
    
    return summary_points[:4]  # Keep it concise

def run_analysis(question, chart_type, top_k_limit):
    """Run analysis with enhanced error handling and progress tracking"""
    try:
        progress = st.progress(0)
        status = st.empty()
        
        # Initialize components
        status.text("🔍 Understanding your question...")
        progress.progress(0.15)
        parser = QuestionParser()
        parsed_plan = parser.parse(question)
        
        status.text("🗺️ Mapping data columns...")
        progress.progress(0.30)
        mapper = ColumnMapper(st.session_state.schemas)
        mapped_plan = mapper.map_columns(parsed_plan)
        
        status.text("📋 Creating execution plan...")
        progress.progress(0.45)
        planner = ExecutionPlanner()
        execution_plan = planner.create_plan(mapped_plan, top_k_limit)
        
        status.text("⚙️ Running analysis...")
        progress.progress(0.60)
        codegen = CodeGenerator()
        code, result_df = codegen.generate_and_execute(execution_plan, st.session_state.dfs)
        
        status.text("✅ Validating results...")
        progress.progress(0.75)
        validator = ResultValidator()
        validation_result = validator.validate(execution_plan, st.session_state.dfs, result_df)
        
        status.text("💡 Generating insights...")
        progress.progress(0.90)
        
        # Generate enhanced answer and insights
        final_answer = generate_enhanced_answer(question, result_df, execution_plan, validation_result)
        executive_summary = generate_executive_summary(result_df, execution_plan, validation_result)
        
        # Create chart
        chart_fig, chart_type_used = create_chart(result_df, execution_plan, chart_type)
        
        # Store results
        st.session_state.analysis_results = {
            'question': question,
            'plan': execution_plan,
            'code': code,
            'result_df': result_df,
            'validation': validation_result,
            'chart': chart_fig,
            'chart_type': chart_type_used,
            'answer': final_answer,
            'executive_summary': executive_summary,
            'trust_score': validation_result.get('trust_score', 50),
            'timestamp': datetime.now()
        }
        
        progress.progress(1.0)
        status.text("✅ Analysis complete!")
        
        # Clear progress indicators
        progress.empty()
        status.empty()
        
        st.success("🎉 Analysis completed successfully!")
        
    except Exception as e:
        # Enhanced error handling with business context
        error_msg = str(e).lower()
        
        if 'column' in error_msg and 'not found' in error_msg:
            show_error("Column mapping failed. Try rephrasing your question using the exact column names from your data.")
        elif 'empty' in error_msg or 'no data' in error_msg:
            show_warning("No data matches your criteria. Try removing filters or broadening your question.")
        elif 'date' in error_msg or 'time' in error_msg:
            show_warning("Date parsing issue. Try asking without specific date ranges first.")
        else:
            show_error(f"Analysis failed: {str(e)}")
        
        st.info("💡 Try a simpler question first, like 'Show me the top 10 records' to explore your data.")
        st.session_state.analysis_results = None

def show_results():
    """Display analysis results with enhanced formatting"""
    results = st.session_state.analysis_results
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Trust score
    show_trust_score(results['trust_score'])
    
    # Key insight with executive summary
    show_key_insight(results['answer'], results.get('executive_summary'))
    
    # Results and chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Data Results")
        if results['result_df'] is not None and not results['result_df'].empty:
            # Format display
            display_df = results['result_df'].copy()
            
            # Smart number formatting
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'int64']:
                    if any(word in col.lower() for word in ['sales', 'revenue', 'amount', 'price', 'deal', 'value']):
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
                    elif any(word in col.lower() for word in ['rate', 'percent']):
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
                    else:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download options
            csv = results['result_df'].to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "analysis_results.csv", "text/csv")
        else:
            st.info("No data to display")
    
    with col2:
        st.markdown("### 📈 Visualization")
        if results['chart']:
            if results['chart_type'] == 'plotly':
                st.plotly_chart(results['chart'], use_container_width=True)
            else:
                st.pyplot(results['chart'])
        else:
            st.info("No chart available for this data")
    
    # Technical details (collapsed)
    with st.expander("🔧 Technical Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Generated Code:**")
            st.code(results['code'], language='python')
        with col2:
            st.markdown("**Validation Results:**")
            validation = results['validation']
            st.write(f"**Method:** {validation.get('validation_method', 'Unknown')}")
            st.write(f"**Trust Score:** {validation.get('trust_score', 0)}%")
            if validation.get('issues'):
                st.write("**Issues:**")
                for issue in validation['issues']:
                    st.write(f"• {issue}")

def main():
    # Initialize session state
    if 'dfs' not in st.session_state:
        st.session_state.dfs = {}
    if 'schemas' not in st.session_state:
        st.session_state.schemas = {}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""

    # Hero section
    show_hero()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        chart_type = st.selectbox("Chart Type", ["auto", "bar", "line", "pie"], 
                                 help="Choose chart type or let AI decide")
        top_k_limit = st.slider("Results Limit", 3, 20, 10, 
                               help="Maximum number of results to show")
        
        if st.session_state.dfs:
            st.markdown("### 📊 Loaded Data")
            for name, df in st.session_state.dfs.items():
                clean_name = name.replace('.csv', '').replace('_', ' ').title()
                st.metric(clean_name, f"{len(df):,} rows")
        
        if st.session_state.analysis_results:
            st.markdown("### 📄 Export")
            if st.button("📋 Generate PDF Report", use_container_width=True):
                try:
                    reporter = PDFReporter()
                    pdf_bytes = reporter.generate_report(st.session_state.analysis_results)
                    st.download_button(
                        "📥 Download PDF",
                        pdf_bytes,
                        f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

    # Main content
    if not st.session_state.dfs:
        # Onboarding flow
        show_how_it_works()
        
        st.markdown("### 📁 Upload Your Data")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload up to 3 CSV or Excel files (max 100MB each)"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Load Demo Data", type="secondary", use_container_width=True):
                with st.spinner("Loading demo data..."):
                    st.session_state.dfs, st.session_state.schemas = load_sample_data()
                st.success("✅ Demo data loaded!")
                st.rerun()
        
        with col2:
            if uploaded_files and st.button("🔍 Process Files", type="primary", use_container_width=True):
                process_files_enhanced(uploaded_files)
                st.rerun()
    
    else:


        # Analysis interface
        st.markdown("### 💬 Ask Your Question")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Use session state to persist the question
            if 'current_question' not in st.session_state:
                st.session_state.current_question = ""
            
            question = st.text_area(
                "Enter your question about the data",
                value=st.session_state.current_question,
                placeholder="Example: What are the top 5 sales reps by total deal value in Q4 2024?",
                height=100,
                help="Ask questions about your data in plain English",
                key="question_input"
            )
            
            # Update session state when question changes
            if question != st.session_state.current_question:
                st.session_state.current_question = question
        
        with col2:
            st.markdown("**Try asking:**")
            examples = get_question_examples(st.session_state.schemas)
            for i, example in enumerate(examples):
                if st.button(example, key=f"example_btn_{i}", use_container_width=True):
                    st.session_state.current_question = example
                    st.rerun()
        
        # Analyze button
        if st.button("🔍 Analyze", type="primary", use_container_width=True, disabled=not st.session_state.current_question.strip()):
            if st.session_state.current_question.strip():
                run_analysis(st.session_state.current_question, chart_type, top_k_limit)
            else:
                show_warning("Please enter a question first")
        
        # Schema viewer (collapsed by default)
        with st.expander("📋 View Data Structure", expanded=False):
            for filename, schema in st.session_state.schemas.items():
                st.markdown(f"#### 📊 {filename}")
                
                # Quick stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{schema['shape'][0]:,}")
                with col2:
                    st.metric("Columns", schema['shape'][1])
                with col3:
                    st.metric("Size", f"{schema['memory_mb']:.1f} MB")
                
                # Column details
                col_data = []
                for col, info in schema['columns'].items():
                    col_data.append({
                        'Column': col,
                        'Type': str(info['dtype']),
                        'Missing': f"{info['null_pct']:.1f}%",
                        'Unique': f"{info['unique_pct']:.1f}%",
                        'Sample': str(info['sample_values'][0]) if info.get('sample_values') else 'N/A'
                    })
                
                st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)
        
        # Results section
        if st.session_state.analysis_results:
            show_results()

if __name__ == "__main__":
    main()
