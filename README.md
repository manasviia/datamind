# DataMind 🧠

**Turn natural language questions into validated data insights - built for enterprise reliability.**

---

## Live Demo

**Try it now:** (https://datamind-manasvi.streamlit.app))

**Quick Start:**
1. Click "📊 Load Demo Data"
2. Ask: *"What are the top 5 sales reps by total revenue?"*
3. See validated results with trust scores

---

## Key Features

### **Multi-Method Validation**
Every analysis result is cross-validated using alternate computation methods. Trust scores indicate reliability.

### **Local Processing** 
No API calls, no data leaves your environment. Built for enterprise security and privacy.

### **Business Intelligence**
Understands business context - maps "revenue" to "deal_value", handles SaaS metrics, detects patterns.

### **Interactive Visualizations**
Auto-generates appropriate charts with Plotly. Supports bar, line, pie, and scatter plots.

### **Professional Reporting**
Export comprehensive PDF reports with executive summaries and technical details.

### **Natural Language Interface**
Ask questions in plain English - no SQL or coding required.

---

## Architecture

```
Question → Parser → Mapper → Planner → CodeGen → Validator → Results
```

### **Core Components:**

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Parser** | Extract intent from natural language | spaCy NLP + Custom Rules |
| **Mapper** | Map terms to dataset columns | Sentence Transformers + Fuzzy Matching |
| **Planner** | Optimize execution strategy | Pandas Operation Planning |
| **CodeGen** | Generate safe pandas code | Sandboxed Code Generation |
| **Validator** | Cross-validate results | Multi-Method Verification |
| **Intelligence** | Generate business insights | Pattern Detection + Narrative Generation |

---

## Technical Details

### **Validation Pipeline**
```python
# Primary computation
result1 = df.groupby(['sales_rep'])['revenue'].sum()

# Validation computation  
result2 = df.pivot_table(values='revenue', index='sales_rep', aggfunc='sum')

# Trust score based on agreement
trust_score = 100 if results_match(result1, result2) else calculate_confidence()
```

### **Smart Column Mapping**
- **Semantic similarity:** "revenue" → "deal_value" (78% confidence)
- **Business rules:** SaaS terms, sales metrics, marketing KPIs
- **Fuzzy matching:** Handles typos and abbreviations

### **Security Model**
- Sandboxed code execution with whitelisted operations
- No file system access, no network calls
- Input sanitization prevents code injection

---

## Demo Questions

Try these with the sample data:

### **Sales Analysis**
- *"What are the top 5 products by total sales?"*
- *"Show me average revenue by region"*
- *"Which sales rep has the highest performance?"*

### **Time-Based Analysis**
- *"Monthly sales trends over time"*
- *"Q4 2024 performance vs Q3 2024"*
- *"Revenue growth by quarter"*

### **Business Intelligence**
- *"Customer lifetime value by segment"*
- *"Marketing channel ROI comparison"*
- *"Product category performance analysis"*

---

## Local Installation

### **Prerequisites**
- Python 3.9+
- 8GB RAM recommended
- 2GB disk space

### **Setup**
```bash
# Clone repository
git clone https://github.com/manasviia/datamind.git
cd datamind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run application
streamlit run app.py
```

### **Docker Installation**
```bash
# Build image
docker build -t datamind .

# Run container
docker run -p 8501:8501 datamind
```

---

## Supported Data Formats

| Format | Max Size | Notes |
|--------|----------|-------|
| **CSV** | 100MB | UTF-8 encoding recommended |
| **Excel** | 100MB | .xlsx and .xls supported |
| **Multiple Files** | 3 files max | Auto-joins on common columns |

### **Data Requirements**
- Headers in first row
- Consistent data types per column
- Date columns in recognizable format
- Numeric columns for calculations

---

## Use Cases

### **Sales Teams**
- Rep performance analysis
- Pipeline forecasting
- Territory optimization
- Quota attainment tracking

### **Marketing Teams**
- Campaign ROI analysis
- Channel effectiveness
- Lead quality assessment
- Customer acquisition cost

### **Executive Leadership**
- KPI dashboards
- Business performance review
- Strategic planning insights
- Board presentation data

### **Data Analysts**
- Ad-hoc analysis requests
- Data quality validation
- Exploratory data analysis
- Report automation

---

## Security & Privacy

### **Data Privacy**
- **Local processing:** Data never leaves your environment
- **No external APIs:** No calls to OpenAI, Google, or other services
- **Memory-only storage:** No data persisted to disk
- **Session isolation:** Each user session is independent

### **Code Security**
- **Sandboxed execution:** Only safe pandas operations allowed
- **Input validation:** Prevents code injection attacks
- **Resource limits:** Memory and CPU usage monitoring
- **Audit logging:** All operations are logged for review

---

## 📈 Performance

### **Benchmarks**
- **Small datasets** (<1MB): Sub-second response
- **Medium datasets** (1-10MB): 1-3 second response
- **Large datasets** (10-100MB): 5-15 second response
- **Validation overhead:** ~20% additional processing time

### **Optimization**
- Automatic query optimization
- Efficient pandas operations
- Memory-conscious processing
- Progressive loading for large files

---

## Trust & Validation

### **Validation Methods**
| Analysis Type | Primary Method | Validation Method | Typical Trust Score |
|---------------|----------------|-------------------|-------------------|
| Aggregation | `groupby()` | `pivot_table()` | 95-100% |
| Filtering | Boolean indexing | `query()` method | 90-100% |
| Sorting | `sort_values()` | `numpy.argsort()` | 98-100% |
| Counting | `value_counts()` | Manual iteration | 95-100% |

### **Trust Score Interpretation**
- **90-100%:** High confidence, results verified
- **70-89%:** Medium confidence, minor discrepancies
- **50-69%:** Moderate confidence, review recommended  
- **<50%:** Low confidence, manual verification needed

---

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black . && isort .

# Type checking
mypy datamind/
```

### **Areas for Contribution**
- New data source connectors
- Additional chart types
- Enhanced business rules
- Performance optimizations
- Documentation improvements

---

## Acknowledgments

- **spaCy** for natural language processing
- **Sentence Transformers** for semantic similarity
- **Streamlit** for the web interface
- **Plotly** for interactive visualizations
- **pandas** for data manipulation

---

