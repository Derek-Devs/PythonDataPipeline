# Advanced Data Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)

A comprehensive, object-oriented data analysis pipeline for professional data scientists and analysts. This all-in-one Python script provides end-to-end capabilities for data extraction, cleaning, analysis, visualization, and reporting.

## 🚀 Features

- **Multi-source Data Extraction**: Import data from CSV, JSON, APIs, and databases
- **Robust Data Cleaning**: Comprehensive preprocessing with configurable options
- **Exploratory Data Analysis**: Automated statistical summaries and data profiling
- **Advanced Statistical Analysis**: T-tests, ANOVA, chi-square, correlation, and regression
- **Visualization Generation**: Multiple chart types with customization options
- **Automated Reporting**: Generate comprehensive HTML reports with findings
- **Multiple Export Formats**: Share results as CSV, Excel, JSON, or HTML

## 📋 Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- scikit-learn
- requests (for API data extraction)

## 🔧 Installation

```bash
# Create a requirements.txt file with the following content:
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.12.0
scikit-learn>=0.24.0
requests>=2.25.0

# Install dependencies
pip install -r requirements.txt

# Download the DataAnalysisPipeline.py file
# No installation needed - just import and use!
```

## 📊 Usage Examples

### Basic Pipeline Setup

```python
from DataAnalysisPipeline import DataAnalysisPipeline

# Initialize the pipeline
pipeline = DataAnalysisPipeline()

# Extract data from multiple sources
data_sources = {
    'sales_data': {
        'type': 'csv',
        'path': 'data/sales_data.csv'
    },
    'customer_info': {
        'type': 'json',
        'path': 'data/customers.json'
    }
}
pipeline.extract_data(data_sources)
```

### Data Cleaning

```python
# Define cleaning steps
cleaning_steps = {
    'sales_data': [
        {'type': 'drop_duplicates'},
        {'type': 'drop_na', 'subset': ['date', 'product', 'revenue']},
        {'type': 'fill_na', 'columns': {'quantity': 0}},
        {'type': 'convert_types', 'conversions': {'revenue': 'float', 'quantity': 'int'}}
    ]
}

# Apply cleaning steps
pipeline.clean_data(cleaning_steps)
```

### Statistical Analysis

```python
# Perform exploratory data analysis
pipeline.perform_eda('sales_data')

# Run statistical tests
analysis_config = {
    'revenue_by_region': {
        'type': 'anova',
        'dataset': 'sales_data',
        'group_column': 'region',
        'value_column': 'revenue'
    },
    'sales_correlation': {
        'type': 'correlation',
        'dataset': 'sales_data',
        'columns': ['revenue', 'quantity', 'customer_satisfaction']
    }
}
pipeline.perform_statistical_analysis(analysis_config)
```

### Visualization and Reporting

```python
# Create visualizations
visualization_config = {
    'monthly_sales': {
        'type': 'line',
        'dataset': 'sales_data',
        'x_column': 'month',
        'y_column': 'revenue',
        'title': 'Monthly Sales Performance',
        'output_file': 'visualizations/monthly_sales.png'
    },
    'region_comparison': {
        'type': 'bar',
        'dataset': 'sales_data',
        'x_column': 'region',
        'y_column': 'revenue',
        'title': 'Revenue by Region',
        'output_file': 'visualizations/region_comparison.png'
    }
}
pipeline.create_visualizations(visualization_config)

# Generate a comprehensive report
report_config = {
    'title': 'Sales Analysis Report - Q1 2024',
    'introduction': 'This report analyzes sales performance across regions and products.',
    'sections': [
        {
            'type': 'datasets',
            'title': 'Data Overview',
            'datasets': ['sales_data']
        },
        {
            'type': 'eda',
            'title': 'Exploratory Analysis',
            'dataset': 'sales_data'
        },
        {
            'type': 'statistical_analysis',
            'title': 'Statistical Findings',
            'analyses': ['revenue_by_region', 'sales_correlation']
        },
        {
            'type': 'visualizations',
            'title': 'Visual Insights',
            'visualizations': ['monthly_sales', 'region_comparison']
        },
        {
            'type': 'conclusion',
            'title': 'Conclusion',
            'content': 'Based on our analysis, the Western region shows the highest revenue growth...'
        }
    ]
}
pipeline.generate_report(report_config, 'reports/q1_sales_analysis.html')
```

## 📝 Configuration Options

The pipeline uses configuration dictionaries to control its behavior. Here are some key configuration options:

### Data Source Types
- `csv`: CSV files with customizable parameters
- `json`: JSON files or arrays
- `api`: RESTful API endpoints with authentication
- `database`: SQL database connections

### Cleaning Operations
- `drop_duplicates`: Remove duplicate rows
- `drop_na`: Remove rows with missing values
- `fill_na`: Fill missing values (mean, median, mode, or custom)
- `rename_columns`: Rename dataframe columns
- `convert_types`: Convert column data types
- `filter_rows`: Filter based on conditions
- `transform_column`: Apply transformations (log, sqrt, standardize, etc.)

### Statistical Tests
- `ttest`: Independent t-tests
- `anova`: One-way ANOVA with post-hoc tests
- `chi_square`: Chi-square test of independence
- `correlation`: Pearson, Spearman, or Kendall correlations
- `regression`: OLS regression analysis

### Visualization Types
- `histogram`: Distribution visualization
- `scatter`: Relationship between variables
- `bar`: Category comparisons
- `line`: Time series or trend analysis
- `heatmap`: Correlation visualization
- `box`: Distribution and outlier analysis
- `pie`: Composition visualization

## 🔍 Advanced Usage

### Using External Configuration Files

```python
import json

# Load configuration from JSON file
with open('config/analysis_config.json', 'r') as f:
    config = json.load(f)

# Initialize pipeline
pipeline = DataAnalysisPipeline()

# Run the complete pipeline using configuration
pipeline.extract_data(config['data_sources'])
pipeline.clean_data(config['cleaning_steps'])
pipeline.perform_statistical_analysis(config['analyses'])
pipeline.create_visualizations(config['visualizations'])
pipeline.generate_report(config['report'], config['output_file'])
```

### Error Handling and Logging

The pipeline provides comprehensive logging to help debug issues:

```python
# Logs are automatically saved to data_pipeline.log
# You can check specific log levels:
import logging
logging.getLogger('data_pipeline').setLevel(logging.DEBUG)
```

## 🚩 Future Improvements

A professional implementation would split this single file into a proper Python package structure:

```
data_analysis_pipeline/
├── __init__.py
├── pipeline.py          # Main pipeline class
├── extractors/          # Data extraction modules
├── transformers/        # Data cleaning operations
├── analyzers/           # Statistical analysis modules
├── visualizers/         # Visualization generators
└── reporters/           # Report generation utilities
```

This modular architecture would improve maintainability and extensibility, making it easier to:
- Add new data source types
- Implement additional statistical tests
- Create new visualization types
- Extend reporting capabilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*This data analysis pipeline was created as part of a professional portfolio project demonstrating advanced Python and data analysis skills.*
