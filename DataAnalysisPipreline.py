import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import requests
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

class DataAnalysisPipeline:
    """
    A comprehensive data analysis pipeline that demonstrates advanced data analysis skills.
    
    This pipeline handles:
    - Data extraction from multiple sources (APIs, CSV, JSON)
    - Data cleaning and preprocessing
    - Exploratory data analysis
    - Statistical analysis
    - Data visualization
    - Reporting
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the data analysis pipeline.
        
        Args:
            config_path (str, optional): Path to configuration file. 
                                        Defaults to None.
        """
        # Set up logging
        self.setup_logging()
        
        # Load configuration if provided
        self.config = {}
        if config_path:
            self.load_config(config_path)
        
        # Initialize data storage
        self.raw_data = {}
        self.clean_data = {}
        self.analysis_results = {}
        self.visualizations = {}
        
        self.logger.info("Data Analysis Pipeline initialized")
    
    def setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger("data_pipeline")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler("data_pipeline.log")
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
    
    def load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        try:
            with open(config_path, 'r') as file:
                self.config = json.load(file)
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def extract_data(self, sources):
        """
        Extract data from multiple sources.
        
        Args:
            sources (dict): Dictionary containing data source configurations.
                Each entry should have a 'type' key and source-specific parameters.
        
        Returns:
            bool: True if extraction was successful, False otherwise.
        """
        self.logger.info("Starting data extraction...")
        
        try:
            for source_name, source_config in sources.items():
                self.logger.info(f"Extracting data from {source_name}")
                
                source_type = source_config.get('type', '').lower()
                
                if source_type == 'csv':
                    self.raw_data[source_name] = self._extract_from_csv(source_config)
                elif source_type == 'json':
                    self.raw_data[source_name] = self._extract_from_json(source_config)
                elif source_type == 'api':
                    self.raw_data[source_name] = self._extract_from_api(source_config)
                elif source_type == 'database':
                    self.raw_data[source_name] = self._extract_from_database(source_config)
                else:
                    self.logger.warning(f"Unsupported source type: {source_type}")
                    continue
                
                self.logger.info(f"Successfully extracted data from {source_name}")
            
            self.logger.info("Data extraction completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during data extraction: {str(e)}")
            return False
    
    def _extract_from_csv(self, config):
        """Extract data from a CSV file."""
        file_path = config.get('path', '')
        params = config.get('params', {})
        
        self.logger.info(f"Reading CSV file: {file_path}")
        return pd.read_csv(file_path, **params)
    
    def _extract_from_json(self, config):
        """Extract data from a JSON file."""
        file_path = config.get('path', '')
        
        self.logger.info(f"Reading JSON file: {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        if config.get('to_dataframe', True):
            return pd.DataFrame(data)
        return data
    
    def _extract_from_api(self, config):
        """Extract data from an API."""
        url = config.get('url', '')
        method = config.get('method', 'GET')
        headers = config.get('headers', {})
        params = config.get('params', {})
        data = config.get('data', None)
        
        self.logger.info(f"Making {method} request to {url}")
        
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, params=params, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        
        if config.get('to_dataframe', True):
            return pd.DataFrame(response.json())
        return response.json()
    
    def _extract_from_database(self, config):
        """Extract data from a database using SQL."""
        # This would normally use SQLAlchemy or another DB connector
        # For this example, we'll just log the attempt
        self.logger.info("Database extraction not implemented in this demo")
        # Simulate data for demonstration
        return pd.DataFrame({'placeholder': [1, 2, 3]})
    
    def clean_data(self, cleaning_steps):
        """
        Clean and preprocess the extracted data.
        
        Args:
            cleaning_steps (dict): Dictionary of cleaning steps to apply to each dataset.
        
        Returns:
            bool: True if cleaning was successful, False otherwise.
        """
        self.logger.info("Starting data cleaning...")
        
        try:
            for data_name, steps in cleaning_steps.items():
                if data_name not in self.raw_data:
                    self.logger.warning(f"Dataset {data_name} not found in raw data")
                    continue
                
                df = self.raw_data[data_name].copy()
                self.logger.info(f"Cleaning dataset {data_name}")
                
                for step in steps:
                    step_type = step.get('type', '').lower()
                    
                    if step_type == 'drop_duplicates':
                        df = self._drop_duplicates(df, step)
                    elif step_type == 'drop_na':
                        df = self._drop_na(df, step)
                    elif step_type == 'fill_na':
                        df = self._fill_na(df, step)
                    elif step_type == 'rename_columns':
                        df = self._rename_columns(df, step)
                    elif step_type == 'convert_types':
                        df = self._convert_types(df, step)
                    elif step_type == 'filter_rows':
                        df = self._filter_rows(df, step)
                    elif step_type == 'transform_column':
                        df = self._transform_column(df, step)
                    elif step_type == 'custom_function':
                        df = self._apply_custom_function(df, step)
                    else:
                        self.logger.warning(f"Unsupported cleaning step type: {step_type}")
                
                self.clean_data[data_name] = df
                self.logger.info(f"Dataset {data_name} cleaned successfully")
            
            self.logger.info("Data cleaning completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            return False
    
    def _drop_duplicates(self, df, config):
        """Drop duplicate rows from DataFrame."""
        subset = config.get('subset', None)
        keep = config.get('keep', 'first')
        
        return df.drop_duplicates(subset=subset, keep=keep)
    
    def _drop_na(self, df, config):
        """Drop rows with NA values from DataFrame."""
        subset = config.get('subset', None)
        how = config.get('how', 'any')
        
        return df.dropna(subset=subset, how=how)
    
    def _fill_na(self, df, config):
        """Fill NA values in DataFrame."""
        columns = config.get('columns', {})
        
        for col, value in columns.items():
            if col in df.columns:
                if value == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif value == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif value == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(value)
        
        return df
    
    def _rename_columns(self, df, config):
        """Rename columns in DataFrame."""
        columns = config.get('columns', {})
        return df.rename(columns=columns)
    
    def _convert_types(self, df, config):
        """Convert column data types."""
        type_conversions = config.get('conversions', {})
        
        for col, dtype in type_conversions.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
        
        return df
    
    def _filter_rows(self, df, config):
        """Filter rows based on condition."""
        column = config.get('column', '')
        condition = config.get('condition', '')
        value = config.get('value', None)
        
        if not column or not condition or value is None:
            return df
        
        if condition == 'eq':
            return df[df[column] == value]
        elif condition == 'ne':
            return df[df[column] != value]
        elif condition == 'gt':
            return df[df[column] > value]
        elif condition == 'lt':
            return df[df[column] < value]
        elif condition == 'gte':
            return df[df[column] >= value]
        elif condition == 'lte':
            return df[df[column] <= value]
        elif condition == 'in':
            return df[df[column].isin(value)]
        elif condition == 'not_in':
            return df[~df[column].isin(value)]
        else:
            self.logger.warning(f"Unsupported filter condition: {condition}")
            return df
    
    def _transform_column(self, df, config):
        """Apply transformation to a column."""
        column = config.get('column', '')
        transform = config.get('transform', '')
        new_column = config.get('new_column', column)
        
        if not column or not transform:
            return df
        
        if column not in df.columns:
            self.logger.warning(f"Column {column} not found in DataFrame")
            return df
        
        if transform == 'log':
            df[new_column] = np.log(df[column])
        elif transform == 'sqrt':
            df[new_column] = np.sqrt(df[column])
        elif transform == 'square':
            df[new_column] = df[column] ** 2
        elif transform == 'standardize':
            df[new_column] = (df[column] - df[column].mean()) / df[column].std()
        elif transform == 'normalize':
            min_val = df[column].min()
            max_val = df[column].max()
            df[new_column] = (df[column] - min_val) / (max_val - min_val)
        else:
            self.logger.warning(f"Unsupported transformation: {transform}")
        
        return df
    
    def _apply_custom_function(self, df, config):
        """Apply a custom function to the DataFrame."""
        function_name = config.get('function', '')
        
        if function_name == 'example_function':
            # This is just a placeholder for demonstration
            return df
        else:
            self.logger.warning(f"Custom function {function_name} not implemented")
            return df
    
    def perform_eda(self, dataset_name, config=None):
        """
        Perform exploratory data analysis on a dataset.
        
        Args:
            dataset_name (str): Name of the dataset to analyze.
            config (dict, optional): Configuration for the EDA. Defaults to None.
        
        Returns:
            dict: Dictionary containing EDA results.
        """
        self.logger.info(f"Starting exploratory data analysis on {dataset_name}")
        
        if dataset_name not in self.clean_data:
            self.logger.error(f"Dataset {dataset_name} not found in clean data")
            return {}
        
        df = self.clean_data[dataset_name]
        eda_results = {}
        
        # Basic statistics
        eda_results['basic_stats'] = {
            'description': df.describe(),
            'info': df.dtypes.to_dict(),
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Correlations
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                eda_results['correlations'] = numeric_df.corr().to_dict()
        except Exception as e:
            self.logger.warning(f"Could not compute correlations: {str(e)}")
        
        # Class distribution for categorical variables
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        eda_results['categorical_distributions'] = {}
        
        for col in categorical_columns:
            eda_results['categorical_distributions'][col] = df[col].value_counts().to_dict()
        
        # Save the results
        self.analysis_results[f"{dataset_name}_eda"] = eda_results
        self.logger.info(f"EDA completed for {dataset_name}")
        
        return eda_results
    
    def perform_statistical_analysis(self, analysis_config):
        """
        Perform statistical analysis on the data.
        
        Args:
            analysis_config (dict): Configuration for the statistical analysis.
        
        Returns:
            dict: Dictionary containing analysis results.
        """
        self.logger.info("Starting statistical analysis")
        
        try:
            results = {}
            
            for analysis_name, config in analysis_config.items():
                self.logger.info(f"Performing {analysis_name} analysis")
                
                analysis_type = config.get('type', '').lower()
                dataset_name = config.get('dataset', '')
                
                if dataset_name not in self.clean_data:
                    self.logger.warning(f"Dataset {dataset_name} not found in clean data")
                    continue
                
                df = self.clean_data[dataset_name]
                
                if analysis_type == 'ttest':
                    results[analysis_name] = self._perform_ttest(df, config)
                elif analysis_type == 'anova':
                    results[analysis_name] = self._perform_anova(df, config)
                elif analysis_type == 'chi_square':
                    results[analysis_name] = self._perform_chi_square(df, config)
                elif analysis_type == 'correlation':
                    results[analysis_name] = self._perform_correlation(df, config)
                elif analysis_type == 'regression':
                    results[analysis_name] = self._perform_regression(df, config)
                else:
                    self.logger.warning(f"Unsupported analysis type: {analysis_type}")
            
            self.analysis_results.update(results)
            self.logger.info("Statistical analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during statistical analysis: {str(e)}")
            return {}
    
    def _perform_ttest(self, df, config):
        """Perform t-test analysis."""
        group_column = config.get('group_column', '')
        value_column = config.get('value_column', '')
        group1 = config.get('group1', '')
        group2 = config.get('group2', '')
        
        if not all([group_column, value_column, group1, group2]):
            self.logger.warning("Missing required parameters for t-test")
            return {}
        
        try:
            group1_data = df[df[group_column] == group1][value_column]
            group2_data = df[df[group_column] == group2][value_column]
            
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
            
            return {
                'test_type': 'Independent t-test',
                't_statistic': t_stat,
                'p_value': p_value,
                'group1_mean': group1_data.mean(),
                'group2_mean': group2_data.mean(),
                'group1_std': group1_data.std(),
                'group2_std': group2_data.std(),
                'significant': p_value < 0.05
            }
        except Exception as e:
            self.logger.warning(f"Error in t-test: {str(e)}")
            return {}
    
    def _perform_anova(self, df, config):
        """Perform ANOVA analysis."""
        group_column = config.get('group_column', '')
        value_column = config.get('value_column', '')
        
        if not all([group_column, value_column]):
            self.logger.warning("Missing required parameters for ANOVA")
            return {}
        
        try:
            groups = df[group_column].unique()
            group_data = [df[df[group_column] == group][value_column] for group in groups]
            
            f_stat, p_value = stats.f_oneway(*group_data)
            
            # Post-hoc test (Tukey's HSD)
            tukey = pairwise_tukeyhsd(df[value_column], df[group_column])
            
            return {
                'test_type': 'One-way ANOVA',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'groups': list(groups),
                'group_means': [data.mean() for data in group_data],
                'tukey_results': {
                    'group1': tukey.groupsunique[tukey.reject],
                    'group2': tukey.groupsunique[~tukey.reject]
                }
            }
        except Exception as e:
            self.logger.warning(f"Error in ANOVA: {str(e)}")
            return {}
    
    def _perform_chi_square(self, df, config):
        """Perform Chi-square test."""
        column1 = config.get('column1', '')
        column2 = config.get('column2', '')
        
        if not all([column1, column2]):
            self.logger.warning("Missing required parameters for Chi-square test")
            return {}
        
        try:
            contingency_table = pd.crosstab(df[column1], df[column2])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            return {
                'test_type': 'Chi-square test of independence',
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05,
                'contingency_table': contingency_table.to_dict()
            }
        except Exception as e:
            self.logger.warning(f"Error in Chi-square test: {str(e)}")
            return {}
    
    def _perform_correlation(self, df, config):
        """Perform correlation analysis."""
        columns = config.get('columns', [])
        method = config.get('method', 'pearson')
        
        if not columns:
            self.logger.warning("No columns specified for correlation analysis")
            return {}
        
        try:
            valid_columns = [col for col in columns if col in df.columns]
            if len(valid_columns) < 2:
                self.logger.warning("Need at least 2 valid columns for correlation")
                return {}
            
            corr_matrix = df[valid_columns].corr(method=method)
            
            return {
                'method': method,
                'correlation_matrix': corr_matrix.to_dict()
            }
        except Exception as e:
            self.logger.warning(f"Error in correlation analysis: {str(e)}")
            return {}
    
    def _perform_regression(self, df, config):
        """Perform regression analysis."""
        formula = config.get('formula', '')
        
        if not formula:
            self.logger.warning("No formula provided for regression analysis")
            return {}
        
        try:
            model = ols(formula, data=df).fit()
            
            return {
                'model_type': 'OLS Regression',
                'r_squared': model.rsquared,
                'adjusted_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'p_value': model.f_pvalue,
                'coefficients': model.params.to_dict(),
                'standard_errors': model.bse.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'significant_predictors': [pred for pred, p_val in model.pvalues.items() if p_val < 0.05]
            }
        except Exception as e:
            self.logger.warning(f"Error in regression analysis: {str(e)}")
            return {}
    
    def create_visualizations(self, visualization_config):
        """
        Create data visualizations.
        
        Args:
            visualization_config (dict): Configuration for the visualizations.
        
        Returns:
            dict: Dictionary of created visualizations.
        """
        self.logger.info("Creating visualizations")
        
        visualizations = {}
        
        try:
            for viz_name, config in visualization_config.items():
                self.logger.info(f"Creating visualization: {viz_name}")
                
                viz_type = config.get('type', '').lower()
                dataset_name = config.get('dataset', '')
                output_file = config.get('output_file', f"{viz_name}.png")
                
                if dataset_name not in self.clean_data:
                    self.logger.warning(f"Dataset {dataset_name} not found in clean data")
                    continue
                
                df = self.clean_data[dataset_name]
                
                # Set the figure size and style
                plt.figure(figsize=config.get('figsize', (10, 6)))
                sns.set_style(config.get('style', 'whitegrid'))
                
                if viz_type == 'histogram':
                    self._create_histogram(df, config)
                elif viz_type == 'scatter':
                    self._create_scatter(df, config)
                elif viz_type == 'bar':
                    self._create_bar(df, config)
                elif viz_type == 'line':
                    self._create_line(df, config)
                elif viz_type == 'heatmap':
                    self._create_heatmap(df, config)
                elif viz_type == 'box':
                    self._create_box(df, config)
                elif viz_type == 'pie':
                    self._create_pie(df, config)
                else:
                    self.logger.warning(f"Unsupported visualization type: {viz_type}")
                    plt.close()
                    continue
                
                # Add title and labels
                if config.get('title'):
                    plt.title(config.get('title'), fontsize=config.get('title_fontsize', 14))
                if config.get('xlabel'):
                    plt.xlabel(config.get('xlabel'), fontsize=config.get('label_fontsize', 12))
                if config.get('ylabel'):
                    plt.ylabel(config.get('ylabel'), fontsize=config.get('label_fontsize', 12))
                
                # Save the figure
                os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
                plt.tight_layout()
                plt.savefig(output_file, dpi=config.get('dpi', 300))
                
                visualizations[viz_name] = {
                    'type': viz_type,
                    'file_path': output_file
                }
                
                plt.close()
                self.logger.info(f"Visualization {viz_name} created and saved to {output_file}")
            
            self.visualizations.update(visualizations)
            self.logger.info("All visualizations created successfully")
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            plt.close()
            return {}
    
    def _create_histogram(self, df, config):
        """Create a histogram visualization."""
        column = config.get('column', '')
        bins = config.get('bins', 10)
        kde = config.get('kde', False)
        
        if not column:
            return
        
        sns.histplot(df[column], bins=bins, kde=kde, color=config.get('color', 'blue'))
    
    def _create_scatter(self, df, config):
        """Create a scatter plot visualization."""
        x_column = config.get('x_column', '')
        y_column = config.get('y_column', '')
        hue = config.get('hue', None)
        
        if not x_column or not y_column:
            return
        
        sns.scatterplot(
            x=x_column, 
            y=y_column, 
            hue=hue,
            data=df,
            alpha=config.get('alpha', 0.7),
            palette=config.get('palette', 'viridis')
        )
    
    def _create_bar(self, df, config):
        """Create a bar plot visualization."""
        x_column = config.get('x_column', '')
        y_column = config.get('y_column', '')
        hue = config.get('hue', None)
        
        if not x_column or not y_column:
            return
        
        sns.barplot(
            x=x_column,
            y=y_column,
            hue=hue,
            data=df,
            palette=config.get('palette', 'viridis')
        )
        
        if config.get('rotate_xlabels', False):
            plt.xticks(rotation=90)
    
    def _create_line(self, df, config):
        """Create a line plot visualization."""
        x_column = config.get('x_column', '')
        y_column = config.get('y_column', '')
        hue = config.get('hue', None)
        
        if not x_column or not y_column:
            return
        
        sns.lineplot(
            x=x_column,
            y=y_column,
            hue=hue,
            data=df,
            marker=config.get('marker', 'o'),
            palette=config.get('palette', 'viridis')
        )
    
    def _create_heatmap(self, df, config):
        """Create a heatmap visualization."""
        columns = config.get('columns', [])
        
        if not columns:
            # Use all numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return
        
        corr = numeric_df.corr()
        
        sns.heatmap(
            corr,
            annot=config.get('annot', True),
            cmap=config.get('cmap', 'coolwarm'),
            linewidths=config.get('linewidths', 0.5),
            fmt=config.get('fmt', '.2f')
        )
    
    def _create_box(self, df, config):
        """Create a box plot visualization."""
        x_column = config.get('x_column', None)
        y_column = config.get('y_column', '')
        hue = config.get('hue', None)
        
        if not y_column:
            return
        
        sns.boxplot(
            x=x_column,
            y=y_column,
            hue=hue,
            data=df,
            palette=config.get('palette', 'viridis')
        )
    
    def _create_pie(self, df, config):
        """Create a pie chart visualization."""
        column = config.get('column', '')
        
        if not column:
            return
        
        value_counts = df[column].value_counts()
        
        plt.pie(
            value_counts,
            labels=value_counts.index,
            autopct=config.get('autopct', '%1.1f%%'),
            startangle=config.get('startangle', 90),
            colors=sns.color_palette(config.get('palette', 'viridis'), len(value_counts))
        )
        plt.axis('equal')
    
    def generate_report(self, report_config, output_file='report.html'):
        """
        Generate a comprehensive HTML report from the analysis results.
        
        Args:
            report_config (dict): Configuration for the report.
            output_file (str, optional): Path to the output file. Defaults to 'report.html'.
        
        Returns:
            bool: True if report generation was successful, False otherwise.
        """
        self.logger.info(f"Generating report to {output_file}")
        
        try:
            # Create basic HTML structure
            html_content = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>Data Analysis Report</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 40px; }",
                "        h1 { color: #2c3e50; }",
                "        h2 { color: #3498db; margin-top: 30px; }",
                "        h3 { color: #2980b9; }",
                "        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
                "        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }",
                "        th { background-color: #f2f2f2; }",
                "        tr:hover { background-color: #f5f5f5; }",
                "        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }",
                "        .viz-container { margin: 20px 0; text-align: center; }",
                "        img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }",
                "    </style>",
                "</head>",
                "<body>"
            ]
            
            # Add title and introduction
            html_content.extend([
                f"    <h1>{report_config.get('title', 'Data Analysis Report')}</h1>",
                f"    <p class='summary'>{report_config.get('introduction', 'This report presents the results of the data analysis.')}</p>"
            ])
            
            # Add sections based on the configuration
            sections = report_config.get('sections', [])
            
            for section in sections:
                section_type = section.get('type', '')
                section_title = section.get('title', '')
                
                html_content.append(f"    <h2>{section_title}</h2>")
                
                if section_type == 'text':
                    html_content.append(f"    <p>{section.get('content', '')}</p>")
                
                elif section_type == 'datasets':
                    html_content.append("    <h3>Dataset Summary</h3>")
                    
                    for dataset_name in section.get('datasets', []):
                        if dataset_name in self.clean_data:
                            df = self.clean_data[dataset_name]
                            html_content.append(f"    <h4>{dataset_name}</h4>")
                            html_content.append(f"    <p>Shape: {df.shape[0]} rows, {df.shape[1]} columns</p>")
                            
                            # Add table with first few rows
                            html_content.append("    <h5>Preview:</h5>")
                            html_content.append(df.head().to_html())
                            
                            # Add column information
                            html_content.append("    <h5>Column Information:</h5>")
                            dtype_df = pd.DataFrame({
                                'Column': df.columns,
                                'Type': df.dtypes.astype(str).values,
                                'Missing Values': df.isnull().sum().values,
                                'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
                            })
                            html_content.append(dtype_df.to_html(index=False))
                
                elif section_type == 'eda':
                    dataset_name = section.get('dataset', '')
                    eda_key = f"{dataset_name}_eda"
                    
                    if eda_key in self.analysis_results:
                        eda_results = self.analysis_results[eda_key]
                        
                        # Basic statistics
                        if 'basic_stats' in eda_results:
                            html_content.append("    <h3>Basic Statistics</h3>")
                            html_content.append(pd.DataFrame(eda_results['basic_stats']['description']).to_html())
                        
                        # Missing values
                        if 'missing_values' in eda_results.get('basic_stats', {}):
                            missing_vals = eda_results['basic_stats']['missing_values']
                            missing_df = pd.DataFrame({
                                'Column': missing_vals.keys(),
                                'Missing Values': missing_vals.values()
                            })
                            html_content.append("    <h3>Missing Values</h3>")
                            html_content.append(missing_df.to_html(index=False))
                        
                        # Correlations
                        if 'correlations' in eda_results:
                            html_content.append("    <h3>Correlation Matrix</h3>")
                            html_content.append(pd.DataFrame(eda_results['correlations']).to_html())
                        
                        # Categorical distributions
                        if 'categorical_distributions' in eda_results:
                            html_content.append("    <h3>Categorical Distributions</h3>")
                            for col, dist in eda_results['categorical_distributions'].items():
                                html_content.append(f"    <h4>{col}</h4>")
                                dist_df = pd.DataFrame({
                                    'Value': dist.keys(),
                                    'Count': dist.values()
                                })
                                html_content.append(dist_df.to_html(index=False))
                
                elif section_type == 'statistical_analysis':
                    analyses = section.get('analyses', [])
                    
                    html_content.append("    <h3>Statistical Analysis</h3>")
                    
                    for analysis_name in analyses:
                        if analysis_name in self.analysis_results:
                            result = self.analysis_results[analysis_name]
                            html_content.append(f"    <h4>{analysis_name}</h4>")
                            
                            # Convert the result to a more readable format
                            if isinstance(result, dict):
                                for key, value in result.items():
                                    if isinstance(value, dict):
                                        html_content.append(f"    <h5>{key}</h5>")
                                        html_content.append(pd.DataFrame(value).to_html())
                                    else:
                                        html_content.append(f"    <p><strong>{key}:</strong> {value}</p>")
                
                elif section_type == 'visualizations':
                    viz_names = section.get('visualizations', [])
                    
                    html_content.append("    <h3>Visualizations</h3>")
                    
                    for viz_name in viz_names:
                        if viz_name in self.visualizations:
                            viz = self.visualizations[viz_name]
                            html_content.append(f"    <div class='viz-container'>")
                            html_content.append(f"        <h4>{viz_name}</h4>")
                            html_content.append(f"        <img src='{viz['file_path']}' alt='{viz_name}'>")
                            html_content.append(f"    </div>")
                
                elif section_type == 'conclusion':
                    html_content.append(f"    <p>{section.get('content', '')}</p>")
            
            # Add footer and close HTML tags
            html_content.extend([
                f"    <hr>",
                f"    <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>",
                "</body>",
                "</html>"
            ])
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write('\n'.join(html_content))
            
            self.logger.info(f"Report successfully generated to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return False
    
    def export_results(self, export_config):
        """
        Export analysis results to various formats.
        
        Args:
            export_config (dict): Configuration for the export.
        
        Returns:
            bool: True if export was successful, False otherwise.
        """
        self.logger.info("Exporting analysis results")
        
        try:
            for export_name, config in export_config.items():
                export_type = config.get('type', '').lower()
                output_file = config.get('output_file', f"{export_name}.csv")
                data_key = config.get('data', '')
                
                self.logger.info(f"Exporting {export_name} to {output_file}")
                
                # Determine the data to export
                if data_key in self.clean_data:
                    data = self.clean_data[data_key]
                elif data_key in self.analysis_results:
                    data = self.analysis_results[data_key]
                else:
                    self.logger.warning(f"Data key {data_key} not found")
                    continue
                
                # Convert to DataFrame if it's not already
                if not isinstance(data, pd.DataFrame):
                    if isinstance(data, dict):
                        try:
                            data = pd.DataFrame(data)
                        except:
                            data = pd.DataFrame([data])
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
                
                # Export based on type
                if export_type == 'csv':
                    data.to_csv(output_file, index=config.get('include_index', False))
                elif export_type == 'excel':
                    data.to_excel(output_file, index=config.get('include_index', False))
                elif export_type == 'json':
                    if isinstance(data, pd.DataFrame):
                        with open(output_file, 'w') as f:
                            f.write(data.to_json(orient=config.get('orient', 'records')))
                    else:
                        with open(output_file, 'w') as f:
                            json.dump(data, f, indent=2)
                elif export_type == 'html':
                    with open(output_file, 'w') as f:
                        f.write(data.to_html(index=config.get('include_index', False)))
                else:
                    self.logger.warning(f"Unsupported export type: {export_type}")
                    continue
                
                self.logger.info(f"Successfully exported {export_name} to {output_file}")
            
            self.logger.info("All exports completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = DataAnalysisPipeline()
    
    # Example configuration
    data_sources = {
        'sales_data': {
            'type': 'csv',
            'path': 'data/sales_data.csv'
        }
    }
    
    cleaning_steps = {
        'sales_data': [
            {'type': 'drop_duplicates'},
            {'type': 'drop_na', 'subset': ['date', 'product', 'revenue']},
            {'type': 'convert_types', 'conversions': {'revenue': 'float', 'quantity': 'int'}}
        ]
    }
    
    # Extract and clean data
    pipeline.extract_data(data_sources)
    pipeline.clean_data(cleaning_steps)
    
    # Perform EDA
    pipeline.perform_eda('sales_data')
    
    # Perform statistical analysis
    analysis_config = {
        'revenue_by_product': {
            'type': 'anova',
            'dataset': 'sales_data',
            'group_column': 'product',
            'value_column': 'revenue'
        }
    }
    pipeline.perform_statistical_analysis(analysis_config)
    
    # Create visualizations
    visualization_config = {
        'revenue_histogram': {
            'type': 'histogram',
            'dataset': 'sales_data',
            'column': 'revenue',
            'bins': 20,
            'kde': True,
            'title': 'Distribution of Revenue',
            'xlabel': 'Revenue',
            'ylabel': 'Frequency',
            'output_file': 'visualizations/revenue_histogram.png'
        }
    }
    pipeline.create_visualizations(visualization_config)
    
    # Generate report
    report_config = {
        'title': 'Sales Data Analysis Report',
        'introduction': 'This report presents the analysis of our sales data.',
        'sections': [
            {
                'type': 'datasets',
                'title': 'Dataset Overview',
                'datasets': ['sales_data']
            },
            {
                'type': 'eda',
                'title': 'Exploratory Data Analysis',
                'dataset': 'sales_data'
            },
            {
                'type': 'statistical_analysis',
                'title': 'Statistical Analysis',
                'analyses': ['revenue_by_product']
            },
            {
                'type': 'visualizations',
                'title': 'Data Visualizations',
                'visualizations': ['revenue_histogram']
            },
            {
                'type': 'conclusion',
                'title': 'Conclusion',
                'content': 'Based on our analysis, we can conclude that...'
            }
        ]
    }
    pipeline.generate_report(report_config, 'reports/sales_analysis_report.html')
    
    # Export results
    export_config = {
        'cleaned_sales_data': {
            'type': 'csv',
            'data': 'sales_data',
            'output_file': 'exports/cleaned_sales_data.csv'
        }
    }
    pipeline.export_results(export_config)
