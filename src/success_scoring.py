"""Success scoring module for Startup Growth Analytics."""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'startup_analytics.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    """Load configuration from params.yaml."""
    try:
        with open('config/params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def analyze_success_distribution(df: pd.DataFrame, output_dir: str = 'reports/figures') -> dict:
    """
    Analyze and visualize success distribution.
    
    Args:
        df: Input DataFrame with Success_Status column
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with success statistics
    """
    logger.info("Analyzing success distribution")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate success statistics
    success_count = df['Success_Status'].sum()
    total_count = len(df)
    success_rate = (success_count / total_count) * 100
    
    stats = {
        'total_startups': total_count,
        'successful_startups': int(success_count),
        'unsuccessful_startups': int(total_count - success_count),
        'success_rate': round(success_rate, 2)
    }
    
    logger.info(f"Success Statistics: {stats}")
    
    # Plot 1: Success Distribution Pie Chart
    plt.figure(figsize=(10, 6))
    labels = ['Successful', 'Not Successful']
    sizes = [success_count, total_count - success_count]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Startup Success Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 2: Success by Industry
    if 'Industry' in df.columns:
        plt.figure(figsize=(14, 8))
        industry_success = df.groupby('Industry')['Success_Status'].agg(['sum', 'count'])
        industry_success['rate'] = (industry_success['sum'] / industry_success['count']) * 100
        industry_success = industry_success.sort_values('rate', ascending=False)
        
        ax = industry_success['rate'].plot(kind='bar', color='steelblue')
        plt.title('Success Rate by Industry', fontsize=16, fontweight='bold')
        plt.xlabel('Industry', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(industry_success['rate']):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_by_industry.png'), dpi=300)
        plt.close()
    
    # Plot 3: Success by Region
    if 'Region' in df.columns:
        plt.figure(figsize=(12, 7))
        region_success = df.groupby('Region')['Success_Status'].agg(['sum', 'count'])
        region_success['rate'] = (region_success['sum'] / region_success['count']) * 100
        region_success = region_success.sort_values('rate', ascending=False)
        
        ax = region_success['rate'].plot(kind='bar', color='coral')
        plt.title('Success Rate by Region', fontsize=16, fontweight='bold')
        plt.xlabel('Region', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(region_success['rate']):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_by_region.png'), dpi=300)
        plt.close()
    
    # Plot 4: Funding vs Success
    if 'Funding Amount (M USD)' in df.columns:
        plt.figure(figsize=(12, 7))
        successful = df[df['Success_Status'] == 1]['Funding Amount (M USD)']
        unsuccessful = df[df['Success_Status'] == 0]['Funding Amount (M USD)']
        
        plt.hist([successful, unsuccessful], bins=30, label=['Successful', 'Not Successful'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
        plt.title('Funding Amount Distribution by Success Status', fontsize=16, fontweight='bold')
        plt.xlabel('Funding Amount (M USD)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'funding_vs_success.png'), dpi=300)
        plt.close()
    
    # Plot 5: Employees vs Success
    if 'Employees' in df.columns:
        plt.figure(figsize=(12, 7))
        successful = df[df['Success_Status'] == 1]['Employees']
        unsuccessful = df[df['Success_Status'] == 0]['Employees']
        
        plt.hist([successful, unsuccessful], bins=30, label=['Successful', 'Not Successful'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
        plt.title('Employee Count Distribution by Success Status', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Employees', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'employees_vs_success.png'), dpi=300)
        plt.close()
    
    # Plot 6: Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(14, 10))
        correlation_matrix = df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
        plt.close()
    
    return stats

def generate_success_insights(df: pd.DataFrame) -> dict:
    """
    Generate key insights about startup success factors.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with insights
    """
    logger.info("Generating success insights")
    
    insights = {}
    
    # Average metrics for successful vs unsuccessful startups
    successful = df[df['Success_Status'] == 1]
    unsuccessful = df[df['Success_Status'] == 0]
    
    if 'Funding Amount (M USD)' in df.columns:
        insights['avg_funding_successful'] = round(successful['Funding Amount (M USD)'].mean(), 2)
        insights['avg_funding_unsuccessful'] = round(unsuccessful['Funding Amount (M USD)'].mean(), 2)
    
    if 'Employees' in df.columns:
        insights['avg_employees_successful'] = round(successful['Employees'].mean(), 2)
        insights['avg_employees_unsuccessful'] = round(unsuccessful['Employees'].mean(), 2)
    
    if 'Valuation (M USD)' in df.columns:
        insights['avg_valuation_successful'] = round(successful['Valuation (M USD)'].mean(), 2)
        insights['avg_valuation_unsuccessful'] = round(unsuccessful['Valuation (M USD)'].mean(), 2)
    
    if 'Revenue (M USD)' in df.columns:
        insights['avg_revenue_successful'] = round(successful['Revenue (M USD)'].mean(), 2)
        insights['avg_revenue_unsuccessful'] = round(unsuccessful['Revenue (M USD)'].mean(), 2)
    
    if 'Startup_Age' in df.columns:
        insights['avg_age_successful'] = round(successful['Startup_Age'].mean(), 2)
        insights['avg_age_unsuccessful'] = round(unsuccessful['Startup_Age'].mean(), 2)
    
    # Top performing industries
    if 'Industry' in df.columns:
        industry_success = df.groupby('Industry')['Success_Status'].agg(['sum', 'count'])
        industry_success['rate'] = (industry_success['sum'] / industry_success['count']) * 100
        top_industries = industry_success.nlargest(3, 'rate')
        insights['top_industries'] = top_industries['rate'].to_dict()
    
    # Top performing regions
    if 'Region' in df.columns:
        region_success = df.groupby('Region')['Success_Status'].agg(['sum', 'count'])
        region_success['rate'] = (region_success['sum'] / region_success['count']) * 100
        top_regions = region_success.nlargest(3, 'rate')
        insights['top_regions'] = top_regions['rate'].to_dict()
    
    logger.info(f"Generated {len(insights)} insights")
    return insights

def main():
    """Main function for success scoring."""
    try:
        # Load configuration
        config = load_config()
        
        # Create output directories
        os.makedirs('reports/figures', exist_ok=True)
        
        # Load final data
        final_data_path = config['data_ingestion']['final_data_path']
        df = pd.read_csv(final_data_path)
        
        # Analyze success distribution
        stats = analyze_success_distribution(df)
        
        # Generate insights
        insights = generate_success_insights(df)
        
        # Save statistics and insights
        import json
        results = {
            'statistics': stats,
            'insights': insights
        }
        
        with open('reports/success_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Success scoring completed successfully")
        logger.info(f"Success Rate: {stats['success_rate']}%")
        
    except Exception as e:
        logger.error(f"Error in success scoring: {e}")
        raise

if __name__ == "__main__":
    main()
