"""Data ingestion module for Startup Growth Analytics."""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple
import yaml

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

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        logger.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """Save data to CSV file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Perform initial data cleaning."""
    logger.info("Starting data cleaning")
    
    # Make a copy to avoid SettingWithCopyWarning
    df = data.copy()
    
    # Handle missing values
    # For numerical columns, fill with median
    num_cols = ['Funding Amount (M USD)', 'Valuation (M USD)', 
                'Revenue (M USD)', 'Employees', 'Market Share (%)']
    for col in num_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # For categorical columns, fill with mode
    cat_cols = ['Industry', 'Region', 'Exit Status']
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    logger.info(f"Data cleaning complete. Remaining rows: {len(df)}")
    return df

def main():
    """Main function for data ingestion."""
    try:
        # Load configuration
        config = load_config()
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Load and clean data
        raw_data_path = config['data_ingestion']['raw_data_path']
        processed_data_path = config['data_ingestion']['processed_data_path']
        
        df = load_data(raw_data_path)
        df_cleaned = clean_data(df)
        
        # Save processed data
        save_data(df_cleaned, processed_data_path)
        logger.info("Data ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise

if __name__ == "__main__":
    main()
