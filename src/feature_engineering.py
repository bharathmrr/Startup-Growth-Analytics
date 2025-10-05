"""Feature engineering module for Startup Growth Analytics."""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import yaml
import joblib
from typing import Tuple

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

def create_success_metric(
    df: pd.DataFrame, 
    funding_thresh: float, 
    employees_thresh: int, 
    valuation_thresh: float
) -> pd.Series:
    """
    Create success metric based on funding, employees, and valuation thresholds.
    
    Args:
        df: Input DataFrame
        funding_thresh: Funding threshold in M USD
        employees_thresh: Employees count threshold
        valuation_thresh: Valuation threshold in M USD
        
    Returns:
        pd.Series: Binary series indicating success (1) or not (0)
    """
    return ((df['Funding Amount (M USD)'] > funding_thresh) & 
            (df['Employees'] > employees_thresh) & 
            (df['Valuation (M USD)'] > valuation_thresh)).astype(int)

def perform_feature_engineering(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Perform feature engineering on the dataset.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple containing:
            - Processed DataFrame
            - Dictionary of label encoders
    """
    logger.info("Starting feature engineering")
    df_processed = df.copy()
    
    # Create success metric
    success_config = config['feature_engineering']['success_thresholds']
    df_processed['Success_Status'] = create_success_metric(
        df_processed,
        success_config['funding_amount'],
        success_config['employees'],
        success_config['valuation']
    )
    
    # Create startup age feature
    current_year = config['feature_engineering']['current_year']
    df_processed['Startup_Age'] = current_year - df_processed['Year Founded']
    
    # Handle categorical variables
    label_encoders = {}
    for col in config['feature_engineering']['categorical_columns']:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f"{col}_encoded"] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Normalize numerical features
    scaler = StandardScaler()
    num_cols = config['feature_engineering']['numerical_columns']
    for col in num_cols:
        if col in df_processed.columns:
            df_processed[f"{col.replace(' ', '_').replace('(', '').replace(')', '')}_Norm"] = \
                scaler.fit_transform(df_processed[[col]])
    
    logger.info("Feature engineering completed")
    return df_processed, label_encoders

def split_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data into train/test sets (test_size={test_size})")
    # Simple random split (can be enhanced with stratified sampling if needed)
    test = df.sample(frac=test_size, random_state=random_state)
    train = df.drop(test.index)
    
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    return train, test

def main():
    """Main function for feature engineering."""
    try:
        # Load configuration
        config = load_config()
        
        # Load processed data
        processed_data_path = config['data_ingestion']['processed_data_path']
        final_data_path = config['data_ingestion']['final_data_path']
        
        df = pd.read_csv(processed_data_path)
        
        # Perform feature engineering
        df_processed, label_encoders = perform_feature_engineering(df, config)
        
        # Save label encoders
        os.makedirs('models', exist_ok=True)
        joblib.dump(label_encoders, 'models/label_encoders.joblib')
        
        # Split data
        train_df, test_df = split_data(
            df_processed,
            test_size=config['model_training']['test_size'],
            random_state=config['model_training']['random_state']
        )
        
        # Save final datasets
        train_df.to_csv('data/processed/train_data.csv', index=False)
        test_df.to_csv('data/processed/test_data.csv', index=False)
        df_processed.to_csv(final_data_path, index=False)
        
        logger.info("Feature engineering completed successfully")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

if __name__ == "__main__":
    main()
