"""Model training module for Startup Growth Analytics."""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_data(train_path: str, test_path: str, features: list, target: str) -> tuple:
    """Load and prepare training and testing data."""
    logger.info("Loading training and testing data")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Ensure all required features exist
    missing_features = [f for f in features if f not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing features in training data: {missing_features}")
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, training_accuracy)
    """
    logger.info("Training Random Forest model")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Training accuracy
    train_preds = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    
    logger.info(f"Model trained with training accuracy: {train_accuracy:.4f}")
    return model, train_accuracy

def evaluate_model(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    output_dir: str = 'reports/figures'
) -> dict:
    """
    Evaluate the trained model and generate performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels for test set
        output_dir: Directory to save evaluation plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Log metrics
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Successful', 'Successful'],
                yticklabels=['Not Successful', 'Successful'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
    
    return metrics

def main():
    """Main function for model training and evaluation."""
    try:
        # Load configuration
        config = load_config()
        model_config = config['model_training']
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports/figures', exist_ok=True)
        
        # Load data
        X_train, X_test, y_train, y_test = load_data(
            train_path='data/processed/train_data.csv',
            test_path='data/processed/test_data.csv',
            features=model_config['features'],
            target=model_config['target_column']
        )
        
        # Train model
        model, train_accuracy = train_model(
            X_train, 
            y_train, 
            random_state=model_config['random_state']
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        model_path = model_config['model_path']
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = 'reports/model_metrics.json'
        with open(metrics_path, 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

if __name__ == "__main__":
    main()
