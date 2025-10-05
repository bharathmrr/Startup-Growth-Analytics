"""Model evaluation module for Startup Growth Analytics."""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import yaml
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
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

def load_model_and_data(model_path: str, test_data_path: str, features: list, target: str):
    """Load trained model and test data."""
    logger.info("Loading model and test data")
    
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    return model, X_test, y_test

def calculate_metrics(y_true, y_pred, y_pred_proba) -> dict:
    """Calculate comprehensive evaluation metrics."""
    logger.info("Calculating evaluation metrics")
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba))
    }
    
    # Classification report
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = class_report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'true_negatives': int(cm[0][0]),
        'false_positives': int(cm[0][1]),
        'false_negatives': int(cm[1][0]),
        'true_positives': int(cm[1][1])
    }
    
    return metrics

def plot_roc_curve(y_true, y_pred_proba, output_dir: str):
    """Plot ROC curve."""
    logger.info("Plotting ROC curve")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, output_dir: str):
    """Plot Precision-Recall curve."""
    logger.info("Plotting Precision-Recall curve")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir: str):
    """Plot confusion matrix heatmap."""
    logger.info("Plotting confusion matrix")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Successful', 'Successful'],
                yticklabels=['Not Successful', 'Successful'],
                annot_kws={'size': 16})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_detailed.png'), dpi=300)
    plt.close()

def plot_metrics_comparison(metrics: dict, output_dir: str):
    """Plot comparison of different metrics."""
    logger.info("Plotting metrics comparison")
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()

def generate_evaluation_report(metrics: dict, output_path: str):
    """Generate a comprehensive evaluation report."""
    logger.info("Generating evaluation report")
    
    report = []
    report.append("=" * 80)
    report.append("STARTUP SUCCESS PREDICTION MODEL - EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("OVERALL PERFORMANCE METRICS")
    report.append("-" * 80)
    report.append(f"Accuracy:  {metrics['accuracy']:.4f}")
    report.append(f"Precision: {metrics['precision']:.4f}")
    report.append(f"Recall:    {metrics['recall']:.4f}")
    report.append(f"F1 Score:  {metrics['f1_score']:.4f}")
    report.append(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    report.append("")
    
    report.append("CONFUSION MATRIX")
    report.append("-" * 80)
    cm = metrics['confusion_matrix']
    report.append(f"True Negatives:  {cm['true_negatives']}")
    report.append(f"False Positives: {cm['false_positives']}")
    report.append(f"False Negatives: {cm['false_negatives']}")
    report.append(f"True Positives:  {cm['true_positives']}")
    report.append("")
    
    report.append("CLASSIFICATION REPORT")
    report.append("-" * 80)
    class_report = metrics['classification_report']
    for label, values in class_report.items():
        if isinstance(values, dict):
            report.append(f"\nClass {label}:")
            report.append(f"  Precision: {values['precision']:.4f}")
            report.append(f"  Recall:    {values['recall']:.4f}")
            report.append(f"  F1-Score:  {values['f1-score']:.4f}")
            report.append(f"  Support:   {values['support']}")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Evaluation report saved to {output_path}")

def main():
    """Main function for model evaluation."""
    try:
        # Load configuration
        config = load_config()
        model_config = config['model_training']
        
        # Create output directories
        os.makedirs('reports/figures', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Load model and data
        model, X_test, y_test = load_model_and_data(
            model_path=model_config['model_path'],
            test_data_path='data/processed/test_data.csv',
            features=model_config['features'],
            target=model_config['target_column']
        )
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate visualizations
        plot_roc_curve(y_test, y_pred_proba, 'reports/figures')
        plot_precision_recall_curve(y_test, y_pred_proba, 'reports/figures')
        plot_confusion_matrix(y_test, y_pred, 'reports/figures')
        plot_metrics_comparison(metrics, 'reports/figures')
        
        # Save metrics
        with open('reports/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate evaluation report
        generate_evaluation_report(metrics, 'reports/evaluation_report.txt')
        
        # Log summary
        logger.info("=" * 80)
        logger.info("MODEL EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info("=" * 80)
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
