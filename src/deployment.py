"""Deployment module for Startup Growth Analytics."""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import yaml
import json
from datetime import datetime
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

class StartupSuccessPredictor:
    """Class for predicting startup success."""
    
    def __init__(self, model_path: str, encoders_path: str):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            encoders_path: Path to label encoders
        """
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoders_path)
        logger.info("Model and encoders loaded successfully")
    
    def preprocess_input(self, data: dict) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Dictionary with startup information
            
        Returns:
            Preprocessed DataFrame
        """
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    df[f"{col}_encoded"] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen labels
                    df[f"{col}_encoded"] = 0
                    logger.warning(f"Unseen label in {col}, using default encoding")
        
        return df
    
    def predict(self, data: dict) -> dict:
        """
        Predict startup success.
        
        Args:
            data: Dictionary with startup information
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess input
            df = self.preprocess_input(data)
            
            # Extract features (adjust based on your model's requirements)
            features = [col for col in df.columns if '_Norm' in col or '_encoded' in col or col == 'Startup_Age']
            X = df[features]
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            prediction_proba = self.model.predict_proba(X)[0]
            
            result = {
                'prediction': 'Successful' if prediction == 1 else 'Not Successful',
                'confidence': float(max(prediction_proba)),
                'success_probability': float(prediction_proba[1]),
                'failure_probability': float(prediction_proba[0]),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

def save_deployment_artifacts(output_dir: str = 'deployment'):
    """Save deployment artifacts and configuration."""
    logger.info("Saving deployment artifacts")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model and encoders
    import shutil
    shutil.copy('models/startup_success_model.joblib', os.path.join(output_dir, 'model.joblib'))
    shutil.copy('models/label_encoders.joblib', os.path.join(output_dir, 'encoders.joblib'))
    
    # Create deployment metadata
    metadata = {
        'model_name': 'Startup Success Predictor',
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'description': 'Machine learning model to predict startup success based on various features',
        'input_features': [
            'Funding Amount (M USD)',
            'Valuation (M USD)',
            'Revenue (M USD)',
            'Employees',
            'Market Share (%)',
            'Year Founded',
            'Industry',
            'Region'
        ],
        'output': {
            'prediction': 'Successful or Not Successful',
            'confidence': 'Prediction confidence score',
            'success_probability': 'Probability of success',
            'failure_probability': 'Probability of failure'
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Deployment artifacts saved to {output_dir}")

def create_prediction_api_template(output_dir: str = 'deployment'):
    """Create a template for REST API deployment."""
    logger.info("Creating API template")
    
    api_code = '''"""
Flask API for Startup Success Prediction
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model.joblib')
encoders = joblib.load('encoders.joblib')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        data = request.json
        
        # Preprocess input
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[f"{col}_encoded"] = encoder.transform(df[col].astype(str))
                except ValueError:
                    df[f"{col}_encoded"] = 0
        
        # Extract features
        features = [col for col in df.columns if '_Norm' in col or '_encoded' in col or col == 'Startup_Age']
        X = df[features]
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        result = {
            'prediction': 'Successful' if prediction == 1 else 'Not Successful',
            'confidence': float(max(prediction_proba)),
            'success_probability': float(prediction_proba[1]),
            'failure_probability': float(prediction_proba[0])
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
    
    with open(os.path.join(output_dir, 'api.py'), 'w') as f:
        f.write(api_code)
    
    # Create requirements for API
    api_requirements = '''flask==2.3.2
joblib==1.2.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
gunicorn==20.1.0
'''
    
    with open(os.path.join(output_dir, 'requirements_api.txt'), 'w') as f:
        f.write(api_requirements)
    
    # Create Dockerfile
    dockerfile = '''FROM python:3.9-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY api.py .
COPY model.joblib .
COPY encoders.joblib .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "api:app"]
'''
    
    with open(os.path.join(output_dir, 'Dockerfile'), 'w') as f:
        f.write(dockerfile)
    
    logger.info("API template created successfully")

def create_deployment_readme(output_dir: str = 'deployment'):
    """Create README for deployment."""
    logger.info("Creating deployment README")
    
    readme = '''# Startup Success Predictor - Deployment Guide

## Overview
This directory contains all necessary files to deploy the Startup Success Prediction model.

## Files
- `model.joblib`: Trained machine learning model
- `encoders.joblib`: Label encoders for categorical variables
- `metadata.json`: Model metadata and information
- `api.py`: Flask REST API implementation
- `requirements_api.txt`: Python dependencies for API
- `Dockerfile`: Docker configuration for containerized deployment

## Local Deployment

### Using Python
1. Install dependencies:
   ```bash
   pip install -r requirements_api.txt
   ```

2. Run the API:
   ```bash
   python api.py
   ```

3. Test the API:
   ```bash
   curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{
       "Funding Amount (M USD)": 150.0,
       "Valuation (M USD)": 1200.0,
       "Revenue (M USD)": 80.0,
       "Employees": 2500,
       "Market Share (%)": 7.5,
       "Year Founded": 2015,
       "Industry": "FinTech",
       "Region": "North America"
     }'
   ```

### Using Docker
1. Build the Docker image:
   ```bash
   docker build -t startup-predictor .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 startup-predictor
   ```

## Cloud Deployment

### AWS (Elastic Beanstalk)
1. Install EB CLI
2. Initialize: `eb init`
3. Create environment: `eb create startup-predictor-env`
4. Deploy: `eb deploy`

### Google Cloud Platform (Cloud Run)
1. Build image: `gcloud builds submit --tag gcr.io/PROJECT_ID/startup-predictor`
2. Deploy: `gcloud run deploy --image gcr.io/PROJECT_ID/startup-predictor --platform managed`

### Azure (App Service)
1. Create App Service plan
2. Deploy using Azure CLI or GitHub Actions

## API Endpoints

### Health Check
```
GET /health
```

### Prediction
```
POST /predict
Content-Type: application/json

{
  "Funding Amount (M USD)": 150.0,
  "Valuation (M USD)": 1200.0,
  "Revenue (M USD)": 80.0,
  "Employees": 2500,
  "Market Share (%)": 7.5,
  "Year Founded": 2015,
  "Industry": "FinTech",
  "Region": "North America"
}
```

## Response Format
```json
{
  "prediction": "Successful",
  "confidence": 0.85,
  "success_probability": 0.85,
  "failure_probability": 0.15
}
```

## Monitoring
- Set up logging and monitoring for production deployments
- Track prediction latency and accuracy
- Monitor API usage and errors

## Security
- Implement authentication (API keys, OAuth)
- Use HTTPS in production
- Rate limiting for API endpoints
- Input validation and sanitization

## Maintenance
- Regularly retrain model with new data
- Monitor model performance drift
- Update dependencies for security patches
'''
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme)
    
    logger.info("Deployment README created successfully")

def main():
    """Main function for deployment preparation."""
    try:
        # Load configuration
        config = load_config()
        
        # Create deployment directory
        deployment_dir = 'deployment'
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Save deployment artifacts
        save_deployment_artifacts(deployment_dir)
        
        # Create API template
        create_prediction_api_template(deployment_dir)
        
        # Create deployment README
        create_deployment_readme(deployment_dir)
        
        logger.info("=" * 80)
        logger.info("DEPLOYMENT PREPARATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Deployment artifacts saved to: {deployment_dir}/")
        logger.info("Next steps:")
        logger.info("1. Review the deployment README")
        logger.info("2. Test the API locally")
        logger.info("3. Deploy to your preferred platform")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in deployment preparation: {e}")
        raise

if __name__ == "__main__":
    main()
