# Startup Growth Analytics - MLOps Pipeline Guide

## 🚀 Complete MLOps Pipeline

This project implements a full MLOps pipeline with the following stages:

```
data_ingestion → feature_engineering → success_scoring → model_training → evaluation → deployment
```

## 📋 Pipeline Stages

### 1. Data Ingestion
**File:** `src/data_ingestion.py`
- Loads raw startup data from CSV
- Handles missing values (median for numerical, mode for categorical)
- Removes duplicate entries
- Saves cleaned data to `data/processed/processed_data.csv`

### 2. Feature Engineering
**File:** `src/feature_engineering.py`
- Creates `Startup_Age` feature (current year - founding year)
- Creates `Success_Status` based on thresholds:
  - Funding > $100M USD
  - Employees > 1000
  - Valuation > $500M USD
- Encodes categorical variables (Industry, Region, Exit Status)
- Normalizes numerical features using StandardScaler
- Splits data into train/test sets (80/20)
- Saves processed datasets and encoders

### 3. Success Scoring
**File:** `src/success_scoring.py`
- Analyzes success distribution across the dataset
- Generates visualizations:
  - Success distribution pie chart
  - Success rate by industry
  - Success rate by region
  - Funding vs success histogram
  - Employees vs success histogram
  - Correlation heatmap
- Calculates key insights and statistics
- Saves analysis to `reports/success_analysis.json`

### 4. Model Training
**File:** `src/model_training.py`
- Trains Random Forest Classifier
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 10
  - class_weight: balanced
- Generates feature importance plot
- Generates confusion matrix
- Saves model to `models/startup_success_model.joblib`

### 5. Evaluation
**File:** `src/evaluation.py`
- Comprehensive model evaluation on test set
- Metrics calculated:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
- Visualizations:
  - ROC curve
  - Precision-Recall curve
  - Detailed confusion matrix
  - Metrics comparison bar chart
- Generates evaluation report
- Saves metrics to `reports/evaluation_metrics.json`

### 6. Deployment
**File:** `src/deployment.py`
- Prepares model for deployment
- Creates Flask REST API template
- Generates Dockerfile for containerization
- Creates deployment documentation
- Saves artifacts to `deployment/` directory

## 🔧 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Data Location
Ensure your data file is at:
```
data/raw/startup_data.csv
```

### 3. Configure Parameters
Edit `config/params.yaml` to adjust:
- Success thresholds
- Model hyperparameters
- Train/test split ratio
- Feature lists

## ▶️ Running the Pipeline

### Option 1: Run Complete Pipeline
```bash
python run_pipeline.py
```

### Option 2: Run Individual Stages
```bash
python src/data_ingestion.py
python src/feature_engineering.py
python src/success_scoring.py
python src/model_training.py
python src/evaluation.py
python src/deployment.py
```

### Option 3: Use DVC Pipeline
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro model_training
```

## 📊 Output Structure

After running the pipeline, you'll have:

```
Startup-Growth-Analytics/
├── data/
│   ├── raw/
│   │   └── startup_data.csv
│   ├── processed/
│   │   ├── processed_data.csv
│   │   ├── train_data.csv
│   │   └── test_data.csv
│   └── final/
│       └── final_data.csv
├── models/
│   ├── startup_success_model.joblib
│   └── label_encoders.joblib
├── reports/
│   ├── figures/
│   │   ├── success_distribution.png
│   │   ├── success_by_industry.png
│   │   ├── success_by_region.png
│   │   ├── funding_vs_success.png
│   │   ├── employees_vs_success.png
│   │   ├── correlation_heatmap.png
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── roc_curve.png
│   │   ├── precision_recall_curve.png
│   │   ├── confusion_matrix_detailed.png
│   │   └── metrics_comparison.png
│   ├── success_analysis.json
│   ├── model_metrics.json
│   ├── evaluation_metrics.json
│   └── evaluation_report.txt
├── deployment/
│   ├── model.joblib
│   ├── encoders.joblib
│   ├── metadata.json
│   ├── api.py
│   ├── requirements_api.txt
│   ├── Dockerfile
│   └── README.md
└── logs/
    ├── startup_analytics.log
    └── pipeline.log
```

## 📈 Key Metrics

The pipeline tracks and reports:
- **Data Quality**: Missing values, duplicates, data distribution
- **Success Rate**: Percentage of successful startups
- **Model Performance**: Accuracy, Precision, Recall, F1, ROC AUC
- **Feature Importance**: Which features most impact predictions
- **Business Insights**: Top industries, regions, funding patterns

## 🔍 Monitoring & Logging

All stages log to:
- Console output (real-time)
- `logs/startup_analytics.log` (detailed logs)
- `logs/pipeline.log` (pipeline execution logs)

## 🚢 Deployment

After running the pipeline, deploy using:

### Local Testing
```bash
cd deployment
python api.py
```

### Docker
```bash
cd deployment
docker build -t startup-predictor .
docker run -p 5000:5000 startup-predictor
```

### Test API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
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

## 🔄 Retraining

To retrain with new data:
1. Add new data to `data/raw/startup_data.csv`
2. Run: `python run_pipeline.py`
3. Compare metrics in `reports/evaluation_metrics.json`
4. If improved, redeploy using `deployment/` artifacts

## 📝 Configuration

Edit `config/params.yaml` to customize:

```yaml
feature_engineering:
  success_thresholds:
    funding_amount: 100  # Adjust threshold
    employees: 1000      # Adjust threshold
    valuation: 500       # Adjust threshold
  current_year: 2025     # Update annually

model_training:
  test_size: 0.2         # Train/test split
  random_state: 42       # Reproducibility
```

## 🐛 Troubleshooting

### Issue: Missing columns error
**Solution**: Check that feature names in `params.yaml` match the normalized column names

### Issue: Model performance is poor
**Solution**: 
- Adjust success thresholds in `params.yaml`
- Try different model hyperparameters
- Collect more training data

### Issue: DVC errors
**Solution**: 
- Initialize DVC: `dvc init`
- Check dependencies are installed
- Verify file paths in `dvc.yaml`

## 📚 Additional Resources

- Model documentation: `deployment/README.md`
- Evaluation report: `reports/evaluation_report.txt`
- Success insights: `reports/success_analysis.json`
- API documentation: `deployment/README.md`

## 🎯 Next Steps

1. ✅ Run the complete pipeline
2. ✅ Review generated reports and visualizations
3. ✅ Test the deployed API locally
4. ✅ Deploy to cloud platform (AWS/GCP/Azure)
5. ✅ Set up monitoring and alerts
6. ✅ Schedule periodic retraining

---

**Happy Analyzing! 🚀**
