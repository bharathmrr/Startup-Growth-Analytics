# Startup Growth Analytics - Project Summary

## 🎯 Project Overview

A comprehensive MLOps pipeline for analyzing startup growth patterns and predicting success metrics using machine learning.

## 📊 Dataset

**File:** `data/raw/startup_data.csv`
- **Total Records:** 500+ startups
- **Features:** 12 columns
  - Startup Name, Industry, Funding Rounds
  - Funding Amount (M USD), Valuation (M USD), Revenue (M USD)
  - Employees, Market Share (%), Profitable
  - Year Founded, Region, Exit Status

## 🔄 MLOps Pipeline Stages

### 1️⃣ Data Ingestion
- **Module:** `src/data_ingestion.py`
- **Purpose:** Load and clean raw data
- **Actions:**
  - Handle missing values (median/mode imputation)
  - Remove duplicates
  - Basic data validation
- **Output:** `data/processed/processed_data.csv`

### 2️⃣ Feature Engineering
- **Module:** `src/feature_engineering.py`
- **Purpose:** Create and transform features
- **Actions:**
  - Create `Startup_Age` = 2025 - Year Founded
  - Create `Success_Status` based on:
    - Funding > $100M USD
    - Employees > 1000
    - Valuation > $500M USD
  - Label encode categorical variables
  - Normalize numerical features (StandardScaler)
  - Split into train/test (80/20)
- **Outputs:**
  - `data/processed/train_data.csv`
  - `data/processed/test_data.csv`
  - `data/final/final_data.csv`
  - `models/label_encoders.joblib`

### 3️⃣ Success Scoring
- **Module:** `src/success_scoring.py`
- **Purpose:** Analyze success patterns
- **Visualizations:**
  - Success distribution pie chart
  - Success rate by industry
  - Success rate by region
  - Funding vs success histogram
  - Employees vs success histogram
  - Correlation heatmap
- **Output:** `reports/success_analysis.json` + 6 PNG charts

### 4️⃣ Model Training
- **Module:** `src/model_training.py`
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:**
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2
  - class_weight: balanced
- **Outputs:**
  - `models/startup_success_model.joblib`
  - `reports/model_metrics.json`
  - Confusion matrix & feature importance plots

### 5️⃣ Evaluation
- **Module:** `src/evaluation.py`
- **Metrics:**
  - Accuracy, Precision, Recall, F1 Score, ROC AUC
- **Visualizations:**
  - ROC curve
  - Precision-Recall curve
  - Detailed confusion matrix
  - Metrics comparison bar chart
- **Outputs:**
  - `reports/evaluation_metrics.json`
  - `reports/evaluation_report.txt`
  - 4 evaluation plots

### 6️⃣ Deployment
- **Module:** `src/deployment.py`
- **Purpose:** Prepare model for production
- **Artifacts:**
  - Flask REST API (`deployment/api.py`)
  - Dockerfile for containerization
  - Requirements file
  - Deployment documentation
  - Model metadata

## 📁 Project Structure

```
Startup-Growth-Analytics/
├── data/
│   ├── raw/                    # Original data
│   ├── processed/              # Cleaned & split data
│   └── final/                  # Final dataset
├── src/
│   ├── data_ingestion.py       # Stage 1
│   ├── feature_engineering.py  # Stage 2
│   ├── success_scoring.py      # Stage 3
│   ├── model_training.py       # Stage 4
│   ├── evaluation.py           # Stage 5
│   └── deployment.py           # Stage 6
├── models/                     # Trained models
├── reports/                    # Analysis & metrics
│   └── figures/                # All visualizations
├── deployment/                 # Production artifacts
├── logs/                       # Execution logs
├── config/
│   └── params.yaml            # Configuration
├── dvc.yaml                   # DVC pipeline
├── run_pipeline.py            # Main runner
├── quick_start.bat            # Windows quick start
├── requirements.txt           # Dependencies
├── PIPELINE_GUIDE.md          # Detailed guide
└── README.md                  # Project README
```

## 🚀 Quick Start

### Method 1: Windows Batch Script
```bash
quick_start.bat
```

### Method 2: Python Script
```bash
python run_pipeline.py
```

### Method 3: Individual Stages
```bash
python src/data_ingestion.py
python src/feature_engineering.py
python src/success_scoring.py
python src/model_training.py
python src/evaluation.py
python src/deployment.py
```

### Method 4: DVC Pipeline
```bash
dvc repro
```

## 📈 Expected Outputs

### Data Outputs
- ✅ Cleaned dataset (500+ records)
- ✅ Train/test split (400/100 approx.)
- ✅ Encoded & normalized features

### Model Outputs
- ✅ Trained Random Forest model
- ✅ Label encoders for categorical variables
- ✅ Model performance metrics

### Visualization Outputs (12+ charts)
- ✅ Success distribution analysis
- ✅ Industry & region insights
- ✅ Feature correlations
- ✅ Model performance curves
- ✅ Confusion matrices
- ✅ Feature importance

### Deployment Outputs
- ✅ REST API template
- ✅ Docker configuration
- ✅ Deployment documentation

## 🎓 Key Insights Expected

1. **Success Rate:** ~X% of startups meet success criteria
2. **Top Industries:** Which sectors have highest success rates
3. **Regional Patterns:** Geographic distribution of success
4. **Funding Impact:** Correlation between funding and success
5. **Model Performance:** Accuracy, precision, recall metrics
6. **Feature Importance:** Which factors most predict success

## 🔧 Configuration

Edit `config/params.yaml` to customize:

```yaml
# Success definition thresholds
success_thresholds:
  funding_amount: 100   # M USD
  employees: 1000
  valuation: 500        # M USD

# Model parameters
test_size: 0.2
random_state: 42
```

## 📊 Logging

All stages log to:
- **Console:** Real-time progress
- **File:** `logs/startup_analytics.log`
- **Pipeline:** `logs/pipeline.log`

## 🐳 Deployment

### Local API Testing
```bash
cd deployment
python api.py
```

### Docker Deployment
```bash
cd deployment
docker build -t startup-predictor .
docker run -p 5000:5000 startup-predictor
```

### API Usage
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

## 📚 Documentation

- **README.md** - Project overview
- **PIPELINE_GUIDE.md** - Detailed pipeline documentation
- **PROJECT_SUMMARY.md** - This file
- **deployment/README.md** - Deployment guide
- **reports/evaluation_report.txt** - Model evaluation

## 🎯 Success Criteria

✅ Clean, reproducible pipeline
✅ Comprehensive data analysis
✅ Trained ML model with >70% accuracy
✅ Detailed visualizations and insights
✅ Production-ready deployment artifacts
✅ Complete documentation

## 🔄 Maintenance

### Retraining
1. Add new data to `data/raw/startup_data.csv`
2. Run: `python run_pipeline.py`
3. Compare metrics in `reports/`
4. Redeploy if improved

### Monitoring
- Track model performance over time
- Monitor prediction accuracy
- Update thresholds as needed
- Retrain quarterly or when drift detected

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run pipeline to test
5. Submit pull request

## 📄 License

MIT License - See LICENSE file

---

**Built with ❤️ for Data Science & MLOps**
