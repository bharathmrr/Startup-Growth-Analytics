# Troubleshooting Guide

## Common Issues and Solutions

### 1. Module Import Errors

**Error:** `ModuleNotFoundError: No module named 'pandas'`

**Solution:**
```bash
pip install -r requirements.txt
```

### 2. File Not Found Errors

**Error:** `FileNotFoundError: data/raw/startup_data.csv`

**Solution:**
- Ensure data file is in correct location: `data/raw/startup_data.csv`
- Check if you ran from project root directory
- Verify file path in `config/params.yaml`

### 3. Feature Name Mismatch

**Error:** `KeyError: 'Funding_Amount_Norm'`

**Solution:**
- Feature names are auto-generated based on column names
- Check `config/params.yaml` features list matches normalized names
- Run feature engineering first to see actual column names

### 4. Memory Errors

**Error:** `MemoryError` during model training

**Solution:**
- Reduce dataset size for testing
- Adjust model parameters (reduce n_estimators)
- Close other applications
- Use a machine with more RAM

### 5. DVC Errors

**Error:** `dvc: command not found`

**Solution:**
```bash
pip install dvc
dvc init
```

**Error:** `ERROR: failed to reproduce 'stage_name'`

**Solution:**
- Check dependencies exist
- Verify file paths in `dvc.yaml`
- Run stages individually first

### 6. Model Performance Issues

**Issue:** Model accuracy is very low (<50%)

**Solutions:**
1. Check success thresholds in `config/params.yaml`
2. Verify data quality and distribution
3. Adjust model hyperparameters
4. Ensure features are properly normalized
5. Check for data leakage

### 7. Visualization Errors

**Error:** `No module named 'matplotlib'` or display issues

**Solution:**
```bash
pip install matplotlib seaborn
```

For headless environments:
```python
import matplotlib
matplotlib.use('Agg')
```

### 8. Encoding Errors

**Error:** `ValueError: y contains new labels`

**Solution:**
- Ensure test data uses same encoders as training
- Check for unseen categories in test set
- Use `handle_unknown='ignore'` in encoders

### 9. Permission Errors

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
- Run with appropriate permissions
- Check file/folder write permissions
- Close files if open in other programs

### 10. API Deployment Issues

**Error:** API not starting or connection refused

**Solution:**
1. Check port 5000 is not in use
2. Verify all deployment files exist
3. Check Flask installation
4. Review logs for specific errors

```bash
# Check if port is in use
netstat -ano | findstr :5000

# Kill process if needed
taskkill /PID <process_id> /F
```

## Debugging Tips

### Enable Verbose Logging

Edit logging configuration in any module:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Check Data at Each Stage

```python
import pandas as pd

# After data ingestion
df = pd.read_csv('data/processed/processed_data.csv')
print(df.info())
print(df.head())

# After feature engineering
df = pd.read_csv('data/final/final_data.csv')
print(df.columns)
print(df.describe())
```

### Verify Model Loading

```python
import joblib

model = joblib.load('models/startup_success_model.joblib')
print(f"Model type: {type(model)}")
print(f"Model features: {model.n_features_in_}")
```

### Test Individual Stages

Run stages one at a time to isolate issues:
```bash
python src/data_ingestion.py
# Check output before proceeding

python src/feature_engineering.py
# Check output before proceeding

# ... and so on
```

## Performance Optimization

### Speed Up Training

1. Reduce dataset size for testing:
```python
df = df.sample(frac=0.1)  # Use 10% of data
```

2. Adjust model parameters:
```yaml
model_training:
  n_estimators: 50  # Reduce from 100
  max_depth: 5      # Reduce from 10
```

3. Use fewer features:
```yaml
features: ["Funding_Amount_M_USD_Norm", "Employees_Norm", "Startup_Age"]
```

### Reduce Memory Usage

1. Process data in chunks
2. Delete intermediate variables
3. Use appropriate data types:
```python
df['Industry'] = df['Industry'].astype('category')
```

## Getting Help

### Check Logs

Always check logs first:
```bash
# View recent logs
tail -n 50 logs/startup_analytics.log

# Search for errors
grep -i "error" logs/startup_analytics.log
```

### Verify Environment

```bash
python --version  # Should be 3.7+
pip list          # Check installed packages
```

### Test Installation

```python
# test_installation.py
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn
import yaml
import joblib

print("All packages imported successfully!")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```

## Contact & Support

If issues persist:
1. Check project documentation
2. Review error logs carefully
3. Search for similar issues online
4. Create an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Environment details
   - Relevant log excerpts

## Quick Fixes Checklist

- [ ] Installed all requirements
- [ ] Data file in correct location
- [ ] Running from project root directory
- [ ] Config file properly formatted
- [ ] Sufficient disk space
- [ ] Sufficient memory
- [ ] Python version 3.7+
- [ ] No conflicting processes
- [ ] Proper file permissions
- [ ] Logs directory exists

---

**Still having issues? Check the logs and documentation first!**
