@echo off
echo ========================================
echo Startup Growth Analytics - Quick Start
echo ========================================
echo.

echo Step 1: Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "reports" mkdir reports
if not exist "reports\figures" mkdir reports\figures
if not exist "deployment" mkdir deployment
echo Done!
echo.

echo Step 2: Running complete MLOps pipeline...
echo.
python run_pipeline.py

echo.
echo ========================================
echo Pipeline execution completed!
echo ========================================
echo.
echo Check the following directories for outputs:
echo - reports/figures/ - All visualizations
echo - models/ - Trained models
echo - deployment/ - Deployment artifacts
echo - logs/ - Execution logs
echo.
pause
