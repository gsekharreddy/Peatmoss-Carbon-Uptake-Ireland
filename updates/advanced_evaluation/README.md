# Advanced Model Evaluation Framework

## Overview
This folder contains advanced evaluation tools for comprehensive analysis of the 4 trained models:
- **XGBoost**
- **Random Forest**
- **Support Vector Regression (SVR)**
- **PyTorch LSTM**

## Scripts

### 1. `regression_visualization.py`
**Purpose:** Generates 4 comprehensive regression evaluation visualization plots

**Output:**
- **xgboost_evaluation.png** - Regression evaluation matrix for XGBoost
- **random_forest_evaluation.png** - Regression evaluation matrix for Random Forest
- **svr_evaluation.png** - Regression evaluation matrix for SVR
- **pytorch_lstm_evaluation.png** - Regression evaluation matrix for PyTorch LSTM
- **model_comparison.png** - Combined RMSE, MAE, and R² comparison
- **metrics_summary.csv** - Summary of all metrics in tabular format

**Each evaluation matrix includes (2x2 grid):**
1. **Predicted vs Actual** - Scatter plot with perfect prediction reference line
2. **Residuals Distribution** - Histogram showing residual errors
3. **Q-Q Plot** - Quantile-Quantile plot for residual normality assessment
4. **Residuals vs Predicted** - Scatter plot to check for heteroscedasticity

**Metrics displayed on plots:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

### 2. `extensive_evaluation.py`
**Purpose:** Comprehensive model evaluation with 13+ metrics including RMSE and MAE

**Output:**
- **detailed_evaluation_metrics.csv** - Complete metrics for all models
- **model_rankings.csv** - Models ranked by R², RMSE, and MAE
- **EVALUATION_REPORT.txt** - Detailed text report with insights and rankings

**Metrics calculated for each model:**
1. **RMSE** - Root Mean Squared Error
2. **MAE** - Mean Absolute Error
3. **MSE** - Mean Squared Error
4. **R² Score** - Coefficient of Determination
5. **Adjusted R²** - R² adjusted for number of features
6. **MAPE** - Mean Absolute Percentage Error
7. **Median Absolute Error** - Robustness indicator
8. **RMSLE** - Root Mean Squared Logarithmic Error
9. **Explained Variance** - Variance explained by model
10. **Directional Accuracy** - Percentage of correct trend predictions
11. **Residual Std Dev** - Standard deviation of residuals
12. **Residual Skewness** - Distribution asymmetry check
13. **5-Fold Cross-Validation** - RMSE, MAE, R² (mean ± std)

**Console Output:**
- Model training progress
- Primary metrics table (RMSE, MAE, R²)
- Extended metrics table
- Additional metrics table
- Cross-validation results
- Model rankings (by R², RMSE, MAE)
- Overall best model determination

### 3. `correlation_matrix.py`
**Purpose:** Builds a Pearson correlation matrix for the 4 model prediction outputs

**Output:**
- **model_prediction_correlation_matrix.csv** - Numerical correlation matrix for the 4 models
- **model_prediction_correlation_matrix.png** - Heatmap visualization of the correlation matrix

**What it measures:**
- Correlation between the test-set predictions from XGBoost, Random Forest, SVR, and PyTorch LSTM
- Helps identify whether the models behave similarly or capture different patterns

## Usage

### Running the scripts:

```bash
# Generate visualization plots
python regression_visualization.py

# Perform extensive evaluation
python extensive_evaluation.py

# Generate the 4-model correlation matrix
python correlation_matrix.py
```

### Expected behavior:
1. **Data Loading** - Script loads from `../Augmented_Drought_Sequence.csv`
2. **Model Training** - All 4 models are trained on the dataset
3. **Evaluation** - Metrics are calculated and comparisons made
4. **Output Generation** - Files are saved to subdirectories:
   - `regression_visualization.py` → saves to `plots/`
   - `extensive_evaluation.py` → saves to `reports/`

## Directory Structure

```
advanced_evaluation/
├── README.md (this file)
├── regression_visualization.py
├── extensive_evaluation.py
├── correlation_matrix.py
├── plots/
│   ├── xgboost_evaluation.png
│   ├── random_forest_evaluation.png
│   ├── svr_evaluation.png
│   ├── pytorch_lstm_evaluation.png
│   ├── model_comparison.png
│   ├── metrics_summary.csv
│   ├── model_prediction_correlation_matrix.csv
│   └── model_prediction_correlation_matrix.png
└── reports/
    ├── detailed_evaluation_metrics.csv
    ├── model_rankings.csv
    └── EVALUATION_REPORT.txt
```

## Key Metrics Explained

### RMSE (Root Mean Squared Error)
- Penalizes larger errors more heavily
- Scale-dependent (same units as target)
- **Lower is better**

### MAE (Mean Absolute Error)
- Average absolute error magnitude
- More interpretable than RMSE
- More robust to outliers than RMSE
- **Lower is better**

### R² Score
- Proportion of variance explained (0-1)
- 1.0 = perfect prediction
- **Higher is better**

### MAPE (Mean Absolute Percentage Error)
- Percentage-based error metric
- Useful for relative performance comparison
- **Lower is better**

### Cross-Validation
- 5-fold cross-validation to assess model generalization
- Mean ± Std for each metric
- Helps detect overfitting

## Requirements

The scripts require the following packages (already in requirements.txt):
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning models and metrics
- `xgboost` - XGBoost regressor
- `torch` - PyTorch for LSTM
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `scipy` - Statistical functions

## Data Format

The scripts expect the following CSV structure:
```
Species,Freq,WTgroup,NEE
species1,freq_val,wt_group,nee_value
species2,freq_val,wt_group,nee_value
...
```

## Tips for Interpretation

1. **Compare RMSE & MAE**: If RMSE >> MAE, model has large outlier errors
2. **Check R² Scores**: Model with highest R² has best variance explanation
3. **Review Plots**: 
   - Predicted vs Actual should show points near diagonal
   - Residuals should be normally distributed
   - Q-Q plot should show points near diagonal
   - Residuals plot should show no patterns
4. **Cross-Validation**: Lower CV std means more stable model
5. **Overall Ranking**: Used weighted combination (R²: 50%, RMSE: 25%, MAE: 25%)

## Troubleshooting

**Issue: "CSV file not found"**
- Ensure `../Augmented_Drought_Sequence.csv` exists in parent directory

**Issue: "Column not found"**
- Check that CSV has columns: Species, Freq, WTgroup, NEE

**Issue: Plots don't show**
- Ensure matplotlib backend is configured properly
- Check file paths are writable

## Author Notes

These evaluation scripts provide comprehensive assessment of model performance from multiple angles, helping identify the best model for the carbon flux prediction task. The combination of visualization and statistical analysis ensures robust evaluation.
