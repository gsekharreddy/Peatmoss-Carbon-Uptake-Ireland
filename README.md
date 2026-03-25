# Sphagnum Carbon Uptake: Precipitation Frequency & Drought Effects

## Overview

This project analyzes how precipitation frequency and drought conditions affect carbon uptake in three species of Sphagnum moss (*S. balticum*, *S. fuscum*, and *S. majus*) from peatland sites in Ireland. It combines ecological field data with machine learning models to predict carbon flux, photosynthetic efficiency, and drought recovery patterns.

### Key Research Questions
- How does precipitation frequency influence carbon uptake (NEE) in different Sphagnum species?
- Can we predict carbon flux based on water table conditions and species type?
- What is the photosynthetic tipping point (crash point) for each species?
- How quickly do Sphagnum communities recover carbon uptake after drought?

---

## Project Structure

```
Nijp+Data_Precipitation+frequency+effects+on+Sphagnum+C+uptake/
├── Carbon_flux.py              # XGBoost model for NEE prediction
├── Drought_Seq.py              # LSTM model for drought-state NEE prediction
├── Photosynthesis.py           # SVR model for photosynthetic efficiency (PSII)
├── visualizer.py               # Generate all analysis visualizations
├── models/                     # Trained models and metrics
│   ├── xgboost_nee.joblib     # XGBoost model
│   ├── lstm_drought.pth        # PyTorch LSTM model
│   ├── svr_psii.joblib         # Support Vector Regression model
│   ├── xgboost_metrics.txt     # XGBoost performance metrics
│   ├── lstm_metrics.txt        # LSTM performance metrics
│   ├── svr_metrics.txt         # SVR performance metrics
│   ├── xgboost_conf_matrix.png # XGBoost confusion matrix
│   ├── lstm_conf_matrix.png    # LSTM confusion matrix
│   └── svr_conf_matrix.png     # SVR confusion matrix
└── visualizations/             # Generated analysis plots
    ├── 1_Precipitation_Dependence.png
    ├── 2_VWC_vs_NEE.png
    ├── 3_VWC_vs_PSII.png
    └── 4_Drought_Recovery.png
```

---

## Data Files Required

Place the following `.dat` files (tab-separated) in the project root directory:

| Dataset | Description | Key Variables |
|---------|-------------|---|
| `Dataset 1_Precipitation_dependence.dat` | Evaporation and precipitation metrics | Species, WTgroup, FP (Fraction Precipitation) |
| `Dataset 2_Carbon_fluxes.dat` | Net Ecosystem Exchange measurements | Core_ID, Species, Freq (frequency), WTgroup, NEE |
| `Dataset 3_VWCvsNEE.dat` | Volumetric Water Content vs carbon flux | Species, VWC, NEE |
| `Dataset 4_VWCvsPSII.dat` | Water content vs photosynthetic efficiency | Species, VWC, PSII |
| `Dataset 5_Recovery.dat` | Post-drought recovery data | Species, NEE_Wet, NEE_REWET |

**Note:** Missing values are represented as `-999` and are automatically filtered.

---

## Models & Methods

### 1. **Carbon_flux.py** — XGBoost Regression
**Purpose:** Predict Net Ecosystem Exchange (NEE) from species, precipitation frequency, and water table depth.

**Model Architecture:**
- Algorithm: XGBoost Regressor
- Features: Species (encoded), Rain Frequency (Freq), Water Table State (WTgroup)
- Target: NEE (Net Ecosystem Exchange)
- Hyperparameters: 200 estimators, learning rate 0.05, max depth 4
- Train/Test Split: 80/20

**Output:**
- Trained model: `models/xgboost_nee.joblib`
- Metrics: MSE, R² Score
- Confusion Matrix: Sink (NEE < 0) vs Source (NEE ≥ 0) classification

**Key Insight:** Predicts whether moss communities act as carbon sinks or sources under different climate conditions.

---

### 2. **Drought_Seq.py** — PyTorch LSTM
**Purpose:** Predict NEE at dry conditions based on NEE measurements during wet and moist periods.

**Model Architecture:**
- Algorithm: LSTM (Long Short-Term Memory)
- Input Sequence: [NEE_Wet, NEE_Moist] → [NEE_Dry]
- Hidden Units: 16
- Sequence Length: 2 time steps
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate 0.01)
- Training: 150 epochs

**Data Processing:**
- Grouped by Core_ID (individual peat cores)
- Averaged duplicate entries via pivot_table
- Standardized using StandardScaler for both input and target

**Output:**
- Trained model: `models/lstm_drought.pth`
- Metrics: MSE (scaled), R² Score
- Confusion Matrix: Source vs Sink prediction accuracy

**Key Insight:** Captures temporal patterns showing how water table transitions affect carbon flux transitions.

---

### 3. **Photosynthesis.py** — Support Vector Regression (SVR)
**Purpose:** Predict photosynthetic efficiency (PSII/Fv-Fm) based on water content and species.

**Model Architecture:**
- Algorithm: Support Vector Regression
- Kernel: Radial Basis Function (RBF)
- Features: Species (encoded), VWC (Volumetric Water Content)
- Target: PSII (Photosystem II efficiency = Fv/Fm)
- Parameters: C=1.0, epsilon=0.01

**Why SVR for Photosynthesis?** 
- SVR naturally handles the non-linear "tipping point" where photosynthesis drops off sharply
- RBF kernel captures threshold-based biological responses

**Output:**
- Trained model: `models/svr_psii.joblib`
- Metrics: MSE, R² Score
- Confusion Matrix: Photosynthetically active vs inactive

**Key Insight:** Identifies the critical water content threshold where each species' photosynthesis collapses.

---

### 4. **visualizer.py** — Analysis Visualizations
Generates four publication-quality plots:

| Plot | Description | X-Axis | Y-Axis |
|------|-------------|--------|--------|
| **1_Precipitation_Dependence.png** | How much rain becomes evaporation vs runoff at different water tables | Water Table State | FP (Evaporation Fraction) |
| **2_VWC_vs_NEE.png** | The carbon tipping point — where NEE shifts from source to sink | VWC | NEE (carbon flux) |
| **3_VWC_vs_PSII.png** | Photosynthesis efficiency crash point at low water content | VWC | PSII/Fv-Fm |
| **4_Drought_Recovery.png** | Resilience: pre-drought vs post-drought carbon uptake | Species | NEE |

**Color Scheme:**
- *S. balticum*: Blue (#1E90FF)
- *S. fuscum*: Orange (#FFA500)
- *S. majus*: Light Blue (#55CAFF)

---

## Installation & Dependencies

### System Requirements
- Python 3.8+

### Required Packages
All dependencies are listed in `requirements.txt`:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- torch (PyTorch)
- joblib

### Install Dependencies

#### Option 1: Using requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

#### Option 2: Manual installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost torch joblib
```

### PyTorch Installation Note
For platform-specific or GPU-enabled PyTorch installation, visit [pytorch.org](https://pytorch.org) to generate a custom install command for your system.

---

## Usage

### Run All Analyses
```bash
# 1. Train and evaluate all models
python Carbon_flux.py
python Drought_Seq.py
python Photosynthesis.py

# 2. Generate visualizations
python visualizer.py
```

### Run Individual Models

#### Carbon Flux Prediction
```bash
python Carbon_flux.py
```
- Loads `Dataset 2_Carbon_fluxes.dat`
- Trains XGBoost model
- Saves to `models/xgboost_nee.joblib`
- Prints MSE and R² metrics

#### Drought Sequence LSTM
```bash
python Drought_Seq.py
```
- Loads `Dataset 2_Carbon_fluxes.dat`
- Creates water table transition sequences
- Trains LSTM to predict dry-state NEE
- Saves to `models/lstm_drought.pth`
- Generates confusion matrix visualization

#### Photosynthesis Efficiency
```bash
python Photosynthesis.py
```
- Loads `Dataset 4_VWCvsPSII.dat`
- Trains SVR model for PSII prediction
- Saves to `models/svr_psii.joblib`
- Prints performance metrics

#### Generate Visualizations Only
```bash
python visualizer.py
```
- Creates 4 PNG plots in `visualizations/` folder
- Handles missing data (-999 values)
- Displays species-specific trends

---

## Model Performance

### Expected Metrics

| Model | Task | Target Metric |
|-------|------|---|
| **XGBoost** | NEE from species & water table | R² > 0.60, MSE < 10 |
| **LSTM** | Wet+Moist → Dry NEE transition | R² > 0.55, MSE (scaled) < 0.5 |
| **SVR** | PSII from VWC | R² > 0.65, MSE < 0.02 |

*Note: Actual performance varies based on data quality and site conditions.*

---

## Key Features & Variables

### Sphagnum Species
- **S. balticum (B)**: Acrotelm dominant, moderate drought tolerance
- **S. fuscum (F)**: Hollow-forming, high photosynthetic rate
- **S. majus (M)**: Moss associated with ombrotrophic wetlands

### Environmental Variables
- **WTgroup**: Water Table State (2=Wet, 3=Moist, 4=Dry)
- **VWC**: Volumetric Water Content (soil moisture %)
- **Freq**: Precipitation Frequency (frequency of rain events)
- **FP**: Fraction of Evaporation from Rain

### Physiological Variables
- **NEE**: Net Ecosystem Exchange (μmol CO₂ m⁻² s⁻¹)
  - Negative = Carbon sink (CO₂ uptake)
  - Positive = Carbon source (CO₂ release)
- **PSII**: Photosystem II efficiency (Fv/Fm ratio)
  - Range: 0.0–0.83 (higher = more photosynthetic)

---

## Interpretation Guide

### Carbon Flux (NEE)
- **Negative NEE**: Ecosystem absorbing CO₂ (good for climate)
- **Positive NEE**: Ecosystem releasing CO₂ (climate concern)
- Transitions occur at specific water table depths

### Photosynthetic Efficiency (PSII)
- Values drop steeply when VWC falls below species-specific thresholds
- Recovery is slower than the initial decline (hysteresis)
- Species show different crash points

### Drought Recovery
- Compares pre-drought (Wet) vs post-drought (Rewet) NEE
- Positive recovery indicates resilience
- Negative recovery indicates damage or community shifts

---

## Output Files

### Models Directory (`models/`)

| File | Type | Description |
|------|------|---|
| `xgboost_nee.joblib` | Binary | Trained XGBoost regressor |
| `lstm_drought.pth` | Binary | PyTorch LSTM state dict |
| `svr_psii.joblib` | Binary | Trained SVR model |
| `xgboost_metrics.txt` | Text | MSE, R² for XGBoost |
| `lstm_metrics.txt` | Text | MSE, R² for LSTM |
| `svr_metrics.txt` | Text | MSE, R² for SVR |
| `xgboost_conf_matrix.png` | Image | Confusion matrix heatmap |
| `lstm_conf_matrix.png` | Image | Confusion matrix heatmap |
| `svr_conf_matrix.png` | Image | Confusion matrix heatmap |

### Visualizations Directory (`visualizations/`)
- Exported publication-ready PNG figures (300 dpi recommended for printing)

---

## Troubleshooting

### FileNotFoundError
Ensure all `.dat` files are in the same directory as the Python scripts.

### Missing Data (-999 values)
These are automatically filtered during data loading. If too many records are filtered, check data quality.

### CUDA/GPU Issues (LSTM)
If GPU is not available, PyTorch will automatically use CPU. Training may be slower but results are identical.

### SKLearn StandardScaler Impact (SVR)
SVR is sensitive to feature scaling. StandardScaler is required for optimal performance.

---

## Citation

If using this analysis in research, please cite:
- Project: "Precipitation Frequency Effects on Sphagnum Carbon Uptake"
- Location: Peatland Sites, Ireland
- Data: Peatmoss physiological and ecosystem exchange measurements

---

## License

This project is provided as-is for research and educational purposes. Ensure proper licensing of underlying datasets before publication.

---

## Contact & Support

For questions about:
- **Model training/tuning**: See individual Python scripts for hyperparameter configuration
- **Data preprocessing**: Refer to comments in `visualizer.py` for missing data handling
- **Visualization**: Edit color schemes and plot parameters in `visualizer.py`

---

## Future Enhancements

- [ ] Add hyperparameter optimization (GridSearchCV)
- [ ] Implement cross-validation for more robust metrics
- [ ] Create ensemble models combining XGBoost + LSTM
- [ ] Add species-specific model variants
- [ ] Incorporate ensemble predictions for uncertainty quantification
- [ ] Build interactive dashboard for model exploration
- [ ] Add statistical significance testing for recovery metrics

---

## References

**Typical references for Sphagnum research:**
- Holden, J., et al. (2012) *Peatland hydrology and carbon release*. Nature Geoscience
- Laiho, R. (2006) *Decomposition in peatlands*. Mires and Peat
- Rydin, H. & Jeglum, J. K. (2013) *The Biology of Peatlands* (Oxford University Press)
