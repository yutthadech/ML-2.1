# 🐍 Pack SGA - Python Scripts Documentation

## 📋 **ไฟล์ทั้งหมด (5 ไฟล์)**

| # | File | Size | Purpose | Type |
|---|------|------|---------|------|
| 1 | `app.py` | 12 KB | Streamlit web application | **Deployment** |
| 2 | `pack_ml_pipeline.py` | 33 KB | Complete ML training pipeline | **Training** |
| 3 | `predict.py` | 9.5 KB | Standalone prediction (CLI) | **Prediction** |
| 4 | `test_model.py` | 7.1 KB | Model testing & evaluation | **Testing** |
| 5 | `retrain_simple.py` | 7.8 KB | Quick model retraining | **Training** |

---

## 1️⃣ **app.py** - Streamlit Web Application

### Purpose
Main web application สำหรับ deploy บน Streamlit Cloud

### Features
- 🔮 Single Prediction - ทำนายแบบ real-time
- 📊 Batch Prediction - Upload CSV/Excel ทำนายเป็นชุด
- 📈 Model Analytics - ดูกราฟและ metrics
- ℹ️ About - คู่มือการใช้งาน

### Usage
```bash
# Local testing
streamlit run app.py

# Access at: http://localhost:8501
```

### Deployment
Upload to GitHub → Deploy on Streamlit Cloud

**Required files:**
- app.py
- pack_model.pkl
- requirements.txt

---

## 2️⃣ **pack_ml_pipeline.py** - Complete ML Pipeline

### Purpose
ระบบ ML แบบสมบูรณ์สำหรับ train model ตั้งแต่ต้นจนจบ

### Features
- ✅ Load & clean data
- ✅ Feature engineering
- ✅ Train 3 models (Random Forest, Gradient Boosting, Ridge)
- ✅ Cross-validation
- ✅ Create 9 Minitab-style visualizations
- ✅ Export model + summary
- ✅ Generate Streamlit app

### Usage
```bash
# Run complete pipeline
python pack_ml_pipeline.py

# Output files:
# - pack_model.pkl (trained model)
# - model_summary.xlsx (comparison & metrics)
# - 01-09_*.png (9 visualization charts)
# - app.py (Streamlit application)
# - requirements.txt (dependencies)
```

### When to Use
- เมื่อมีข้อมูลใหม่ต้องการ train model ตั้งแต่ต้น
- ต้องการสร้างกราฟวิเคราะห์ทั้งหมด
- ต้องการเปรียบเทียบหลาย models

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap openpyxl
```

---

## 3️⃣ **predict.py** - Standalone Prediction

### Purpose
ทำ prediction แบบ command line (ไม่ต้องใช้ Streamlit)

### Features
- Single prediction mode
- Batch prediction mode
- Export results to CSV/Excel
- Summary statistics

### Usage

#### Single Prediction
```bash
python predict.py --single \
  --shift A \
  --furnace 2 \
  --line 1 \
  --customer "Customer A" \
  --type "Type1" \
  --description "Product1" \
  --total_fg 200000 \
  --year 2025 \
  --month 11 \
  --day 9

# Output:
# Predicted %Pack: 0.9123 (91.23%)
```

#### Batch Prediction
```bash
python predict.py --batch input.csv --output predictions.csv

# Input: CSV/Excel with columns matching training data
# Output: Same file + Predicted_%Pack column
```

### Options
```
--model MODEL       Model file path (default: pack_model.pkl)
--single            Single prediction mode
--batch FILE        Batch prediction from file
--output FILE       Output file for batch predictions
--shift {A,B,C}     Shift
--furnace INT       Furnace number
--line INT          Line number
--customer STR      Customer name
--type STR          Product type
--description STR   Product description
--total_fg INT      Total FG (bottles)
--year INT          Year
--month INT         Month (1-12)
--day INT           Day (1-31)
--day_of_week INT   Day of week (0-6)
--quarter INT       Quarter (1-4)
```

### When to Use
- ต้องการ automate predictions
- Integration กับระบบอื่น
- ทำ prediction แบบ batch processing
- ไม่ต้องการ UI

---

## 4️⃣ **test_model.py** - Model Testing

### Purpose
ทดสอบ accuracy ของ model กับข้อมูล test set

### Features
- Calculate R², RMSE, MAE
- Cross-validation
- Predictions summary
- Export predictions with errors

### Usage
```bash
python test_model.py --data test_data.xlsx --output test_results.csv

# Output:
# ================================================================================
# MODEL PERFORMANCE REPORT
# ================================================================================
# R² Score:           0.7888
# RMSE:               0.0442
# MAE:                0.0160
# CV R² (mean±std):   0.7737 ± 0.0436
# ================================================================================
```

### Options
```
--model MODEL       Model file path (default: pack_model.pkl)
--data FILE         Test data file (must have %Pack column)
--output FILE       Save predictions with errors (optional)
--cv INT            Cross-validation folds (default: 5)
```

### When to Use
- ตรวจสอบ accuracy หลัง train model
- Validate model กับข้อมูลใหม่
- เปรียบเทียบ model versions
- Generate performance report

---

## 5️⃣ **retrain_simple.py** - Quick Retraining

### Purpose
Retrain model แบบเร็วด้วยข้อมูลใหม่

### Features
- Simple 1-command retraining
- Automatic feature engineering
- Train-test split
- Save new model

### Usage
```bash
python retrain_simple.py --data new_data.xlsx --output new_model.pkl

# With custom parameters
python retrain_simple.py \
  --data new_data.xlsx \
  --output new_model.pkl \
  --test-size 0.2 \
  --n-estimators 200 \
  --max-depth 25

# Output:
# ================================================================================
# RETRAINING COMPLETE
# ================================================================================
# Model saved:        new_model.pkl
# Test R² Score:      0.7888
# Test RMSE:          0.0442
# ================================================================================
```

### Options
```
--data FILE           Training data file (required)
--output FILE         Output model file (default: pack_model_new.pkl)
--test-size FLOAT     Test set proportion (default: 0.2)
--n-estimators INT    Number of trees (default: 100)
--max-depth INT       Max tree depth (default: 20)
```

### When to Use
- มีข้อมูลใหม่ต้องการ update model
- ต้องการ retrain แบบเร็ว (ไม่ต้องการกราฟ)
- Test parameters ต่างๆ

---

## 🚀 **Quick Start Guide**

### Scenario 1: First Time Setup
```bash
# 1. Train model
python pack_ml_pipeline.py

# 2. Test model
python test_model.py --data Pack_SGA_2425.xlsx

# 3. Deploy
streamlit run app.py
```

### Scenario 2: Make Predictions
```bash
# Single
python predict.py --single --shift A --furnace 2 ...

# Batch
python predict.py --batch input.csv --output predictions.csv
```

### Scenario 3: Update Model
```bash
# Quick retrain
python retrain_simple.py --data new_data.xlsx --output new_model.pkl

# Test new model
python test_model.py --model new_model.pkl --data test.xlsx

# Replace old model
mv new_model.pkl pack_model.pkl
```

### Scenario 4: Complete Pipeline
```bash
# Full pipeline with all visualizations
python pack_ml_pipeline.py
```

---

## 📊 **Workflow Comparison**

| Task | Fast Method | Complete Method |
|------|-------------|-----------------|
| **Train Model** | retrain_simple.py (2 min) | pack_ml_pipeline.py (5 min) |
| **Test Model** | test_model.py (30 sec) | Full evaluation in pipeline |
| **Predict** | predict.py (instant) | Streamlit app (interactive) |
| **Deploy** | Upload app.py + model | Full package with charts |

---

## 🔧 **Common Tasks**

### Update Model with New Data
```bash
# Step 1: Retrain
python retrain_simple.py --data updated_data.xlsx --output pack_model.pkl

# Step 2: Test
python test_model.py --data test_set.xlsx

# Step 3: Redeploy (if using Streamlit Cloud)
git add pack_model.pkl
git commit -m "Update model with new data"
git push origin main
```

### Batch Processing Workflow
```bash
# Process multiple files
for file in data/*.xlsx; do
    python predict.py --batch "$file" --output "predictions/$(basename $file)"
done
```

### Model Comparison
```bash
# Train different models
python retrain_simple.py --data data.xlsx --n-estimators 100 --output model_100.pkl
python retrain_simple.py --data data.xlsx --n-estimators 200 --output model_200.pkl

# Test both
python test_model.py --model model_100.pkl --data test.xlsx
python test_model.py --model model_200.pkl --data test.xlsx
```

---

## 📦 **Dependencies**

All scripts require:
```txt
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
```

Additional for specific scripts:
```txt
# pack_ml_pipeline.py
matplotlib>=3.8.0
seaborn>=0.13.0
shap>=0.44.0
openpyxl>=3.1.0

# app.py
streamlit>=1.32.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## ⚠️ **Important Notes**

### File Compatibility
- All scripts work with `.csv` and `.xlsx` files
- Data must include `%Pack` column for training/testing
- Feature columns must match training data for predictions

### Model Files
- Model files (`.pkl`) are version-specific
- Re-train if changing package versions
- Always test after retraining

### Python Version
- Recommended: Python 3.11
- Supported: Python 3.9-3.13
- See `runtime.txt` for deployment

---

## 🆘 **Troubleshooting**

### Error: "Model file not found"
```bash
# Make sure pack_model.pkl is in current directory
ls -lh pack_model.pkl

# Or specify path
python predict.py --model /path/to/pack_model.pkl --single ...
```

### Error: "Module not found"
```bash
# Install dependencies
pip install -r requirements.txt
```

### Error: "Invalid column names"
```bash
# Check your data columns match training data
python -c "import pickle; print(pickle.load(open('pack_model.pkl','rb'))['feature_names'])"
```

### Error: "Memory error during training"
```bash
# Reduce data size or model complexity
python retrain_simple.py --data data.xlsx --n-estimators 50 --max-depth 15
```

---

## 📚 **Examples**

### Example 1: Complete Workflow
```bash
# Train
python pack_ml_pipeline.py

# Single prediction
python predict.py --single --shift A --furnace 2 --line 1 \
  --customer "Customer A" --type "Type1" --description "Desc1" \
  --total_fg 200000

# Batch prediction
python predict.py --batch test_data.csv --output results.csv

# Test accuracy
python test_model.py --data test_data.csv
```

### Example 2: Production Update
```bash
# Retrain with new data
python retrain_simple.py --data latest_data.xlsx

# Validate
python test_model.py --data validation_set.xlsx

# Deploy
streamlit run app.py
```

### Example 3: Automation Script
```bash
#!/bin/bash
# Daily prediction automation

DATE=$(date +%Y%m%d)
python predict.py --batch daily_data.csv --output predictions_$DATE.csv
echo "Predictions saved to predictions_$DATE.csv"
```

---

## 🎯 **Best Practices**

1. **Always test** after retraining model
2. **Backup** old models before replacing
3. **Validate** predictions on known data
4. **Document** parameter changes
5. **Version control** models with git

---

**Created by:** Kyoko (MIT USA)  
**Last Updated:** 2025-11-09  
**Version:** 2.0 (Python 3.13 compatible)

---

## 📞 **Support**

For issues or questions:
1. Check error messages carefully
2. Verify data format matches training data
3. Ensure all dependencies installed
4. Review documentation above

**All scripts are ready to use! 🚀**
