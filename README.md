# 🫀 CardioGuard AI: Cardiovascular Disease Risk Prediction Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-v1.5.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.66-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-93.44%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9665-success.svg)]()

> **CardioGuard AI** is a clinical-grade Machine Learning pipeline and Decision Support System for early, non-invasive risk stratification of Coronary Artery Disease (CAD). Engineered using an optimized Random Forest ensemble, stratified cross-validation, and an interactive diagnostic web portal.

---

## 🎯 Executive Overview & Clinical Context

Cardiovascular diseases (CVDs) remain the leading cause of global mortality (~17.9 million annual deaths). Traditional manual risk scoring systems (e.g., Framingham) rely on linear assumptions that fail to capture non-linear, multi-factorial physiological interactions. 

This repository implements a reproducible, end-to-end Machine Learning lifecycle adhering to the stages of modern AI software development:
1. **Curated Clinical Data:** UCI Cleveland Cardiovascular Disease Database (303 patients, 14 attributes).
2. **Descriptive & Exploratory Analysis:** Central tendency, skewness, kurtosis, and 5 exploratory diagnostic visualizations.
3. **Robust Preprocessing Pipeline:** Domain-informed imputation, 99th percentile Winsorization capping, nominal One-Hot Encoding, and strict data-leakage prevention.
4. **Stratified Cross-Validation:** 80/20 train-test partition with 5-Fold Stratified Cross-Validation.
5. **Hyperparameter Optimization:** Exhaustive `GridSearchCV` on Random Forest, scoring for ROC-AUC to minimize lethal False Negatives.
6. **Multi-Model Benchmark:** Comparative evaluation against Logistic Regression, Support Vector Machines (RBF), and Gradient Boosting.
7. **Production Deployment:** Interactive Streamlit Clinical Decision Support System (`app.py`) incorporating Docker, Kubernetes horizontal autoscaling, drift observability, and ethical AI governance.

---

## 📊 Benchmark Model Comparison (Unseen Test Partition, N=61)

| Model Architecture | Accuracy (%) | Precision (%) | Recall (Sensitivity) (%) | F1-Score (%) | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Tuned Random Forest (Primary)** | **93.44%** | **92.86%** | **92.86%** | **92.86%** | **0.9665** |
| 🌲 Baseline Random Forest | 86.89% | 83.33% | 89.29% | 86.21% | 0.9524 |
| ⚡ Support Vector Machine (RBF) | 93.44% | 90.00% | 96.43% | 93.10% | 0.9675 |
| 📈 Logistic Regression (L2) | 85.25% | 80.65% | 89.29% | 84.75% | 0.9567 |
| 🚀 Gradient Boosting (GBM) | 86.89% | 83.33% | 89.29% | 86.21% | 0.9437 |

### ⚙️ Optimal Random Forest Hyperparameters (via 5-Fold Stratified GridSearchCV):
- `n_estimators`: `150` (Eliminates bagging variance without latency bottlenecks)
- `max_depth`: `3` (Constrains tree growth to prevent memorizing physiological outliers)
- `min_samples_leaf`: `2` (Enforces regularized terminal leaf boundaries)
- `min_samples_split`: `2` (Optimal branch partition sensitivity)
- `criterion`: `'gini'` (Maximum impurity reduction speed)

---

## 📁 Repository Structure

```
├── app.py                              # Interactive Streamlit Clinical Decision Support System
├── train_and_evaluate.py               # Complete Python ML training & benchmark pipeline
├── heart_disease_practical_exam.ipynb  # Interactive Google Colab / Jupyter Notebook
├── best_random_forest_model.joblib     # Serialized optimal Random Forest model
├── scaler.joblib                       # Serialized StandardScaler (fitted strictly on X_train)
├── feature_names.json                  # Feature column mapping for inference
├── evaluation_summary.json             # Benchmark metrics JSON export
├── heart_disease_uci.csv               # Authentic UCI Cleveland cardiac dataset
├── heart_disease_preprocessed.csv      # Cleaned and one-hot encoded dataset
├── build_pdf_report.py                 # Automated PDF report compiler (ReportLab)
├── PROJECT_REPORT.pdf                  # Academic Practical Examination Lab Report (2.0 MB)
├── presentation.pptx                   # 12-Slide PowerPoint Presentation Deck
├── PRESENTATION_SLIDES.md              # Presentation transcript and slide notes
├── VIVA_VOCE_PREPARATION.md            # Comprehensive 25+ question viva defense guide
├── package_submission.py               # Automated submission zip packager
├── figures/                            # High-resolution publication-quality PNG charts
│   ├── 1_eda_distributions.png
│   ├── 2_correlation_heatmap.png
│   ├── 3_outlier_boxplots.png
│   ├── 4_categorical_risk_patterns.png
│   ├── 5_bivariate_scatter.png
│   ├── 6_confusion_matrices.png
│   ├── 7_roc_curves.png
│   ├── 8_feature_importances.png
│   └── 9_model_comparison_bar.png
└── README.md                           # Project documentation
```

---

## 🚀 Quickstart: Running the Application Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Ayush1702j/AIML_Mini_Project.git
cd AIML_Mini_Project
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
# Or install core packages directly:
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

### 3. Launch the Clinical Web Application
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

### 4. Re-run Model Training & Evaluation
```bash
python train_and_evaluate.py
```

---

## 🛡️ Production & Ethical Considerations

- **Scalability:** Multi-stage Docker containerization (`python:3.12-slim`), Kubernetes Horizontal Pod Autoscaler (HPA), and Celery/Redis asynchronous batch pipelines.
- **Reliability:** Evidently AI continuous data drift detection (Population Stability Index), Prometheus/Grafana telemetry, and automated health checks (`/healthz`).
- **Fairness & Ethics:** Demographic parity across gender/age to counter historical under-recognition of atypical female cardiac symptoms, Explainable AI (SHAP & Gini importance), and strict HIPAA/GDPR de-identification.
- **Human-in-the-Loop (HITL):** Operating strictly as a Clinical Decision Support System (CDSS) to augment, not replace, clinical authority.

---

## 🎓 Academic Submission Metadata

| Attribute | Details |
|:---|:---|
| **Institution** | MIT Academy of Engineering (MITAOE) |
| **Department** | Department of Computer Engineering (Program Elective) |
| **Course** | Third Year B.Tech Semester - V (2025–2026) |
| **Course Code** | 2304322L — Artificial Intelligence Practical Examination |
| **Candidate Name** | Ayush |
| **PRN Number** | 202401040128 |
| **Allocated Algorithm** | Random Forest Classifier |
| **Practical Date** | 20 November 2025 |
| **Max Marks** | 30 Marks |

---

## 📜 License
This project is open-source and distributed under the [MIT License](LICENSE).
