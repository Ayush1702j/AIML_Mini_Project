# 🫀 CardioGuard AI: Cardiovascular Disease Risk Prediction Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-v1.5.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.66-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-93.44%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9767-success.svg)]()

> **CardioGuard AI** is a clinical-grade Machine Learning pipeline and Decision Support System for early, non-invasive risk stratification of Coronary Artery Disease (CAD). Systematically benchmarked across the **7 core Machine Learning classification algorithms**: Logistic Regression, Naive Bayes, KNN, Decision Tree, SVM, Random Forest, and Gradient Boosting.

---

## 🎯 Executive Overview & Clinical Context

Cardiovascular diseases (CVDs) remain the leading cause of global mortality (~17.9 million annual deaths). Traditional manual risk scoring systems (e.g., Framingham) rely on linear assumptions that fail to capture non-linear, multi-factorial physiological interactions. 

This repository implements a reproducible, end-to-end Machine Learning lifecycle adhering to all stages of modern AI software development:
1. **Curated Clinical Data:** UCI Cleveland Cardiovascular Disease Database (303 patients, 14 clinical attributes).
2. **Exploratory Data Analysis (EDA):** Skewness, kurtosis, distributions, outlier bounds, and correlation structure.
3. **Robust Preprocessing Pipeline:** Domain-informed median/mode imputation, 99th percentile Winsorization capping, nominal One-Hot Encoding, and strict data-leakage prevention (preprocessors fit strictly on training splits).
4. **Stratified Cross-Validation:** 80/20 train-test partition with 5-Fold Stratified Cross-Validation.
5. **Hyperparameter Tuning:** Stratified `GridSearchCV` optimization on Random Forest targeting ROC-AUC and Recall (Sensitivity) to minimize life-threatening False Negatives.
6. **7 Core Algorithm Benchmark:** Systematic comparative evaluation across all 7 designated classification models: Logistic Regression, Naive Bayes, KNN, Decision Tree, SVM, Random Forest, and Gradient Boosting.
7. **Production Clinical Decision Support System (Streamlit):** Web application featuring real-time risk scoring, dynamic model selection across all 7 algorithms, interactive biomarker radar, and clinical risk breakdown.

---

## 📊 Comprehensive 7-Algorithm Benchmark (Unseen Test Partition, N=61)

| Model Architecture | Algorithm Family | Accuracy (%) | Precision (%) | Recall (Sensitivity) (%) | Specificity (%) | F1-Score (%) | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **KNN (k=7)** | Distance-Based Models | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9767** |
| ⚡ **SVM (RBF Kernel)** | Kernel Methods | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9675** |
| 🌲 **Random Forest (Tuned)** | Ensemble (Bagging) | **93.44%** | **92.86%** | **92.86%** | **93.94%** | **92.86%** | **0.9665** |
| 📈 **Logistic Regression** | Linear Models | 85.25% | 80.65% | 89.29% | 81.82% | 84.75% | 0.9567 |
| 🚀 **Gradient Boosting** | Ensemble (Boosting) | 88.52% | 86.21% | 89.29% | 87.88% | 87.72% | 0.9307 |
| 🌳 **Decision Tree (CART)** | Tree-Based Models | 80.33% | 86.36% | 67.86% | 90.91% | 76.00% | 0.8923 |
| 🧬 **Naive Bayes (Gaussian)** | Probabilistic Models | 60.66% | 54.35% | 89.29% | 36.36% | 67.57% | 0.8474 |

---

## 📁 Repository Structure

```
├── app.py                              # Streamlit Clinical Decision Support System with dynamic model switching
├── train_and_evaluate.py               # Complete 7-algorithm training & evaluation pipeline
├── heart_disease_practical_exam.ipynb  # Interactive Jupyter Notebook with all 7 algorithms
├── all_models.joblib                   # Serialized bundle of the 7 trained ML models
├── best_random_forest_model.joblib     # Serialized optimal Random Forest model
├── scaler.joblib                       # Serialized StandardScaler (fitted strictly on X_train)
├── all_algorithms_benchmark.csv        # Tabular evaluation results across the 7 algorithms
├── evaluation_summary.json             # Benchmark metrics JSON export
├── heart_disease_uci.csv               # Authentic UCI Cleveland cardiac dataset
├── heart_disease_preprocessed.csv      # Cleaned and preprocessed dataset
├── generate_notebook.py                # Automated Jupyter Notebook generator
├── package_submission.py               # Package generation script
├── requirements.txt                    # Project dependencies
├── figures/                            # High-resolution publication-quality PNG charts
│   ├── 1_eda_distributions.png
│   ├── 2_correlation_heatmap.png
│   ├── 3_outlier_boxplots.png
│   ├── 4_categorical_risk_patterns.png
│   ├── 5_bivariate_scatter.png
│   ├── 6_confusion_matrices.png        # 2x4 grid for all 7 algorithms
│   ├── 7_roc_curves.png                # Multi-model ROC curves
│   ├── 8_feature_importances.png       # Random Forest Gini importance
│   ├── 9_model_comparison_bar.png      # Multi-metric grouped bar chart
│   └── 10_algorithm_family_heatmap.png # Performance matrix heatmap
└── README.md                           # Documentation and benchmark overview
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
```

### 3. Launch the Clinical Web Application
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`. You can pick **any of the 7 algorithms** (Logistic Regression, Naive Bayes, KNN, Decision Tree, SVM, Random Forest, Gradient Boosting) from the dropdown to run live predictions.

### 4. Re-run Multi-Algorithm Training & Benchmark
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

## 📜 License
This project is open-source and distributed under the [MIT License](LICENSE).
