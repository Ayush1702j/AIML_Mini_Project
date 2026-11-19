# 🫀 CardioGuard AI: Cardiovascular Disease Risk Prediction Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-v1.5.2-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v3.4-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.66-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-93.44%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9767-success.svg)]()

> **CardioGuard AI** is a clinical-grade Machine Learning pipeline and Decision Support System for early, non-invasive risk stratification of Coronary Artery Disease (CAD). Systematically benchmarked across **21 distinct Machine Learning algorithms** spanning all major paradigms of AI.

---

## 🎯 Executive Overview & Clinical Context

Cardiovascular diseases (CVDs) remain the leading cause of global mortality (~17.9 million annual deaths). Traditional manual risk scoring systems (e.g., Framingham) rely on linear assumptions that fail to capture non-linear, multi-factorial physiological interactions. 

This repository implements a reproducible, end-to-end Machine Learning lifecycle adhering to all stages of modern AI software development:
1. **Curated Clinical Data:** UCI Cleveland Cardiovascular Disease Database (303 patients, 14 clinical attributes).
2. **Exploratory Data Analysis (EDA):** Skewness, kurtosis, distributions, outlier bounds, and correlation structure.
3. **Robust Preprocessing Pipeline:** Domain-informed median/mode imputation, 99th percentile Winsorization capping, nominal One-Hot Encoding, and strict data-leakage prevention (preprocessors fit strictly on training splits).
4. **Stratified Cross-Validation:** 80/20 train-test partition with 5-Fold Stratified Cross-Validation.
5. **Comprehensive Multi-Algorithm Benchmark:** Systematic training and evaluation across 21 classification models covering all 5 core machine learning paradigms.
6. **Hyperparameter Tuning:** Stratified `GridSearchCV` optimization targeting ROC-AUC and Recall (Sensitivity) to minimize life-threatening False Negatives.
7. **Production Clinical Decision Support System (Streamlit):** Web application featuring real-time risk scoring, dynamic model selection across all 21 algorithms, interactive biomarker radar, and clinical risk breakdown.

---

## 📊 Comprehensive 21-Algorithm Benchmark (Unseen Test Partition, N=61)

| Model Architecture | Algorithm Family | Accuracy (%) | Precision (%) | Recall (Sensitivity) (%) | Specificity (%) | F1-Score (%) | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **K-Nearest Neighbors (KNN, k=7)** | Bayesian & Distance | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9767** |
| 📦 **Stacking Classifier (Meta-Learner)** | Meta-Ensembles | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9708** |
| ⚡ **Support Vector Machine (RBF Kernel)** | Support Vector Machines | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9675** |
| 🌲 **Tuned Random Forest (Primary)** | Decision Trees & Ensembles | **93.44%** | **92.86%** | **92.86%** | **93.94%** | **92.86%** | **0.9665** |
| 🧬 **Bernoulli Naive Bayes** | Bayesian & Distance | **91.80%** | 87.10% | **96.43%** | 87.88% | **91.53%** | **0.9654** |
| 🌳 **Extra Trees Classifier** | Decision Trees & Ensembles | **91.80%** | 89.66% | **92.86%** | 90.91% | **91.23%** | **0.9632** |
| 🌿 **AdaBoost Classifier** | Decision Trees & Ensembles | **90.16%** | 86.67% | **92.86%** | 87.88% | **89.66%** | **0.9708** |
| ⚡ **Hist Gradient Boosting** | Gradient Boosting | **90.16%** | 86.67% | **92.86%** | 87.88% | **89.66%** | **0.9621** |
| 📊 **Soft Voting Ensemble** | Meta-Ensembles | **90.16%** | 89.29% | **89.29%** | 90.91% | **89.29%** | **0.9621** |
| 🚀 **XGBoost Classifier** | Gradient Boosting | **90.16%** | 86.67% | **92.86%** | 87.88% | **89.66%** | **0.9502** |
| 📐 Support Vector Machine (Poly) | Support Vector Machines | 88.52% | 92.00% | 82.14% | 93.94% | 86.79% | 0.9405 |
| 📈 Ridge Classifier | Linear & Discriminant | 86.89% | 81.25% | 92.86% | 81.82% | 86.67% | 0.9556 |
| 📉 Logistic Regression (L1 Lasso) | Linear & Discriminant | 86.89% | 83.33% | 89.29% | 84.85% | 86.21% | 0.9524 |
| 🌲 Baseline Random Forest | Decision Trees & Ensembles | 86.89% | 83.33% | 89.29% | 84.85% | 86.21% | 0.9524 |
| 🧠 Multi-Layer Perceptron (Neural Net) | Neural Networks | 86.89% | 83.33% | 89.29% | 84.85% | 86.21% | 0.9491 |
| 🌲 Gradient Boosting Classifier (GBM) | Gradient Boosting | 86.89% | 83.33% | 89.29% | 84.85% | 86.21% | 0.9437 |
| 📈 Logistic Regression (L2) | Linear & Discriminant | 85.25% | 80.65% | 89.29% | 81.82% | 84.75% | 0.9567 |
| 📐 Support Vector Machine (Linear) | Support Vector Machines | 85.25% | 80.65% | 89.29% | 81.82% | 84.75% | 0.9545 |
| 🔬 Linear Discriminant Analysis (LDA) | Linear & Discriminant | 83.61% | 80.00% | 85.71% | 81.82% | 82.76% | 0.9491 |
| 📉 SGD Classifier (Log Loss) | Linear & Discriminant | 83.61% | 80.00% | 85.71% | 81.82% | 82.76% | 0.9318 |
| 🌲 Decision Tree (CART) | Decision Trees & Ensembles | 75.41% | 78.26% | 64.29% | 84.85% | 70.59% | 0.7781 |
| 🧬 Gaussian Naive Bayes | Bayesian & Distance | 60.66% | 54.35% | 89.29% | 36.36% | 67.57% | 0.8474 |

---

## 📁 Repository Structure

```
├── app.py                              # Streamlit Clinical Decision Support System with dynamic model switching
├── train_and_evaluate.py               # Complete 21-algorithm training & evaluation pipeline
├── heart_disease_practical_exam.ipynb  # Comprehensive Jupyter Notebook with all 21 models
├── all_models.joblib                   # Serialized bundle of all 21 trained ML models
├── best_random_forest_model.joblib     # Serialized tuned Random Forest model
├── scaler.joblib                       # Serialized StandardScaler (fitted strictly on X_train)
├── all_algorithms_benchmark.csv        # Tabular evaluation results across all 21 models
├── evaluation_summary.json             # Benchmark metrics JSON export
├── heart_disease_uci.csv               # Authentic UCI Cleveland cardiac dataset
├── heart_disease_preprocessed.csv      # Cleaned and preprocessed dataset
├── generate_notebook.py                # Automated Jupyter Notebook generator
├── build_pdf_report.py                 # PDF report builder script
├── package_submission.py               # Package generation script
├── requirements.txt                    # Project dependencies
├── figures/                            # High-resolution publication-quality PNG charts
│   ├── 1_eda_distributions.png
│   ├── 2_correlation_heatmap.png
│   ├── 3_outlier_boxplots.png
│   ├── 4_categorical_risk_patterns.png
│   ├── 5_bivariate_scatter.png
│   ├── 6_confusion_matrices.png
│   ├── 7_roc_curves.png
│   ├── 8_feature_importances.png
│   ├── 9_model_comparison_bar.png
│   └── 10_algorithm_family_heatmap.png
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
The dashboard will open automatically in your browser at `http://localhost:8501`. You can pick **any** of the 21 machine learning algorithms from the dropdown to run live predictions.

### 4. Re-run Full Multi-Algorithm Training & Benchmark
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
