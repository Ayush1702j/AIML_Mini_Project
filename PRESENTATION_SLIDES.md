# MIT ACADEMY OF ENGINEERING
## Department of Computer Engineering | Third Year B.Tech (Sem-V) 2025–2026
### Course Code: 2304322L — Artificial Intelligence Practical Examination
**Candidate Name:** Ayush | **PRN Number:** 202401040128 | **Date:** 20 Nov 2025  
**Allocated Algorithm:** Random Forest Classifier  
**Project Title:** End-to-End Machine Learning Pipeline for Cardiovascular Disease Risk Prediction

---

## Slide 1: Title & Candidate Information
- **Institution:** MIT Academy of Engineering, Pune
- **Department:** Computer Engineering (Program Elective)
- **Exam:** Third Year B.Tech Semester-V Practical Examination (2025–2026)
- **Subject:** Artificial Intelligence (Course Code: 2304322L)
- **Candidate Name:** Ayush
- **PRN Number:** 202401040128
- **Allocated Algorithm:** Random Forest Classifier
- **Problem Statement:** Early Cardiovascular Disease Risk Prediction Using Hyperparameter-Tuned Random Forest, Stratified Cross-Validation, and Production Deployment.

---

## Slide 2: Problem Statement & Clinical Motivation
- **The Healthcare Burden:** Cardiovascular Diseases (CVDs) cause an estimated 17.9 million global deaths annually. Early non-invasive screening prevents acute myocardial infarction.
- **Why Traditional Scoring Falls Short:** Conventional risk charts (Framingham Score) assume linear independence and fail to capture non-linear, multi-factorial physiological interactions.
- **Our AI Solution:** An end-to-end, reproducible Machine Learning workflow featuring data curation, exploratory data analysis, robust preprocessing, cross-validation, hyperparameter tuning via GridSearchCV, benchmark comparative analysis, and a production Streamlit Clinical Decision Support System (CDSS).

---

## Slide 3: Section A — Data Collection & Clinical Features [CO1 L3]
- **Dataset Source:** Cleveland Cardiovascular Disease Database (UCI Machine Learning Repository).
- **Cohort Size:** 303 adult patients (164 healthy controls, 139 confirmed coronary disease cases).
- **Core Clinical Predictors:**
  - `age`: Patient age (29–77 yrs) — vascular compliance decreases with age.
  - `sex`: Biological sex (1=Male, 0=Female) — hormonal cardio-protection in females.
  - `cp`: Chest pain type (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic).
  - `trestbps`: Resting blood pressure (94–200 mmHg) on admission.
  - `chol`: Serum cholesterol (126–564 mg/dl) — low-density atheroma plaque driver.
  - `thalach`: Maximum heart rate during exercise (71–202 bpm) — chronotropic response.
  - `exang`: Exercise-induced angina (0=No, 1=Yes) — myocardial oxygen supply mismatch.
  - `oldpeak`: ST depression induced by exercise (0.0–6.2 mm) — primary ECG ischemic marker.
  - `ca`: Number of major coronary vessels colored by fluoroscopy (0–3 vessels).
  - `target`: Angiographic coronary artery disease (0: Healthy, 1: Heart Disease Present).

---

## Slide 4: Section B & C — Statistical Summary & Exploratory Insights [CO1 L3]
- **Statistical Operations:** Central tendency (Mean, Median), dispersion (Std Dev, IQR), and shape (Fisher-Pearson Skewness & Kurtosis).
- **Key Statistical Inferences:**
  - **Serum Cholesterol:** Positive skewness (+1.136) and leptokurtic distribution (Kurtosis = 4.492) with severe outliers > 400 mg/dl requiring Winsorization.
  - **Resting Blood Pressure:** Mean 131.69 mmHg (median 130 mmHg), indicating stage-1 hypertension tendencies.
  - **Maximum Heart Rate:** Negative skewness (-0.537); patients with coronary disease show impaired peak exertion rates.
- **5 Visualization Methods:** Univariate KDE distributions, Pearson correlation matrix heatmap, boxplots for outlier detection, categorical cross-tabulation, and bivariate scatter plots.
- **Clinical Takeaways:**
  - `thalach < 140 bpm` strongly correlates with coronary disease.
  - **The Asymptomatic Paradox:** Chest pain type 4 (asymptomatic) shows the highest rate of severe disease due to silent myocardial ischemia.

---

## Slide 5: Section D & E — Preprocessing & Cross-Validation [CO1 L3, CO2 L3]
- **Data Wrangling & Cleaning (Section D):**
  - **Imputation:** Median for ordinal `ca` (4 missing) and mode for nominal `thal` (2 missing) preserving discrete medical boundaries.
  - **Winsorization:** Capped `chol` at 99th percentile (406.7 mg/dl) and `trestbps` (180.0 mmHg) to prevent leverage distortion.
  - **One-Hot Encoding:** Applied `pd.get_dummies(drop_first=True)` to prevent artificial numerical hierarchies in chest pain and ECG categories.
  - **Feature Scaling:** `StandardScaler` fitted strictly on `X_train` to prevent data leakage.
- **Stratified Partitioning (Section E):**
  - Stratified 80/20 Train-Test split (242 train, 61 test) preserving exact 45.9% disease prevalence.
  - Stratified 5-Fold Cross-Validation: Fold 1 (83.67%), Fold 2 (81.63%), Fold 3 (79.17%), Fold 4 (79.17%), Fold 5 (77.08%).
  - **Mean CV Accuracy:** **80.14% (± 2.28%)** — proving high generalization stability.

---

## Slide 6: Section F — Random Forest Implementation & Tuning [CO2 L4]
- **Allocated Algorithm:** Random Forest Classifier (Ensemble of $N$ decision trees trained on bootstrap samples with random feature subspace sampling).
- **Hyperparameter Optimization via GridSearchCV (5-Fold CV, Scored by ROC-AUC):**
  - `n_estimators = 150`: Sufficient bagging aggregation to eliminate ensemble variance.
  - `max_depth = 3`: Strict tree depth restriction to prevent memorizing physiological noise (vital regularizer).
  - `min_samples_leaf = 2`: Mandates at least 2 samples per terminal node to smooth decision boundaries.
  - `criterion = 'gini'`: Optimal impurity reduction with high computational efficiency.
- **Best CV ROC-AUC Score:** **0.8979**.

---

## Slide 7: Section G — Unseen Test Evaluation & Benchmark Comparison [CO3 L5]
- **Tuned Random Forest Performance on Unseen Test Partition (N=61):**
  - **Test Accuracy:** **93.44%** (Up from 86.89% baseline)
  - **Precision:** **92.86%**
  - **Recall (Sensitivity):** **92.86%** (Only 2 False Negatives out of 28 cardiac patients!)
  - **F1-Score:** **92.86%**
  - **ROC-AUC Score:** **0.9665**
- **Benchmark Comparison Across Algorithms:**
  | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
  |:---|:---:|:---:|:---:|:---:|:---:|
  | **Tuned Random Forest** | **93.44%** | **92.86%** | **92.86%** | **92.86%** | **0.9665** |
  | Baseline Random Forest | 86.89% | 83.33% | 89.29% | 86.21% | 0.9524 |
  | Support Vector Machine (RBF) | 93.44% | 90.00% | 96.43% | 93.10% | 0.9675 |
  | Logistic Regression (L2) | 85.25% | 80.65% | 89.29% | 84.75% | 0.9567 |
  | Gradient Boosting (GBM) | 86.89% | 83.33% | 89.29% | 86.21% | 0.9437 |

---

## Slide 8: Section G — Diagnostic Tradeoffs: Precision vs Recall
- **The Clinical Priority of Recall:** In cardiology, the asymmetric loss function heavily penalizes False Negatives (discharging a patient with occult CAD risks imminent cardiac arrest).
- **Minimizing False Negatives:** Tuned RF achieved 92.86% Recall, misclassifying only 2 diseased patients while retaining 92.86% precision.
- **Non-Linear Superiority:** Random Forest and SVM outperformed linear Logistic Regression by +8.19% accuracy, demonstrating the necessity of modeling multi-attribute interactions.

---

## Slide 9: Section H — Production Deployment & Scalability [CO4 L5]
- **Interactive Web Application (`app.py`):** Built with Streamlit, enabling clinicians to input physiological parameters and receive real-time risk stratification and feature contributions.
- **Microservice Architecture:**
  - Front-end decoupled from backend inference using a FastAPI REST API / gRPC service.
  - Multi-stage Docker container (`python:3.12-slim`) with an image footprint under 250 MB.
- **Horizontal Scalability:**
  - Kubernetes Horizontal Pod Autoscaler (HPA) on AWS EKS / GCP GKE dynamically scales pods from 2 to 10 instances when CPU > 70%.
  - Asynchronous batch inference for overnight hospital EHR screening of 50,000+ hospitalized patients using Celery and Redis message queues.

---

## Slide 10: Section H — Reliability, Observability & Ethical AI [CO4 L5]
- **Reliability & Monitoring:**
  - **Data Drift Detection:** Evidently AI computes Population Stability Index (PSI) on incoming patient distributions; PSI > 0.2 triggers automatic retraining alerts.
  - **Prometheus & Grafana:** Monitors p99 inference latency (<15 ms), HTTP error rates, and throughput.
  - **Health Probes:** Kubernetes `/healthz` readiness probes; corrupted sensor data triggers a safe fallback requiring manual clinician review.
- **Ethical AI & Regulatory Governance:**
  - **Demographic Fairness:** Classification thresholds are calibrated across gender and age to prevent higher False Negative rates in female cohorts who present with atypical symptoms.
  - **Explainability (XAI):** Gini importance and SHAP explanations provide full transparency into model reasoning.
  - **HIPAA & GDPR Compliance:** Zero Protected Health Information (PHI) is passed to the model inference layer.
  - **Human-in-the-Loop (HITL):** Deployed strictly as a Clinical Decision Support System (CDSS) to assist doctors, never as an autonomous prescribing authority.

---

## Slide 11: Section I — Key Viva-Voce Questions & Defense
1. **Why Random Forest over a single Decision Tree?**  
   *Decision trees suffer from high variance and overfit clinical noise. Random Forest reduces variance via Bootstrap Aggregation (bagging) and random subspace feature sampling.*
2. **Why optimize for ROC-AUC rather than pure Accuracy?**  
   *ROC-AUC evaluates performance across all possible diagnostic thresholds, balancing Sensitivity (Recall) and Specificity independently of class threshold bias.*
3. **How did `max_depth = 3` improve model generalization?**  
   *Unconstrained trees expand to 100% leaf purity by memorizing isolated patient anomalies. Setting `max_depth = 3` regularizes tree capacity and forces the model to capture broader pathophysiological rules.*
4. **How did you prevent Data Leakage?**  
   *`StandardScaler` was fitted strictly on `X_train` and applied via `.transform()` to `X_test`. Imputation medians and Winsorization caps were derived exclusively from training data.*

---

## Slide 12: Conclusion & Summary
- **Complete End-to-End Workflow:** Systematically developed and analyzed each stage of the ML lifecycle (Sections A through I).
- **High-Performance Results:** Tuned Random Forest achieved **93.44% Test Accuracy**, **92.86% Recall**, and **0.9665 ROC-AUC**.
- **Production-Ready Engineering:** Interactive Streamlit deployment, containerization blueprint, drift observability, and ethical healthcare AI safeguards.
- **Demonstration:** Streamlit CDSS application ready for live demonstration.
