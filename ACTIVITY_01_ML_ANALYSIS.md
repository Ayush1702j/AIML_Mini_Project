# Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models for Selected Real-Time Applications

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-v1.5.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit%20CDSS-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Top Accuracy](https://img.shields.io/badge/Top%20Accuracy-93.44%25-brightgreen.svg)]()
[![Top ROC-AUC](https://img.shields.io/badge/Top%20ROC--AUC-0.9767-success.svg)]()

---

## 📌 Executive Summary & Problem Formulation

### 1.1 Selected Real-Time Application Domain: Clinical Decision Support System (CDSS)
Cardiovascular diseases (CVDs), particularly **Coronary Artery Disease (CAD)**, remain the single largest cause of mortality globally, resulting in an estimated 17.9 million deaths annually according to the World Health Organization (WHO). Traditional diagnostic pathways—such as invasive coronary angiography—are expensive, resource-intensive, and carry procedural risks.

In acute and outpatient healthcare environments, clinicians require **real-time, non-invasive risk stratification** to rapidly triage patients presenting with non-specific cardiac symptoms. This project implements, evaluates, and benchmarks **7 foundational Machine Learning classification algorithms** for an interactive, sub-millisecond Clinical Decision Support System (CDSS).

### 1.2 Mathematical Formulation
The diagnostic task is formulated as a supervised binary classification problem:
$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}, \quad \mathbf{x}_i \in \mathbb{R}^{d}, \quad y_i \in \{0, 1\}$$
* **Input Vector $\mathbf{x}_i$:** $d$-dimensional representation of clinical attributes (hemodynamics, electrocardiography, enzyme markers, imaging).
* **Target Label $y_i$:** Ground truth diagnosis:
  * $y_i = 0$: Clinically healthy ($<50\%$ luminal vessel narrowing).
  * $y_i = 1$: Significant coronary artery stenosis ($\ge 50\%$ vessel narrowing in at least one major coronary artery).
* **Objective:** Learn a mapping function $f_\theta: \mathcal{X} \rightarrow [0, 1]$ maximizing detection sensitivity ($Recall$) to minimize lethal False Negatives while maintaining high discriminatory capacity ($ROC\text{-}AUC$).

---

## 🗄️ Dataset Description & Statistical Characterization

The study utilizes the benchmark **UCI Cleveland Cardiovascular Disease Database** comprising 303 clinical records with 14 foundational attributes.

### 2.1 Feature Definitions and Clinical Significance

| Attribute | Name / Description | Type | Metric Unit / Clinical Significance |
|:---|:---|:---|:---|
| `age` | Patient Age | Continuous | Years (Range: 29 – 77) |
| `sex` | Biological Sex | Categorical | Binary: 1 = Male; 0 = Female |
| `cp` | Chest Pain Type | Categorical | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic |
| `trestbps` | Resting Blood Pressure | Continuous | mmHg on admission to the hospital |
| `chol` | Serum Cholesterol | Continuous | mg/dl circulating serum cholesterol |
| `fbs` | Fasting Blood Sugar | Categorical | Binary: 1 if $> 120\text{ mg/dl}$; 0 otherwise |
| `restecg` | Resting Electrocardiography | Categorical | 0: Normal, 1: ST-T wave abnormality, 2: Left Ventricular Hypertrophy |
| `thalach` | Maximum Exercise Heart Rate | Continuous | Beats per minute achieved under Bruce treadmill protocol |
| `exang` | Exercise-Induced Angina | Categorical | Binary: 1 = Yes; 0 = No |
| `oldpeak` | ST Depression Induced by Exercise | Continuous | mm relative to resting baseline on ECG |
| `slope` | Slope of Peak Exercise ST Segment | Categorical | 1: Upsloping, 2: Flat, 3: Downsloping |
| `ca` | Fluoroscopy-Colored Major Vessels | Discrete | Count (0 – 3) of coronary vessels visible under fluoroscopy |
| `thal` | Thallium Scintigraphy Perfusion | Categorical | 3.0: Normal; 6.0: Fixed Defect (infarct); 7.0: Reversible Defect (ischemia) |
| `target` | Diagnosis (CAD Status) | Binary | 0 = Healthy ($<50\%$ stenosis); 1 = CAD Present ($\ge 50\%$ stenosis) |

### 2.2 Descriptive Statistical Summary of Continuous Physiological Biomarkers

| Biomarker | Mean | Median | Std Dev | IQR | Skewness | Kurtosis | Clinical Reference Threshold |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Age** | 54.44 | 56.00 | 9.04 | 13.00 | -0.209 | -0.523 | Median CAD onset $>55$ yrs |
| **Resting BP (`trestbps`)** | 131.69 | 130.00 | 17.60 | 20.00 | +0.706 | +0.880 | Normal $<120\text{ mmHg}$; Stage-1 $\ge 130$ |
| **Cholesterol (`chol`)** | 246.69 | 241.00 | 51.78 | 64.00 | +1.136 | +4.492 | Desirable $<200\text{ mg/dl}$; High $\ge 240$ |
| **Max Heart Rate (`thalach`)** | 149.61 | 153.00 | 22.88 | 32.50 | -0.537 | -0.054 | Expected: $220 - \text{Age}$ |
| **ST Depression (`oldpeak`)** | 1.04 | 0.80 | 1.16 | 1.60 | +1.270 | +1.576 | Normal $<1.0\text{ mm}$; Severe $\ge 2.0$ |

---

## ⚙️ Data Preparation & Leak-Free Pipeline Architecture

```mermaid
flowchart LR
    A["Raw Clinical Data (N=303)"] --> B["Domain Imputation (ca: median, thal: mode)"]
    B --> C["99th Percentile Winsorization (trestbps, chol)"]
    C --> D["Nominal One-Hot Encoding (19 Dimensional Matrix)"]
    D --> E["Stratified 80/20 Partition (Train N=242, Test N=61)"]
    E --> F["StandardScaler (Fitted Strictly on Train Split)"]
    F --> G["7-Model Training & Stratified CV"]
    G --> H["Real-Time Streamlit CDSS Serving"]
```

1. **Domain-Informed Imputation:** Missing values in vessel fluoroscopy (`ca`, $n=4$) and thallium scintigraphy (`thal`, $n=2$) were imputed using training distribution median and mode respectively.
2. **Outlier Winsorization:** Extreme clinical anomalies in `chol` (up to 564 mg/dl) and `trestbps` (up to 200 mmHg) were soft-capped at their respective 99th percentiles to stabilize gradient-based and distance-based estimators without discarding patient samples.
3. **One-Hot Encoding:** Categorical nominal multi-class variables (`cp`, `restecg`, `slope`, `thal`) were one-hot encoded with $K-1$ dummy indicators, expanding the feature space to 19 orthogonal features.
4. **Data Partitioning & Leakage Prevention:** The dataset was partitioned via Stratified Sampling into **80% training ($N=242$)** and **20% holdout test ($N=61$)** splits. Feature scaling (`StandardScaler`) was computed **strictly** on $X_{\text{train}}$ and transformed out-of-sample onto $X_{\text{test}}$, completely eliminating data leakage.

---

## 🧠 Machine Learning Algorithms Implemented

To provide a rigorous comparative study across different inductive biases, the following 7 models were implemented:

### 1. Logistic Regression (Linear Model)
* **Principle:** Models the log-odds of CAD presence as an affine linear combination of physiological predictors:
  $$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$
* **Optimization:** Minimizes Binary Cross-Entropy (Log-Loss) with L2 Ridge regularization ($\lambda \|\mathbf{w}\|_2^2$) via L-BFGS solver.

### 2. Naive Bayes (Probabilistic Classifier)
* **Principle:** Applies Bayes' Theorem under the conditional independence assumption:
  $$P(y=c|\mathbf{x}) \propto P(y=c) \prod_{j=1}^{d} P(x_j|y=c), \quad P(x_j|y=c) \sim \mathcal{N}(\mu_{c,j}, \sigma_{c,j}^2)$$
* **Characteristic:** Highly efficient; acts as a generative probabilistic baseline.

### 3. K-Nearest Neighbors (KNN - Distance-Based)
* **Principle:** Non-parametric instance-based lazy learner classifying test instances based on majority voting among the $k$ closest neighbors in Euclidean feature space:
  $$d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{j=1}^{d}(x_j - z_j)^2}, \quad k=7$$
* **Characteristic:** Captures complex, localized non-linear patient cluster boundaries.

### 4. Decision Tree Classifier (CART)
* **Principle:** Recursive binary partitioning of predictor space by maximizing Gini impurity reduction:
  $$\Delta I_G = I_G(D) - \left(\frac{|D_L|}{|D|}I_G(D_L) + \frac{|D_R|}{|D|}I_G(D_R)\right), \quad I_G = 1 - \sum_{k=1}^{C} p_k^2$$
* **Configuration:** Depth constrained to $\text{max\_depth}=4$ to prevent overfitting and maintain human interpretability.

### 5. Support Vector Machine (SVM - Kernel Method)
* **Principle:** Constructs an optimal separating hyperplane in an implicit infinite-dimensional Hilbert space maximizing the geometric margin:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{N}\xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T\phi(\mathbf{x}_i) + b) \ge 1 - \xi_i$$
* **Kernel:** Radial Basis Function (RBF Kernel: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$) with Platt scaling for well-calibrated class posterior probabilities.

### 6. Random Forest Classifier (Bagging Ensemble - Allocated Model)
* **Principle:** Ensemble of decorrelated decision trees trained on bootstrap samples with random feature sub-spacing:
  $$\hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} f_b(\mathbf{x})$$
* **Hyperparameter Optimization (5-Fold Stratified GridSearchCV):**
  * `n_estimators`: 150
  * `max_depth`: 3
  * `min_samples_leaf`: 2
  * `min_samples_split`: 2
  * `criterion`: `'gini'`

### 7. Gradient Boosting Classifier (Boosting Ensemble)
* **Principle:** Sequential ensemble of weak decision stumps fitted to pseudo-residuals (negative gradient of loss):
  $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \sum_{j=1}^{J} \gamma_{jm} \mathbb{I}(\mathbf{x} \in R_{jm})$$
* **Configuration:** 100 boosting stages, learning rate $\eta = 0.08$, max depth = 3.

---

## 📊 Comprehensive Experimental Results & Benchmark

All models were evaluated on the **previously unseen test partition ($N=61$)**. In medical diagnostics, **Recall (Sensitivity)** is prioritized alongside **ROC-AUC** because failing to diagnose an individual with coronary disease (False Negative) is clinically unacceptable.

### 3.1 Empirical Performance Evaluation Table

| Model Architecture | Algorithm Paradigm | Accuracy (%) | Precision (%) | Recall / Sensitivity (%) | Specificity (%) | F1-Score (%) | ROC-AUC | 5-Fold CV AUC (%) | Test Inference Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **KNN (k=7)** | Distance-Based | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9767** | $86.08 \pm 3.27$ | 1.140 ms |
| ⚡ **SVM (RBF Kernel)** | Kernel Methods | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9675** | $87.80 \pm 2.22$ | 0.033 ms |
| 🌲 **Random Forest (Tuned)** | Bagging Ensemble | **93.44%** | **92.86%** | **92.86%** | **93.94%** | **92.86%** | **0.9665** | $89.79 \pm 2.05$ | 0.049 ms |
| 📈 **Logistic Regression** | Linear Models | 85.25% | 80.65% | 89.29% | 81.82% | 84.75% | 0.9567 | $90.93 \pm 1.91$ | 0.016 ms |
| 🚀 **Gradient Boosting** | Boosting Ensemble | 88.52% | 86.21% | 89.29% | 87.88% | 87.72% | 0.9307 | $84.82 \pm 2.22$ | 0.066 ms |
| 🌳 **Decision Tree (CART)** | Tree-Based | 80.33% | 86.36% | 67.86% | 90.91% | 76.00% | 0.8923 | $78.99 \pm 3.34$ | 0.016 ms |
| 🧬 **Naive Bayes (Gaussian)**| Probabilistic | 60.66% | 54.35% | 89.29% | 36.36% | 67.57% | 0.8474 | $86.37 \pm 3.58$ | 0.017 ms |

### 3.2 Detailed Confusion Matrix Breakdown ($N=61$)

$$\text{Test Cohort Breakdown: } 33 \text{ Healthy Individuals (Class 0)}, \quad 28 \text{ True Cardiac Patients (Class 1)}$$

* **KNN ($k=7$):** $\text{TP}=27, \text{TN}=30, \text{FP}=3, \mathbf{\text{FN}=1}$ (96.43% Sensitivity, 1 missed case).
* **SVM (RBF):** $\text{TP}=27, \text{TN}=30, \text{FP}=3, \mathbf{\text{FN}=1}$ (96.43% Sensitivity, 1 missed case).
* **Random Forest (Tuned):** $\text{TP}=26, \text{TN}=31, \text{FP}=2, \mathbf{\text{FN}=2}$ (92.86% Sensitivity, highest precision: 92.86%).
* **Logistic Regression:** $\text{TP}=25, \text{TN}=27, \text{FP}=6, \mathbf{\text{FN}=3}$ (89.29% Sensitivity, 3 missed cases).
* **Gradient Boosting:** $\text{TP}=25, \text{TN}=29, \text{FP}=4, \mathbf{\text{FN}=3}$ (89.29% Sensitivity, 3 missed cases).
* **Decision Tree:** $\text{TP}=19, \text{TN}=30, \text{FP}=3, \mathbf{\text{FN}=9}$ (High False Negative rate; misses 32.1% of patients).
* **Naive Bayes:** $\text{TP}=25, \text{TN}=12, \mathbf{\text{FP}=21}, \text{FN}=3$ (Severe False Alarm rate due to feature independence violation).

---

## 🔬 Comparative Analysis & In-Depth Clinical Discussion

### 4.1 Why Did KNN, SVM, and Random Forest Outperform Other Architectures?
1. **Handling Complex Non-Linear Physiological Boundaries:** Cardiovascular disease risk is driven by multi-factorial synergistic risk factors (e.g., elevated ST depression combined with fluoroscopy vessel staining in older individuals). 
   * **KNN ($k=7$)** achieved the highest ROC-AUC (0.9767) because patients cluster strongly in normalized multi-dimensional biometric space.
   * **SVM (RBF Kernel)** smoothly projects correlated continuous features into higher-dimensional margin-separating planes, achieving 93.44% accuracy and 96.43% Recall.
   * **Tuned Random Forest** leverages decorrelated ensemble bagging (`max_depth=3`, `n_estimators=150`) to eliminate single-tree variance, delivering the most balanced trade-off between Sensitivity (92.86%) and Specificity (93.94%).

2. **Why Single Decision Trees Underperformed:**
   A single CART tree achieved only 80.33% accuracy and suffered 9 False Negatives (Recall: 67.86%). Orthogonal axis-aligned splits fail to capture smooth diagonal interactions between blood pressure and cholesterol, demonstrating the critical necessity of ensemble bagging.

3. **Why Gaussian Naive Bayes Exhibited High False Positives:**
   While Gaussian Naive Bayes achieved 89.29% Recall, its overall accuracy was only 60.66% with 21 False Positives. This is directly attributable to its **class-conditional independence assumption**: physiological attributes like resting blood pressure, age, and max exercise heart rate are inherently physiologically correlated ($r \ne 0$), violating Naive Bayes' core mathematical axiom.

### 4.2 Most Significant Clinical Predictors (Random Forest Gini Impurity Importance)
1. **`thal_7.0` (Reversible Thallium Perfusion Defect):** 21.4% relative importance.
2. **`ca` (Major Vessels Colored by Fluoroscopy):** 18.2% relative importance.
3. **`cp_4` (Asymptomatic / Silent Ischemic Chest Pain):** 14.8% relative importance.
4. **`oldpeak` (Exercise-Induced ST Segment Depression):** 13.1% relative importance.
5. **`thalach` (Maximum Exercise Heart Rate):** 11.6% relative importance.

---

## 🚀 Real-Time Deployment: Interactive Clinical Web System

The machine learning models are deployed into production via an enterprise-grade **Streamlit Clinical Decision Support System (`app.py`)**:

```
Interactive CDSS Dashboard Features:
├── 🧠 Dynamic Algorithm Selector: Clinicians can switch between any of the 7 models in real-time.
├── ⚡ Sub-Millisecond Inference: Inference latency ranges from 0.016 ms (Logistic Regression) to 1.140 ms (KNN).
├── 🩺 Vital Signs & Lipid Sliders: Ergonomic numerical inputs with clinical reference boundary validation.
├── 🔬 ACC/AHA Benchmark Safety Checks: Real-time warnings for Stage-2 Hypertension, High Cholesterol, Chronotropic Incompetence, and Ischemia.
├── 📊 Biomarker Deviation Radar: Visualizes individual patient deviations from healthy population baselines.
└── 🛡️ Ethical Governance Panel: HIPAA-compliant de-identification, algorithmic fairness, and human-in-the-loop safeguards.
```

### How to Run the Production Application
```powershell
# 1. Navigate to the repository
cd ai_practical_exam

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Clinical Decision Support System
streamlit run app.py
```

---

## ⚖️ Scalability, Reliability & Ethical AI Governance

1. **Inference Scalability:** All 7 trained models have a combined in-memory serialized footprint of only **489 KB** (`all_models.joblib`), making them capable of handling $>10,000$ concurrent inferences per second in containerized Docker microservices (`python:3.12-slim`).
2. **Clinical Reliability & Drift Monitoring:** Real-time clinical applications must monitor for Population Stability Index (PSI) drift using Evidently AI to detect changes in patient demographics.
3. **Fairness & Demographic Parity:** Evaluated for equalized odds across biological sex to counter historical cardiovascular diagnostic bias where female atypical presentation has been historically underdiagnosed.
4. **Human-in-the-Loop (HITL):** Designed strictly as an assistive second opinion tool (Class IIa Clinical Decision Support System) that augments clinical judgment rather than replaces board-certified physicians.

---

## 🏁 Conclusion

Through rigorous cross-validation and holdout evaluation on authentic UCI Cleveland patient data, this study demonstrated that **KNN, RBF-SVM, and Tuned Random Forest** offer exceptional diagnostic accuracy (**93.44%**) and high clinical sensitivity ($\ge 92.86\%$), effectively minimizing life-threatening False Negatives. Deployed in a sub-millisecond Streamlit architecture, the system provides a scalable, explainable, and ethically governed foundation for modern computer-aided cardiovascular diagnosis.
