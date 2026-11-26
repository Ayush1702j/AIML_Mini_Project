# CardioGuard AI: A Comparative Machine Learning Framework for Real-Time Clinical Decision Support in Coronary Artery Disease Risk Stratification

**Author:** Ayush  
**Department of Computer Engineering**  
**Course:** Artificial Intelligence & Machine Learning  
**Date:** November 2026  

---

### Abstract
*Cardiovascular diseases (CVDs), primarily Coronary Artery Disease (CAD), constitute the leading cause of global mortality, accounting for 17.9 million deaths annually. While invasive coronary angiography remains the gold standard for diagnosing anatomical stenosis, its high procedural cost, invasiveness, and logistical demands restrict its utility in routine screening and rapid emergency triage. This study presents **CardioGuard AI**, an end-to-end Machine Learning (ML) framework and interactive Clinical Decision Support System (CDSS) for non-invasive CAD risk stratification. Utilizing the benchmark UCI Cleveland Cardiovascular Disease dataset (303 patients, 14 clinical attributes), we engineered a leak-free preprocessing pipeline incorporating domain-informed median/mode imputation, 99th-percentile Winsorization soft-capping, and nominal one-hot encoding. We conducted a systematic, empirical comparative study across seven foundational machine learning classification algorithms representing distinct mathematical paradigms: Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors (KNN), Decision Trees (CART), Support Vector Machines (SVM), Random Forest, and Gradient Boosting. On an independent, previously unseen test partition ($N=61$), **KNN ($k=7$)**, **SVM with Radial Basis Function kernel**, and **GridSearchCV-tuned Random Forest** demonstrated superior diagnostic efficacy, each attaining an accuracy of **93.44%**. Crucially, from a clinical safety standpoint where false negatives present lethal consequences, KNN and SVM achieved the highest sensitivity (**Recall = 96.43%**, missing only 1 out of 28 cardiac cases) and an outstanding discriminatory capacity (**ROC-AUC of 0.9767 and 0.9675** respectively), while Tuned Random Forest attained the highest precision (**92.86%**) and specificity (**93.94%**). The validated models were deployed within an interactive, sub-millisecond Streamlit CDSS architecture featuring dynamic algorithm switching, automated American College of Cardiology/American Heart Association (ACC/AHA) physiological safety triggers, and algorithmic fairness controls, establishing a reproducible blueprint for translation into acute clinical workflows.*

**Keywords:** Coronary Artery Disease, Machine Learning, Clinical Decision Support System, Random Forest, Support Vector Machines, K-Nearest Neighbors, ROC-AUC, Algorithmic Fairness.

---

## 1. Introduction

Cardiovascular diseases (CVDs) remain the preeminent contributor to morbidity and mortality worldwide, representing approximately 32% of all global deaths. Among these, Coronary Artery Disease (CAD)—characterized by atherosclerotic plaque accumulation within the epicardial coronary arteries leading to myocardial ischemia or infarction—imposes an immense public health and economic burden. Conventional clinical assessment relies on multi-variable risk charts such as the Framingham Risk Score (FRS) or the Systematic Coronary Risk Evaluation (SCORE). However, these scoring instruments are grounded in rigid parametric assumptions that frequently underestimate risk in atypical presentations and fail to capture complex, non-linear interactions between disparate physiological biomarkers.

Although invasive coronary angiography remains the definitive diagnostic modality for quantifying arterial stenosis, its invasive nature carries procedural complications (e.g., vascular trauma, contrast-induced nephropathy, stroke) and requires specialized catheterization infrastructure. Consequently, there is an urgent clinical imperative for robust, non-invasive, computer-aided diagnostic tools capable of operating in real-time within outpatient and emergency departments.

Machine Learning (ML) presents an advantageous methodology for CAD risk stratification. By uncovering subtle, multi-dimensional geometric boundaries in patient physiological data, ML models can synthesize standard non-invasive indicators—such as resting hemodynamics, exercise electrocardiography (ECG), lipid panels, and fluoroscopy—into reliable, calibrated probabilities of disease. 

Despite substantial literature exploring automated heart disease detection, many published works suffer from methodological limitations:
1. **Pervasive Data Leakage:** Imputing missing values or fitting scaling preprocessors across the entire dataset prior to partitioning, yielding overly optimistic, non-generalizable accuracy estimates.
2. **Narrow Model Exploration:** Evaluating only one or two algorithms without systematically benchmarking across diverse inductive biases.
3. **Absence of Clinical Translation:** Presenting theoretical results in offline notebooks without developing real-time inference interfaces or evaluating model latency and ethical constraints.

To overcome these deficiencies, this research develops **CardioGuard AI**. The primary contributions of this paper are:
* **Rigorous Leak-Free Methodology:** Implementing a strictly segregated preprocessing architecture with domain-informed imputation, outlier Winsorization, and cross-validated parameter optimization.
* **Comprehensive Multi-Paradigm Benchmark:** Evaluating seven distinct ML classification algorithms on an unseen holdout cohort ($N=61$) across clinical diagnostic metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, 5-Fold Cross-Validation variance, and inference latency.
* **Production-Grade CDSS Deployment:** Translating the empirical models into an interactive, sub-millisecond Streamlit decision support interface featuring dynamic model switching and automated ACC/AHA clinical benchmark checks.

---

## 2. Related Work

The application of computational intelligence to cardiovascular medicine has evolved through distinct paradigms over recent decades:

### 2.1 Statistical and Linear Models
Early work predominantly leveraged multivariable Logistic Regression to derive parametric odds ratios for cardiac risk factors. While linear models provide direct interpretability via log-odds coefficients, their linear decision boundaries struggle to model non-linear physiological synergies, such as the compounding ischemic risk of concurrent ST-segment depression and elevated systolic blood pressure.

### 2.2 Instance-Based and Kernel Methods
K-Nearest Neighbors (KNN) and Support Vector Machines (SVM) have been widely investigated for biomedical classification. Cover and Hart established the theoretical foundations of nearest-neighbor rules, demonstrating that the asymptotic error rate of the 1-NN classifier is bounded by twice the Bayes error rate. In cardiac diagnostics, SVM with Radial Basis Function (RBF) kernels has frequently demonstrated high accuracy due to its ability to project non-linearly separable clinical measurements into infinite-dimensional reproducing kernel Hilbert spaces (RKHS). However, prior studies frequently overlook the hyperparameter sensitivity of the soft-margin penalty $C$ and kernel bandwidth $\gamma$, leading to brittle real-world generalization.

### 2.3 Tree Ensembles and Boosting Architectures
Ensemble learning techniques, particularly Bagging (Random Forests) and Boosting (Gradient Boosting Machines, AdaBoost), represent the prevailing standard for tabular biomedical datasets. Breiman demonstrated that Random Forests mitigate the inherent variance and high variance-to-bias ratio of single decision trees (CART) by training decorrelated trees over bootstrap replicates with randomized feature sub-spacing. Modern boosting algorithms sequentially fit shallow base learners to pseudo-residuals, optimizing arbitrary differentiable loss functions.

### 2.4 Research Gap
A significant proportion of contemporary literature reports classification accuracies in excess of 95% on the UCI Cleveland dataset. However, meta-analyses reveal that these figures are frequently artifacts of improper validation schemes—most notably scaling features before splitting or performing feature selection over the full sample. Furthermore, the clinical trade-off between **Precision** (avoiding unnecessary invasive angiograms) and **Recall** (avoiding fatal discharge of ischemic patients) is rarely operationalized into a production-ready application. This work specifically resolves these issues through strict leak-free partitioning and deployment into an interactive Clinical Decision Support System.

---

## 3. Materials and Methods

### 3.1 Clinical Dataset and Cohort Characteristics
The empirical investigation was conducted on the benchmark **UCI Cleveland Cardiovascular Disease Database**, provided by the Cleveland Clinic Foundation. The dataset encompasses 303 patient records, each characterized by 14 foundational clinical attributes comprising patient demographics, resting hemodynamics, exercise stress responses, cardiac scintigraphy, and fluoroscopy imaging.

The ground truth binary outcome variable ($y \in \{0, 1\}$) represents the angiographic presence or absence of coronary artery disease:
* $y = 0$: Clinically Healthy (less than 50% lumen diameter narrowing across all major epicardial coronary vessels).
* $y = 1$: CAD Present ($\ge 50\%$ diameter stenosis in at least one major coronary artery).

| Attribute | Clinical Description | Data Type | Units / Valid Range |
|:---|:---|:---|:---|
| `age` | Patient chronological age | Continuous | Years ($29 - 77$) |
| `sex` | Biological sex | Categorical | Binary: 1 = Male; 0 = Female |
| `cp` | Chest pain classification | Categorical | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic |
| `trestbps` | Resting systolic blood pressure | Continuous | mmHg on hospital admission ($94 - 200$) |
| `chol` | Total circulating serum cholesterol | Continuous | mg/dl ($126 - 564$) |
| `fbs` | Fasting blood sugar $> 120\text{ mg/dl}$ | Categorical | Binary: 1 = True; 0 = False |
| `restecg` | Resting electrocardiographic findings | Categorical | 0: Normal, 1: ST-T wave changes, 2: Left Ventricular Hypertrophy |
| `thalach` | Maximum achieved exercise heart rate | Continuous | Beats per minute ($71 - 202$) |
| `exang` | Exercise-induced angina | Categorical | Binary: 1 = Yes; 0 = No |
| `oldpeak` | Exercise-induced ST depression | Continuous | mm relative to resting baseline ($0.0 - 6.2$) |
| `slope` | Slope of peak exercise ST segment | Categorical | 1: Upsloping, 2: Flat, 3: Downsloping |
| `ca` | Major vessels visible under fluoroscopy| Discrete | Count of colored vessels ($0 - 3$) |
| `thal` | Thallium-201 myocardial scintigraphy | Categorical | 3.0: Normal; 6.0: Fixed Defect; 7.0: Reversible Defect |
| `target` | CAD Angiographic Diagnosis | Binary | 0 = Healthy ($<50\%$ stenosis); 1 = CAD Present ($\ge 50\%$ stenosis) |

### 3.2 Descriptive Statistical Characterization
Continuous physiological attributes were analyzed across central tendency, dispersion, and shape parameters to identify underlying distributions:

$$\text{IQR} = Q_3 - Q_1, \quad \text{Skewness} = \frac{n}{(n-1)(n-2)}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3, \quad \text{Kurtosis} = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$

| Biomarker | Mean | Median | Std Dev | IQR | Skewness | Kurtosis | Clinical Threshold |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Age** | 54.44 | 56.00 | 9.04 | 13.00 | -0.209 | -0.523 | Median CAD onset $>55$ yrs |
| **Resting BP (`trestbps`)** | 131.69 | 130.00 | 17.60 | 20.00 | +0.706 | +0.880 | Normal $<120\text{ mmHg}$; Stage-1 $\ge 130$ |
| **Cholesterol (`chol`)** | 246.69 | 241.00 | 51.78 | 64.00 | +1.136 | +4.492 | Desirable $<200\text{ mg/dl}$; High $\ge 240$ |
| **Max Heart Rate (`thalach`)** | 149.61 | 153.00 | 22.88 | 32.50 | -0.537 | -0.054 | Target: $220 - \text{Age}$ |
| **ST Depression (`oldpeak`)** | 1.04 | 0.80 | 1.16 | 1.60 | +1.270 | +1.576 | Normal $<1.0\text{ mm}$; Severe $\ge 2.0$ |

Analysis revealed positive skewness and heavy tails in both serum cholesterol ($\text{Skew} = +1.136, \text{Kurtosis} = +4.492$) and ST depression ($\text{Skew} = +1.270, \text{Kurtosis} = +1.576$), indicating non-Gaussian distributions with extreme right-tail outlier values requiring targeted robust transformation.

### 3.3 Leak-Free Preprocessing Pipeline Architecture
To eliminate data leakage, the data transformation pipeline was engineered with strict separation of training and testing distributions:

1. **Domain-Informed Imputation:** Missing values in vessel fluoroscopy (`ca`, $n=4$) and thallium scintigraphy (`thal`, $n=2$) were imputed using training set distribution median and mode respectively.
2. **Winsorization Soft-Capping:** Continuous variables exhibiting heavy tails (`chol` and `trestbps`) were softly capped at their 99th percentile values derived strictly from the training partition:
   $$\tilde{x}_i = \min(x_i, Q_{0.99})$$
   This approach neutralizes extreme outliers without dropping valid clinical records.
3. **Nominal One-Hot Encoding:** Multi-class nominal variables (`cp`, `restecg`, `slope`, `thal`) were one-hot encoded with $K-1$ dummy indicators, avoiding artificial ordinal assumptions and expanding the feature representation space to 19 orthogonal features.
4. **Stratified Partitioning:** The dataset was partitioned using stratified sampling into an **80% training set ($N_{\text{train}}=242$)** and a **20% holdout test set ($N_{\text{test}}=61$)**, preserving class prevalence ($54.1\%$ Healthy, $45.9\%$ CAD).
5. **Feature Standardization:** Continuous features were standardized to zero mean and unit variance using `StandardScaler`:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   The scaling parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) were fitted strictly on $X_{\text{train}}$ and transformed out-of-sample onto $X_{\text{test}}$.

---

## 4. Machine Learning Algorithmic Formulations

Seven foundational classification algorithms were implemented to evaluate distinct inductive biases:

### 4.1 Logistic Regression (Linear Log-Odds Paradigm)
Models the posterior probability of CAD presence via the logistic sigmoid function operating on an affine combination of input features:
$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T\mathbf{x} + b))}$$
The model parameters are optimized by minimizing the negative log-likelihood with an L2 Tikhonov regularization penalty to prevent overfitting:
$$\mathcal{L}_{\text{LR}}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln \sigma(\mathbf{w}^T\mathbf{x}_i + b) + (1 - y_i)\ln(1 - \sigma(\mathbf{w}^T\mathbf{x}_i + b)) \right] + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$
Optimization is performed via the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) quasi-Newton algorithm.

### 4.2 Gaussian Naive Bayes (Generative Probabilistic Paradigm)
Computes the class posterior probability using Bayes' Rule under the assumption that features are conditionally independent given the class label:
$$P(y=c|\mathbf{x}) = \frac{P(y=c)\prod_{j=1}^d P(x_j|y=c)}{P(\mathbf{x})}$$
The continuous features are modeled via class-conditional Gaussian densities:
$$P(x_j|y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,j}^2}} \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}\right)$$
Maximum Likelihood Estimation (MLE) is employed to estimate class priors $P(y=c)$, means $\mu_{c,j}$, and variances $\sigma_{c,j}^2$.

### 4.3 K-Nearest Neighbors (Instance-Based Non-Parametric Paradigm)
KNN assigns class labels to unseen test query instances $\mathbf{x}^*$ by aggregating the labels of the $k$ nearest training samples according to standardized Euclidean distance:
$$d_E(\mathbf{x}^*, \mathbf{x}_i) = \sqrt{\sum_{j=1}^d (x_j^* - x_{ij})^2}$$
The predicted probability of CAD is computed as the uniform proportion of positive neighbors within the neighborhood $\mathcal{N}_k(\mathbf{x}^*)$:
$$P(y=1|\mathbf{x}^*) = \frac{1}{k}\sum_{i \in \mathcal{N}_k(\mathbf{x}^*)} \mathbb{I}(y_i = 1)$$
Hyperparameter tuning identified $k=7$ as the optimal trade-off between bias (oversmoothing) and variance (sensitivity to local noise).

### 4.4 Decision Tree Classifier (CART - Recursive Partitioning)
Constructs a binary tree by recursively splitting the feature space to maximize the reduction in Gini impurity:
$$I_G(D) = 1 - \sum_{c \in \{0, 1\}} p_c^2$$
At each internal node, the split $(j, \theta)$ across feature $j$ and threshold $\theta$ is selected to maximize the Gini gain:
$$\Delta I_G = I_G(D) - \frac{|D_L|}{|D|}I_G(D_L) - \frac{|D_R|}{|D|}I_G(D_R)$$
To prevent over-specialization to training noise while maintaining human interpretability, tree depth was regularized to $\text{max\_depth}=4$.

### 4.5 Support Vector Machine (Kernel Maximum-Margin Paradigm)
Constructs an optimal separating hyperplane in a transformed reproducing kernel Hilbert space $\Phi(\mathbf{x})$ by solving the soft-margin dual quadratic program:
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \quad \sum_{i=1}^N \alpha_i y_i = 0$$
We employ the non-linear Radial Basis Function (RBF) Gaussian kernel:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$
Platt scaling (fitting a post-hoc logistic sigmoid over decision values $f(\mathbf{x})$) was implemented to produce well-calibrated posterior probabilities.

### 4.6 Random Forest Classifier (Bagging Ensemble - Allocated Model)
An ensemble of $B$ independent, decorrelated decision trees built on bootstrap resamples $\mathcal{D}_b$ of the training set. At each split, a random subset of $m \approx \sqrt{d}$ candidate features is evaluated. Final class probability is derived via ensemble aggregation:
$$\hat{P}(y=1|\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B \hat{P}_b(y=1|\mathbf{x})$$
Exhaustive hyperparameter optimization was conducted via 5-Fold Stratified Cross-Validation (`GridSearchCV`) targeting the ROC-AUC scoring objective:
* `n_estimators`: 150
* `max_depth`: 3
* `min_samples_leaf`: 2
* `min_samples_split`: 2
* `criterion`: `'gini'`

### 4.7 Gradient Boosting Classifier (Forward Stagewise Boosting)
Builds an additive ensemble sequentially by fitting shallow regression trees to the pseudo-residuals (negative gradient of the binomial cross-entropy loss function):
$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F(\mathbf{x}) = F_{m-1}(\mathbf{x})} = y_i - \sigma(F_{m-1}(\mathbf{x}_i))$$
The ensemble updates sequentially with a shrinkage learning rate parameter $\eta = 0.08$ over $M=100$ stages:
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \sum_{j=1}^J \gamma_{jm}\mathbb{I}(\mathbf{x} \in R_{jm})$$

---

## 5. Experimental Results and Comparative Analysis

### 5.1 Performance Evaluation Protocol
Model performance was evaluated on the independent holdout test cohort ($N=61$) across standard diagnostic classification metrics:
* **Accuracy:** $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
* **Precision (Positive Predictive Value):** $\frac{\text{TP}}{\text{TP} + \text{FP}}$
* **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$
* **Specificity (True Negative Rate):** $\frac{\text{TN}}{\text{TN} + \text{FP}}$
* **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
* **ROC-AUC:** Area under the Receiver Operating Characteristic curve.
* **Generalization Stability:** 5-Fold Stratified Cross-Validation AUC mean and standard deviation on the training split.
* **Computational Efficiency:** Mean training time (ms) and inference latency per sample (ms).

### 5.2 Comparative Benchmark Table (Holdout Test Set, $N=61$)

| Model Architecture | Algorithm Paradigm | Accuracy (%) | Precision (%) | Recall (Sensitivity) (%) | Specificity (%) | F1-Score (%) | ROC-AUC | 5-Fold CV AUC (%) | Inference Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **KNN (k=7)** | Distance-Based | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9767** | $86.08 \pm 3.27$ | 1.140 ms |
| ⚡ **SVM (RBF Kernel)** | Kernel Methods | **93.44%** | 90.00% | **96.43%** | 90.91% | **93.10%** | **0.9675** | $87.80 \pm 2.22$ | 0.033 ms |
| 🌲 **Random Forest (Tuned)** | Bagging Ensemble | **93.44%** | **92.86%** | **92.86%** | **93.94%** | **92.86%** | **0.9665** | $89.79 \pm 2.05$ | 0.049 ms |
| 📈 **Logistic Regression** | Linear Models | 85.25% | 80.65% | 89.29% | 81.82% | 84.75% | 0.9567 | $90.93 \pm 1.91$ | 0.016 ms |
| 🚀 **Gradient Boosting** | Boosting Ensemble | 88.52% | 86.21% | 89.29% | 87.88% | 87.72% | 0.9307 | $84.82 \pm 2.22$ | 0.066 ms |
| 🌳 **Decision Tree (CART)** | Tree-Based | 80.33% | 86.36% | 67.86% | 90.91% | 76.00% | 0.8923 | $78.99 \pm 3.34$ | 0.016 ms |
| 🧬 **Naive Bayes (Gaussian)**| Probabilistic | 60.66% | 54.35% | 89.29% | 36.36% | 67.57% | 0.8474 | $86.37 \pm 3.58$ | 0.017 ms |

```
Holdout Test Set Cohort Composition (N=61):
├── Healthy Individuals (Negative Ground Truth, y=0) : 33 patients
└── Confirmed CAD Cases (Positive Ground Truth, y=1) : 28 patients
```

### 5.3 Diagnostic Error Breakdown (Confusion Matrix Analysis)
In acute clinical diagnostics, **Type II errors (False Negatives)** are orders of magnitude more hazardous than **Type I errors (False Positives)**: a false negative results in discharging a patient with critical coronary stenosis who is at immediate risk of acute myocardial infarction.

* **KNN ($k=7$) & SVM (RBF):** Achieved exceptional clinical sensitivity ($\mathbf{96.43\%}$), producing only **1 False Negative** ($\text{TP}=27, \text{FN}=1$) and 3 False Positives ($\text{TN}=30, \text{FP}=3$).
* **Tuned Random Forest:** Produced the most balanced clinical performance profile ($\mathbf{93.44\%}$ Accuracy), generating **2 False Negatives** ($\text{Recall}=92.86\%$) and only **2 False Positives**, securing the highest Specificity ($\mathbf{93.94\%}$) and highest Precision ($\mathbf{92.86\%}$).
* **Single Decision Tree (CART):** Exhibited severe failure modes, yielding **9 False Negatives** ($\text{Recall}=67.86\%$, missing nearly one-third of all diseased patients). Single orthogonal partitions are unable to capture diagonal, compensatory physiological boundaries.
* **Gaussian Naive Bayes:** While capturing 25 true positive cases ($\text{Recall}=89.29\%$), it generated **21 False Positives**, yielding an unacceptably low specificity of $36.36\%$.

### 5.4 Feature Importance Analysis
Using Mean Decrease in Impurity (MDI / Gini Importance) across the 150 trees of the tuned Random Forest, the top physiological predictors were ranked:
1. **`thal_7.0` (Reversible Thallium Perfusion Defect):** $21.4\%$ relative importance. Demonstrates transient exercise-induced ischemia.
2. **`ca` (Major Vessels Colored by Fluoroscopy):** $18.2\%$ relative importance. Direct anatomical marker of multi-vessel calcification.
3. **`cp_4` (Asymptomatic Chest Pain):** $14.8\%$ relative importance. Counterintuitively, asymptomatic presentations in clinical cohorts correlate strongly with advanced, silent CAD.
4. **`oldpeak` (Exercise ST Depression):** $13.1\%$ relative importance. Quantitative ECG marker of subendocardial ischemia.
5. **`thalach` (Max Exercise Heart Rate):** $11.6\%$ relative importance. Chronotropic incompetence serves as a key physiological warning indicator.

---

## 6. Real-Time Clinical Decision Support System Deployment

To bridge the gap between empirical machine learning models and bedside clinical practice, the entire pipeline was integrated into a production-grade **Streamlit Clinical Decision Support System (`app.py`)**.

### 6.1 Interactive Architecture and Sub-Millisecond Latency
The application architecture is structured into four functional modules:
* **Real-Time Diagnostic Ingestion Engine:** Clinicians input patient vitals through clinical sliders with boundary validation against clinical reference intervals.
* **Dynamic Algorithm Switcher:** Empowers clinicians to run inference across any of the seven trained algorithms on the fly, comparing model concordance.
* **Sub-Millisecond Inference:** Per-sample inference latency ranges from **$0.016\text{ ms}$ (Logistic Regression)** to **$1.140\text{ ms}$ (KNN)**, easily satisfying emergency medicine sub-second response benchmarks.
* **Biomarker Radar Visualization:** Renders interactive polar radar charts comparing individual patient metrics against healthy demographic reference baselines.

### 6.2 ACC/AHA Physiological Safety Triggers
In addition to model probabilities, the application runs automated heuristic safety checks based on American College of Cardiology / American Heart Association (ACC/AHA) guidelines:
1. **Systolic Hypertension:** $\ge 140\text{ mmHg}$ (Stage-2) or $\ge 120\text{ mmHg}$ (Pre-hypertensive).
2. **Serum Cholesterol:** $\ge 240\text{ mg/dl}$ (Hypercholesterolemia) or $\ge 200\text{ mg/dl}$ (Borderline).
3. **Chronotropic Incompetence:** Peak heart rate $< 135\text{ bpm}$ under physical exertion.
4. **ST Segment Ischemia:** ST depression $\ge 2.0\text{ mm}$ (Severe) or $\ge 1.0\text{ mm}$ (Mild).

---

## 7. Discussion, Ethical AI & Limitations

### 7.1 Algorithmic Insights across Mathematical Paradigms
The experimental findings provide clear mathematical insights:
* **Distance & Margin Methods Excel on Correlated Clinical Spaces:** Both KNN and SVM (RBF) achieved the highest ROC-AUC ($0.9767$ and $0.9675$). Because physiological variables (blood pressure, cholesterol, max heart rate) form smooth multi-dimensional manifolds, distance and non-linear RBF kernel margins capture diagnostic neighborhoods more effectively than rigid axis-aligned cuts.
* **The Failure of Independence in Naive Bayes:** The poor specificity of Gaussian Naive Bayes ($36.36\%$) directly demonstrates the hazard of the class-conditional independence assumption in biomedicine. Blood pressure and age share significant covariance ($r = 0.28$); assuming independence compounds posterior probability estimation errors, producing excessive false alarms.
* **Ensemble Bagging Eliminates Variance:** Tuned Random Forest elevated single-tree accuracy from $80.33\%$ to $93.44\%$ and reduced False Negatives from 9 to 2, demonstrating the theoretical power of variance reduction through bootstrap aggregation.

### 7.2 Ethical AI Governance and Bias Mitigation
1. **Demographic Parity & Gender Bias:** Historically, CAD clinical trials underrepresented female patients, leading to diagnostic guidelines biased toward typical male angina symptoms. CardioGuard AI evaluates cross-gender sensitivity parity to ensure female patients presenting with atypical chest pain (`cp=2`) are not systematically under-triaged.
2. **HIPAA & GDPR Compliance:** No Protected Health Information (PHI) is persisted. Patient input vectors are processed purely in ephemeral memory for real-time scoring.
3. **Class IIa Software as a Medical Device (SaMD):** CardioGuard AI is intentionally designed as an assistive Clinical Decision Support System. Under Human-in-the-Loop (HITL) principles, the system produces explainable feature attributions to support, rather than supersede, board-certified physician diagnosis.

### 7.3 Study Limitations
* **Sample Size & Single-Center Cohort:** The study was evaluated on the Cleveland cohort ($N=303$). While robust through cross-validation, prospective multi-center validation on contemporary diverse demographic cohorts (e.g., UK Biobank) is necessary prior to hospital-wide deployment.
* **Class Imbalance in Multi-Vessel Sub-Types:** The current formulation binarizes CAD presence ($<50\%$ vs $\ge 50\%$). Future work should expand to ordinal multi-vessel disease staging (1, 2, or 3-vessel disease).

---

## 8. Conclusion and Future Directions

This study presented **CardioGuard AI**, a clinically guided machine learning framework for real-time coronary artery disease risk stratification. By implementing strict data-leakage controls, domain-informed imputation, and Winsorization, we established an authentic benchmark across seven foundational machine learning architectures. 

On an independent holdout test cohort ($N=61$), **K-Nearest Neighbors ($k=7$)**, **RBF Support Vector Machines**, and **Tuned Random Forest** each achieved a top diagnostic accuracy of **93.44%**. KNN and SVM secured the highest clinical safety profile with **96.43% Sensitivity** (missing only 1 diseased patient) and **ROC-AUC scores of 0.9767 and 0.9675**. Deployed into a sub-millisecond Streamlit application with ACC/AHA clinical triggers and ethical governance, this work demonstrates that machine learning can provide rapid, reliable, and interpretable non-invasive risk stratification to augment clinical workflows.

Future work will expand this framework to deep multimodal architectures integrating 12-lead raw ECG waveform time-series with electronic health record (EHR) tabular indicators.

---

## References

1. World Health Organization (WHO), "Cardiovascular diseases (CVDs) Fact Sheet," *WHO Media Centre*, 2021.
2. R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, and V. Froelicher, "International application of a new probability algorithm for the diagnosis of coronary artery disease," *The American Journal of Cardiology*, vol. 64, no. 5, pp. 304–310, 1989.
3. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
4. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.
5. T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.
6. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.
7. J. R. Quinlan, "Induction of Decision Trees," *Machine Learning*, vol. 1, no. 1, pp. 81–106, 1986.
8. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 4765–4774, 2017.
9. American College of Cardiology / American Heart Association (ACC/AHA), "Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults," *Journal of the American College of Cardiology*, vol. 71, no. 19, pp. e127–e248, 2018.
10. F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
