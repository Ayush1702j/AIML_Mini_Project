# MIT ACADEMY OF ENGINEERING
## Department of Computer Engineering | Third Year B.Tech (Sem-V) 2025–2026
### Course Code: 2304322L — Artificial Intelligence Practical Examination
**Candidate Name:** Ayush | **PRN Number:** 202401040128  
**Topic:** End-to-End Machine Learning Pipeline for Cardiovascular Disease Risk Prediction  
**Allocated Algorithm:** Random Forest Classifier  
**Subject:** Section I — Presentation, Oral and Question & Answer Session [4 Marks, CO4 L5]

---

# Comprehensive Viva-Voce & Oral Exam Defense Guide

This guide provides technical answers to the most probable questions external and internal examiners ask during the practical examination.

---

### Category 1: Algorithm Fundamentals & Random Forest (CO2 L4)

#### Q1: What is a Random Forest Classifier, and how does it work?
**Answer:**  
Random Forest is an ensemble machine learning algorithm that combines multiple decision trees using **Bootstrap Aggregating (Bagging)** and **Random Subspace Feature Selection**. During training:
1. $N$ bootstrap samples (subsets sampled with replacement) are drawn from the training data.
2. An individual decision tree is grown on each bootstrap sample.
3. At each node split, only a random subset of features (typically $\sqrt{p}$ or $\log_2(p)$) is evaluated to find the best split, rather than considering all $p$ features.
4. During inference, each tree casts a vote, and the majority class is selected (or average probability computed).

#### Q2: Why does Random Forest introduce random feature selection at each node split?
**Answer:**  
If all features are evaluated at every split, dominant clinical features (such as `thalach` or `cp`) will be chosen at the root of almost every tree, leading to **highly correlated trees**. The variance of an average of $N$ correlated variables is:
$$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{N}\sigma^2$$
As $N \to \infty$, the variance cannot decrease below $\rho \sigma^2$. By forcing trees to split only on a random subset of features, Random Forest **decorrelates the trees** ($\rho \to 0$), allowing ensemble variance to drop dramatically.

#### Q3: What is the difference between Bagging and Boosting?
**Answer:**
- **Bagging (e.g., Random Forest):** Trees are grown independently in parallel on bootstrap samples. Its primary goal is to **reduce model variance** without increasing bias.
- **Boosting (e.g., Gradient Boosting / XGBoost):** Trees are grown sequentially in series. Each subsequent tree specifically learns to predict the pseudo-residuals (errors) of the preceding ensemble. Its primary goal is to **reduce model bias**.

#### Q4: What is the difference between Gini Impurity and Entropy?
**Answer:**
- **Gini Impurity:** Measures the probability of misclassifying a randomly chosen element from the set if it were randomly labeled according to the distribution of labels:
  $$\text{Gini} = 1 - \sum_{i=1}^C p_i^2$$
- **Entropy (Information Gain):** Measures the average amount of information or uncertainty in the distribution:
  $$\text{Entropy} = -\sum_{i=1}^C p_i \log_2(p_i)$$
In our GridSearchCV tuning, Gini impurity yielded equivalent accuracy with lower computational overhead because it avoids logarithmic evaluations.

#### Q5: How is Out-of-Bag (OOB) error calculated in Random Forest?
**Answer:**  
Because each tree is trained on a bootstrap sample drawn with replacement, approximately $(1 - \frac{1}{N})^N \approx e^{-1} \approx 36.8\%$ of training samples are left out of each tree's training set. These unselected samples are called **Out-of-Bag (OOB) samples**. The OOB error is computed by passing each sample through only those trees for which it was left out, providing an unbiased validation estimate without needing a separate validation split.

---

### Category 2: Data Preprocessing & Validation (CO1 L3, CO2 L3)

#### Q6: How did you prevent Data Leakage during preprocessing?
**Answer:**  
Data leakage occurs when information from outside the training partition is used to fit the model. We strictly prevented data leakage by:
1. Performing the **Stratified Train-Test split BEFORE any feature scaling or transformation**.
2. Fitting the `StandardScaler` **exclusively on `X_train`** (`scaler.fit_transform(X_train)`).
3. Applying the pre-fitted scaler to `X_test` using **strictly `.transform()`**, ensuring test distribution statistics never leaked into training.
4. Deriving imputation medians and Winsorization capping thresholds solely from the training partition.

#### Q7: Why did you use Stratified K-Fold Cross Validation instead of standard K-Fold?
**Answer:**  
Standard K-Fold randomly assigns instances to folds, which risks uneven class distributions across folds—especially in moderate-sized datasets ($N=303$). **Stratified K-Fold** enforces that every fold preserves the exact class prevalence (45.9% heart disease, 54.1% healthy) matching the overall population, eliminating fold-to-fold class imbalance bias.

#### Q8: Why did you choose Winsorization instead of dropping outliers?
**Answer:**  
In clinical datasets, extreme physiological measurements (such as cholesterol of 564 mg/dl or blood pressure of 200 mmHg) are not data-entry errors; they represent real, critically ill patients. Dropping them would discard the most informative clinical cases. **Winsorization (capping at the 99th percentile)** preserves the high-risk patient records while bounding extreme numerical leverage that could distort gradient calculations or margin boundaries.

#### Q9: Why did you use One-Hot Encoding instead of Label Encoding for Chest Pain Type (`cp`)?
**Answer:**  
`cp` consists of 4 categories: (1) Typical Angina, (2) Atypical Angina, (3) Non-Anginal, and (4) Asymptomatic. If we used Label Encoding, the model would falsely assume a numerical ordering ($4 > 3 > 2 > 1$), implying that Asymptomatic is four times greater than Typical Angina. **One-Hot Encoding with `drop_first=True`** creates binary indicator columns without imposing an artificial mathematical hierarchy or multicollinearity.

---

### Category 3: Model Evaluation & Performance Analysis (CO3 L5)

#### Q10: In cardiovascular risk diagnosis, which metric is more critical: Precision or Recall?
**Answer:**  
**Recall (Sensitivity) is substantially more critical.**
- A **False Negative (FN)** means predicting that a patient with occult coronary artery disease is healthy. That patient is discharged without treatment and risks fatal myocardial infarction (heart attack).
- A **False Positive (FP)** means a healthy patient is flagged as high-risk, resulting only in follow-up non-invasive tests (e.g., stress ECG, echocardiogram).
Because the clinical cost of an FN is catastrophic compared to an FP, we optimized for high Recall. Our tuned Random Forest achieved **92.86% Recall**, misclassifying only 2 diseased patients out of 28.

#### Q11: What is the ROC curve and what does the ROC-AUC score represent?
**Answer:**  
The **Receiver Operating Characteristic (ROC)** curve plots the **True Positive Rate (Sensitivity / Recall)** on the y-axis against the **False Positive Rate ($1 - \text{Specificity}$)** on the x-axis across all possible decision thresholds ($0.0 \le \tau \le 1.0$).  
The **Area Under the ROC Curve (ROC-AUC)** measures the probability that the classifier ranks a randomly chosen positive patient higher than a randomly chosen negative patient. A score of 0.5 represents random guessing, while 1.0 is a perfect classifier. Our tuned model achieved an exceptional **0.9665 ROC-AUC**.

#### Q12: Why did Hyperparameter Tuning boost accuracy from 86.89% to 93.44%?
**Answer:**  
The baseline Random Forest grew unconstrained trees (`max_depth=None`), allowing individual trees to expand until leaf nodes were 100% pure. In a dataset with 303 rows, deep trees memorize idiosyncratic noise and physiological outliers. By constraining `max_depth = 3` and enforcing `min_samples_leaf = 2`, we regularized tree capacity, forcing trees to split only on broadly generalizable pathophysiology, eliminating overfitting on the test partition.

#### Q13: How does Random Forest compare with Logistic Regression and Support Vector Machines?
**Answer:**
- **Logistic Regression (Accuracy: 85.25%):** Assumes a linear decision boundary between log-odds and predictors. It failed to capture non-linear interactions (e.g., how age non-linearly modulates maximum exercise heart rate).
- **SVM with RBF Kernel (Accuracy: 93.44%, Recall: 96.43%):** Highly effective in transformed kernel space, but functions as a complete mathematical black-box without native feature importance.
- **Random Forest (Accuracy: 93.44%, Recall: 92.86%):** Matches SVM in accuracy, naturally handles mixed categorical/continuous data, is immune to monotonic transformations, and provides intrinsic Gini feature importance for clinical interpretability.

---

### Category 4: Production Deployment & Systems Engineering (CO4 L5)

#### Q14: How does your Streamlit deployment demonstrate production readiness?
**Answer:**  
Our Streamlit app (`app.py`) demonstrates:
1. **Interactive Real-Time Inference:** Doctors input 13 clinical features; the app executes preprocessing, applies pre-fitted scaling, and outputs risk probability within milliseconds.
2. **Clinical Risk Stratification:** Rather than binary 0/1 output, it maps probabilities into clinical tiers: Low Risk (<30%), Moderate Risk (30-60%), and High Critical Risk (>60%).
3. **Local Explainability:** Ranks top feature drivers for each specific patient assessment so clinicians understand why a risk score was assigned.
4. **Architectural Blueprints:** Explicit tabs detailing scalability, observability, and AI ethics.

#### Q15: How would you deploy this pipeline to handle high hospital request volume?
**Answer:**
1. **Containerization:** Package the pipeline into a multi-stage Docker container (`python:3.12-slim`) with pinned dependencies and a sub-250MB image footprint.
2. **Microservice Decoupling:** Decouple front-end intake forms from inference using **FastAPI** over REST/gRPC, served via **Uvicorn** workers.
3. **Horizontal Pod Autoscaling (HPA):** Deploy on Kubernetes (AWS EKS / GCP GKE) with an HPA rule configured to scale from 2 to 10 pods when CPU utilization exceeds 70%.
4. **Asynchronous Batch Inference:** For nightly batch screening of 50,000+ hospitalized EHR records, route requests through **Celery / Redis** message queues to isolate batch loads from interactive emergency room queries.

#### Q16: What is Data Drift, and how do you detect and monitor it in production?
**Answer:**  
**Data Drift (Covariate Shift)** occurs when the statistical distribution of incoming patient data $P(X)$ changes over time (e.g., older demographic admitted during winter or new sensor calibration) even if $P(Y|X)$ remains constant.  
We monitor drift in production using **Evidently AI** or **Great Expectations** by computing:
- **Population Stability Index (PSI):**
  $$\text{PSI} = \sum \left( \% \text{Actual} - \% \text{Expected} \right) \times \ln\left( \frac{\% \text{Actual}}{\% \text{Expected}} \right)$$
  - $\text{PSI} < 0.1$: No significant drift.
  - $0.1 \le \text{PSI} \le 0.2$: Moderate shift; trigger inspection.
  - $\text{PSI} > 0.2$: Significant drift; trigger automated retraining alert via **MLflow**.

#### Q17: What ethical safeguards did you incorporate for Clinical AI?
**Answer:**
1. **Demographic Parity & Bias Mitigation:** Heart disease symptoms in women are historically under-recognized. We audit false negative rates across gender to ensure equitable diagnostic sensitivity.
2. **Explainable AI (XAI):** Black-box predictions are dangerous in healthcare. Gini importance and SHAP values ensure clinicians understand the physiological justification for every prediction.
3. **HIPAA & GDPR Compliance:** Zero Protected Health Information (PHI) such as patient names or government IDs are passed to the inference engine; data is pseudonymized.
4. **Human-in-the-Loop (HITL):** The system is strictly a **Clinical Decision Support System (CDSS)** to augment physician judgment, never an autonomous authority that prescribes treatment without doctor sign-off.
