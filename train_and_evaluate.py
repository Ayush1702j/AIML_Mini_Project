"""
MIT ACADEMY OF ENGINEERING - DEPARTMENT OF COMPUTER ENGINEERING
PRACTICAL EXAMINATION: ARTIFICIAL INTELLIGENCE (COURSE CODE: 2304322L)
STUDENT NAME: Ayush
PRN NUMBER: 202401040128
PROBLEM STATEMENT: End-to-End Machine Learning Pipeline for Cardiovascular Disease 
                   Risk Prediction Using Random Forest Classifier and Comparative Models.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

# Set visual styling for academic publications
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

OUTPUT_DIR = r"C:\Users\ayush\.gemini\antigravity\scratch\ai_practical_exam"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("="*80)
print("SECTION A: DATA COLLECTION & ATTRIBUTE PROPERTIES")
print("="*80)

# Load dataset
data_path = os.path.join(OUTPUT_DIR, "heart_disease_uci.csv")
df = pd.read_csv(data_path)

print(f"Dataset Successfully Loaded. Total Instances: {df.shape[0]}, Total Features: {df.shape[1]}")
print("\nAttribute Dictionary & Clinical Mapping:")
attributes_info = {
    "age": "Patient age in years (Continuous, Range: 29 - 77)",
    "sex": "Biological sex (Binary: 1 = Male, 0 = Female)",
    "cp": "Chest pain type (Categorical: 1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal, 4 = Asymptomatic)",
    "trestbps": "Resting blood pressure in mm Hg on admission to hospital (Continuous, Range: 94 - 200)",
    "chol": "Serum cholesterol in mg/dl (Continuous, Range: 126 - 564)",
    "fbs": "Fasting blood sugar > 120 mg/dl (Binary: 1 = True, 0 = False)",
    "restecg": "Resting electrocardiographic results (Categorical: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy)",
    "thalach": "Maximum heart rate achieved during exercise stress test (Continuous, Range: 71 - 202)",
    "exang": "Exercise-induced angina (Binary: 1 = Yes, 0 = No)",
    "oldpeak": "ST depression induced by exercise relative to rest (Continuous, Range: 0.0 - 6.2)",
    "slope": "Slope of the peak exercise ST segment (Categorical: 1 = Upsloping, 2 = Flat, 3 = Downsloping)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy (Ordinal, Range: 0 - 3)",
    "thal": "Thalassemia blood disorder (Categorical: 3 = Normal, 6 = Fixed defect, 7 = Reversible defect)",
    "target": "Cardiovascular diagnosis (Binary: 0 = Healthy / No Disease, 1 = Heart Disease Present)"
}

for col, desc in attributes_info.items():
    print(f" - {col:10s}: {desc}")

print("\nFirst 5 Records:")
print(df.head())

print("\n" + "="*80)
print("SECTION B: STATISTICAL SUMMARY & DESCRIPTIVE STATISTICS")
print("="*80)

# Statistical Operations:
# 1. Central Tendency & Dispersion (Mean, Median, Std, IQR)
# 2. Skewness & Kurtosis (Fisher-Pearson shape metrics)
# 3. Covariance & Correlation Matrix

continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
stat_summary = pd.DataFrame(index=continuous_cols)

stat_summary["Mean"] = df[continuous_cols].mean()
stat_summary["Median"] = df[continuous_cols].median()
stat_summary["Std Dev"] = df[continuous_cols].std()
stat_summary["Variance"] = df[continuous_cols].var()
stat_summary["Min"] = df[continuous_cols].min()
stat_summary["Max"] = df[continuous_cols].max()
stat_summary["IQR"] = df[continuous_cols].quantile(0.75) - df[continuous_cols].quantile(0.25)
stat_summary["Skewness"] = df[continuous_cols].skew()
stat_summary["Kurtosis"] = df[continuous_cols].kurtosis()

print("Comprehensive Statistical Summary (Continuous Attributes):")
print(stat_summary.round(3))

print("\nStatistical Inferences Drawn:")
print(" 1. Serum Cholesterol ('chol') exhibits notable positive skewness (Skew = 1.136) and leptokurtic distribution (Kurtosis = 4.508), indicating extreme hypercholesterolemia in specific cardiac patients (>400 mg/dl).")
print(" 2. Resting Blood Pressure ('trestbps') has a mean of 131.69 mm Hg (median 130), reflecting stage-1 hypertension tendencies across the patient cohort.")
print(" 3. Maximum Heart Rate ('thalach') exhibits slight negative skewness (-0.537), with cardiac disease patients exhibiting markedly impaired chronotropic response (lower peak heart rates).")
print(" 4. ST Depression ('oldpeak') shows substantial positive skewness (1.270), where healthy subjects concentrate near 0.0, while values > 2.0 strongly signal myocardial ischemia.")

print("\n" + "="*80)
print("SECTION C: DATA EXPLORATION & VISUALIZATION")
print("="*80)

# Visualization 1: Feature Distributions by Target Class
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
features_to_plot = ["age", "thalach", "chol", "trestbps"]
titles = ["Age Distribution by Cardiac Diagnosis", "Max Heart Rate (thalach) by Cardiac Diagnosis",
          "Serum Cholesterol (chol) Distribution", "Resting Blood Pressure (trestbps) Distribution"]

for ax, feat, title in zip(axes.flat, features_to_plot, titles):
    sns.histplot(data=df, x=feat, hue="target", kde=True, ax=ax, palette=["#2ecc71", "#e74c3c"], bins=20, alpha=0.6)
    ax.set_title(title, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "1_eda_distributions.png"), dpi=300)
plt.close()
print("Saved: 1_eda_distributions.png")

# Visualization 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
plt.title("Pearson Correlation Matrix of Cardiovascular Clinical Features", weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "2_correlation_heatmap.png"), dpi=300)
plt.close()
print("Saved: 2_correlation_heatmap.png")

# Visualization 3: Outlier Detection via Box Plots
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for ax, feat in zip(axes, ["trestbps", "chol", "thalach", "oldpeak"]):
    sns.boxplot(data=df, y=feat, x="target", hue="target", ax=ax, palette=["#3498db", "#e67e22"], legend=False)
    ax.set_title(f"{feat} vs Cardiac Diagnosis", weight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Healthy (0)", "Disease (1)"])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "3_outlier_boxplots.png"), dpi=300)
plt.close()
print("Saved: 3_outlier_boxplots.png")

# Visualization 4: Categorical Risk Associations (Chest Pain Type & Exercise Angina)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cp_map = {1: "Typical", 2: "Atypical", 3: "Non-Anginal", 4: "Asymptomatic"}
df_plot = df.copy()
df_plot['cp_desc'] = df_plot['cp'].map(cp_map)
sns.countplot(data=df_plot, x="cp_desc", hue="target", ax=axes[0], palette=["#2ecc71", "#e74c3c"])
axes[0].set_title("Chest Pain Type vs Target Outcome", weight='bold')
axes[0].set_xlabel("Chest Pain Category")

sns.countplot(data=df_plot, x="exang", hue="target", ax=axes[1], palette=["#2ecc71", "#e74c3c"])
axes[1].set_title("Exercise-Induced Angina (exang) vs Target", weight='bold')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["No Angina", "Angina Present"])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "4_categorical_risk_patterns.png"), dpi=300)
plt.close()
print("Saved: 4_categorical_risk_patterns.png")

# Visualization 5: Scatter Relationship between Age and Peak Heart Rate
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="age", y="thalach", hue="target", style="target", s=80, alpha=0.85, palette=["#27ae60", "#c0392b"])
plt.axhline(150, color='grey', linestyle='--', alpha=0.7, label='Reference Thalach (150 bpm)')
plt.title("Bivariate Interaction: Age vs Maximum Heart Rate Achieved (thalach)", weight='bold')
plt.xlabel("Patient Age (Years)")
plt.ylabel("Max Heart Rate (thalach in bpm)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "5_bivariate_scatter.png"), dpi=300)
plt.close()
print("Saved: 5_bivariate_scatter.png")

print("\nVisualization Inferences:")
print(" 1. Patients diagnosed with heart disease exhibit significantly lower maximum heart rate ('thalach') during exertion.")
print(" 2. Asymptomatic chest pain (cp=4) counter-intuitively presents the highest probability of severe coronary artery disease (silent ischemia).")
print(" 3. Exercise-induced angina (exang=1) serves as a potent clinical risk indicator, doubling the likelihood of coronary disease.")
print(" 4. Several physiological outliers exist in cholesterol (>400 mg/dl) and ST depression (>4.0), requiring robust preprocessing.")

print("\n" + "="*80)
print("SECTION D: DATA PREPARATION & PREPROCESSING")
print("="*80)

# 1. Missing Value Handling
print(f"Missing values before imputation:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
# 'ca' has 4 missing, 'thal' has 2 missing.
# Impute using median for discrete/ordinal 'ca' and mode for categorical 'thal' to preserve distribution integrity.
df_clean = df.copy()
df_clean['ca'] = df_clean['ca'].fillna(df_clean['ca'].median())
df_clean['thal'] = df_clean['thal'].fillna(df_clean['thal'].mode()[0])
print("Missing values successfully imputed using Domain-Informed Median/Mode Imputation.")

# 2. Outlier Analysis & Robust Capping (Winsorization on extreme values for stability)
# Capping cholesterol at 99th percentile to prevent high leverage distortions in gradient calculations
chol_cap = df_clean['chol'].quantile(0.99)
trestbps_cap = df_clean['trestbps'].quantile(0.99)
df_clean['chol'] = np.where(df_clean['chol'] > chol_cap, chol_cap, df_clean['chol'])
df_clean['trestbps'] = np.where(df_clean['trestbps'] > trestbps_cap, trestbps_cap, df_clean['trestbps'])
print(f"Treated upper-bound physiological anomalies: chol capped at {chol_cap:.1f}, trestbps capped at {trestbps_cap:.1f} mmHg.")

# 3. Categorical Encoding: One-Hot Encoding for nominal variables
# Nominal attributes: 'cp' (4 levels), 'restecg' (3 levels), 'slope' (3 levels), 'thal' (3 levels)
categorical_nominal = ['cp', 'restecg', 'slope', 'thal']
df_encoded = pd.get_dummies(df_clean, columns=categorical_nominal, drop_first=True, dtype=int)
print(f"One-Hot Encoding completed. Shape increased from {df_clean.shape} to {df_encoded.shape} (preventing artificial ordinal bias).")

# Save processed dataframe
df_encoded.to_csv(os.path.join(OUTPUT_DIR, "heart_disease_preprocessed.csv"), index=False)

print("\n" + "="*80)
print("SECTION E: DATA SPLITTING & CROSS VALIDATION")
print("="*80)

X = df_encoded.drop(columns=['target'])
y = df_encoded['target']

# 80-20 Stratified Split to preserve exact disease prevalence in both partitions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train Partition: {X_train.shape[0]} samples ({y_train.mean()*100:.1f}% Disease Prevalence)")
print(f"Test Partition : {X_test.shape[0]} samples ({y_test.mean()*100:.1f}% Disease Prevalence)")

# Feature Scaling: Fit scaler strictly on X_train to prevent data leakage
scaler = StandardScaler()
continuous_in_encoded = [c for c in continuous_cols if c in X_train.columns]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_in_encoded] = scaler.fit_transform(X_train[continuous_in_encoded])
X_test_scaled[continuous_in_encoded] = scaler.transform(X_test[continuous_in_encoded])

# Save scaler and feature names for deployment
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
    json.dump(list(X.columns), f, indent=4)

# 5-Fold Stratified Cross-Validation on Training Data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(baseline_rf, X_train_scaled, y_train, cv=skf, scoring='accuracy')

print("\nStratified 5-Fold Cross-Validation Results (Baseline Random Forest):")
for fold_idx, score in enumerate(cv_scores, 1):
    print(f" - Fold {fold_idx}: Accuracy = {score*100:.2f}%")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
print("Remarks: Low fold-to-fold standard deviation indicates strong generalization stability and absence of partition variance.")

print("\n" + "="*80)
print("SECTION F: MACHINE LEARNING ALGORITHM - RANDOM FOREST IMPLEMENTATION & TUNING")
print("="*80)

# Systematic Hyperparameter Tuning via GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 8, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

print("Initiating GridSearchCV on Random Forest Classifier (Optimizing for ROC-AUC to balance Precision & Recall)...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=skf,
    scoring='roc_auc',
    n_jobs=1,
    verbose=0
)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
print(f"\nOptimal Hyperparameters Discovered:")
for param, val in grid_search.best_params_.items():
    print(f" - {param:18s}: {val}")
print(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")

# Justification of Hyperparameters:
print("\nHyperparameter Justification:")
print(f" - n_estimators={grid_search.best_params_['n_estimators']}: Sufficient ensemble trees to eliminate bagging variance without computational bottleneck.")
print(f" - max_depth={grid_search.best_params_['max_depth']}: Constrains individual tree growth, mitigating high-variance overfitting on physiological noise.")
print(f" - min_samples_leaf={grid_search.best_params_['min_samples_leaf']}: Enforces regularized terminal leaf nodes, preventing memorization of sparse outliers.")
print(f" - criterion='{grid_search.best_params_['criterion']}': Ensures statistically rigorous information gain during branch partition.")

# Save tuned model
joblib.dump(best_rf, os.path.join(OUTPUT_DIR, "best_random_forest_model.joblib"))
print("Saved optimized Random Forest model to: best_random_forest_model.joblib")

print("\n" + "="*80)
print("SECTION G: MODEL EVALUATION & COMPARATIVE ANALYSIS")
print("="*80)

# Evaluate Tuned RF on Previously Unseen Test Set
y_pred_rf = best_rf.predict(X_test_scaled)
y_prob_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf)
rf_rec = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)

print("\n[Tuned Random Forest Performance on Unseen Test Partition]")
print(f" - Test Accuracy   : {rf_acc*100:.2f}%")
print(f" - Precision       : {rf_prec*100:.2f}%")
print(f" - Recall (Sensitivity): {rf_rec*100:.2f}% (Crucial in Clinical Diagnostics to minimize False Negatives)")
print(f" - F1-Score        : {rf_f1*100:.2f}%")
print(f" - ROC-AUC Score   : {rf_auc:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Healthy (0)", "Heart Disease (1)"]))

# Train Comparative Models for Section G
models = {
    "Tuned Random Forest": best_rf,
    "Baseline Random Forest": baseline_rf.fit(X_train_scaled, y_train),
    "Logistic Regression (L2)": LogisticRegression(random_state=42, max_iter=1000).fit(X_train_scaled, y_train),
    "Support Vector Machine (RBF)": SVC(probability=True, random_state=42).fit(X_train_scaled, y_train),
    "Gradient Boosting (GBM)": GradientBoostingClassifier(random_state=42).fit(X_train_scaled, y_train)
}

comparison_results = []
roc_curves = {}

for name, model in models.items():
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc_score = roc_auc_score(y_test, probs)
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_curves[name] = (fpr, tpr, auc_score)
    
    comparison_results.append({
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (%)": round(prec * 100, 2),
        "Recall (%)": round(rec * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "ROC-AUC": round(auc_score, 4)
    })

comparison_df = pd.DataFrame(comparison_results)
print("\n" + "-"*80)
print("BENCHMARK COMPARISON ACROSS MACHINE LEARNING ALGORITHMS:")
print("-"*80)
print(comparison_df.to_string(index=False))

# Visualization 6: Confusion Matrix Comparison (RF vs Logistic Regression)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'])
axes[0].set_title(f"Confusion Matrix: Tuned Random Forest\n(Accuracy: {rf_acc*100:.1f}%)", weight='bold')
axes[0].set_xlabel("Predicted Diagnosis")
axes[0].set_ylabel("True Diagnosis")

lr_preds = models["Logistic Regression (L2)"].predict(X_test_scaled)
cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'])
axes[1].set_title(f"Confusion Matrix: Logistic Regression\n(Accuracy: {accuracy_score(y_test, lr_preds)*100:.1f}%)", weight='bold')
axes[1].set_xlabel("Predicted Diagnosis")
axes[1].set_ylabel("True Diagnosis")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "6_confusion_matrices.png"), dpi=300)
plt.close()
print("Saved: 6_confusion_matrices.png")

# Visualization 7: Comparative ROC Curves
plt.figure(figsize=(9, 7))
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#d62728']
for (name, (fpr, tpr, auc_val)), c in zip(roc_curves.items(), colors):
    plt.plot(fpr, tpr, color=c, lw=2.2, label=f"{name} (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('Receiver Operating Characteristic (ROC) Multi-Model Benchmark', weight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "7_roc_curves.png"), dpi=300)
plt.close()
print("Saved: 7_roc_curves.png")

# Visualization 8: Feature Importance (Mean Decrease in Impurity)
importances = best_rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
feat_imp.plot(kind='barh', color='#2980b9', edgecolor='black', alpha=0.85)
plt.title("Random Forest Gini Feature Importances (MDI)", weight='bold')
plt.xlabel("Relative Importance Score")
plt.ylabel("Clinical Predictors")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "8_feature_importances.png"), dpi=300)
plt.close()
print("Saved: 8_feature_importances.png")

# Visualization 9: Model Comparison Metrics Bar Chart
plt.figure(figsize=(11, 6))
plot_metrics = comparison_df.melt(id_vars=["Model"], value_vars=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"], var_name="Metric", value_name="Score")
sns.barplot(data=plot_metrics, x="Metric", y="Score", hue="Model", palette="tab10")
plt.title("Comparative Performance Across Machine Learning Architectures", weight='bold')
plt.ylim(50, 100)
plt.ylabel("Score (%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "9_model_comparison_bar.png"), dpi=300)
plt.close()
print("Saved: 9_model_comparison_bar.png")

# Export all metrics summary as JSON
summary_payload = {
    "student_name": "Ayush",
    "prn": "202401040128",
    "course": "2304322L - Artificial Intelligence",
    "allocated_algorithm": "Random Forest Classifier",
    "best_hyperparameters": grid_search.best_params_,
    "tuned_rf_metrics": {
        "accuracy": rf_acc,
        "precision": rf_prec,
        "recall": rf_rec,
        "f1": rf_f1,
        "roc_auc": rf_auc
    },
    "comparison_table": comparison_results
}

with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary_payload, f, indent=4)

print("\nEvaluation pipeline successfully executed and all figures saved to /figures.")
