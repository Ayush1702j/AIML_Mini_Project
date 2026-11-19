"""
CardioGuard AI - Multi-Algorithm Machine Learning Pipeline
Cardiovascular Disease Risk Prediction Benchmark across 7 Core ML Algorithms
Author: Ayush | Course: 2304322L - Artificial Intelligence
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

# The 7 Core Machine Learning Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Publication visual styling
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

print("="*85)
print("SECTION A & B: DATA LOADING, ATTRIBUTES & DESCRIPTIVE STATISTICS")
print("="*85)

data_path = os.path.join(OUTPUT_DIR, "heart_disease_uci.csv")
df = pd.read_csv(data_path)
print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} features")

continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
stat_summary = pd.DataFrame(index=continuous_cols)
stat_summary["Mean"] = df[continuous_cols].mean()
stat_summary["Median"] = df[continuous_cols].median()
stat_summary["Std Dev"] = df[continuous_cols].std()
stat_summary["IQR"] = df[continuous_cols].quantile(0.75) - df[continuous_cols].quantile(0.25)
stat_summary["Skewness"] = df[continuous_cols].skew()
stat_summary["Kurtosis"] = df[continuous_cols].kurtosis()
print(stat_summary.round(3))

print("\n" + "="*85)
print("SECTION C & D: DATA PREPROCESSING & WRANGLING")
print("="*85)

df_clean = df.copy()
df_clean['ca'] = df_clean['ca'].fillna(df_clean['ca'].median())
df_clean['thal'] = df_clean['thal'].fillna(df_clean['thal'].mode()[0])

# Winsorization (99th percentile capping)
chol_cap = df_clean['chol'].quantile(0.99)
trestbps_cap = df_clean['trestbps'].quantile(0.99)
df_clean['chol'] = np.where(df_clean['chol'] > chol_cap, chol_cap, df_clean['chol'])
df_clean['trestbps'] = np.where(df_clean['trestbps'] > trestbps_cap, trestbps_cap, df_clean['trestbps'])

# One-Hot Encoding
categorical_nominal = ['cp', 'restecg', 'slope', 'thal']
df_encoded = pd.get_dummies(df_clean, columns=categorical_nominal, drop_first=True, dtype=int)
print(f"Preprocessed Dataset Shape: {df_encoded.shape}")

print("\n" + "="*85)
print("SECTION E: STRATIFIED DATA PARTITION & SCALING")
print("="*85)

X = df_encoded.drop(columns=['target'])
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
continuous_in_encoded = [c for c in continuous_cols if c in X_train.columns]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_in_encoded] = scaler.fit_transform(X_train[continuous_in_encoded])
X_test_scaled[continuous_in_encoded] = scaler.transform(X_test[continuous_in_encoded])

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
    json.dump(list(X.columns), f, indent=4)

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

print("\n" + "="*85)
print("SECTION F: HYPERPARAMETER TUNING ON RANDOM FOREST")
print("="*85)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 8, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

print("Running GridSearchCV on Random Forest...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=skf,
    scoring='roc_auc',
    n_jobs=1
)
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_
print(f"Optimal Hyperparameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# Save tuned RF
joblib.dump(best_rf, os.path.join(OUTPUT_DIR, "best_random_forest_model.joblib"))

# ==============================================================================
# SECTION G: BENCHMARK OF THE 7 DESIGNATED MACHINE LEARNING ALGORITHMS
# 1. Logistic Regression
# 2. Naive Bayes
# 3. KNN
# 4. Decision Tree
# 5. SVM
# 6. Random Forest
# 7. Gradient Boosting
# ==============================================================================

print("\n" + "="*85)
print("SECTION G: BENCHMARK OF THE 7 CORE MACHINE LEARNING ALGORITHMS")
print("="*85)

algorithms = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "family": "Linear Models"
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "family": "Probabilistic Models"
    },
    "KNN": {
        "model": KNeighborsClassifier(n_neighbors=7),
        "family": "Distance-Based Models"
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=4, random_state=42),
        "family": "Tree-Based Models"
    },
    "SVM": {
        "model": SVC(kernel='rbf', probability=True, random_state=42),
        "family": "Kernel Methods"
    },
    "Random Forest": {
        "model": best_rf,
        "family": "Ensemble (Bagging)"
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
        "family": "Ensemble (Boosting)"
    }
}

print(f"Total Algorithms to Train and Benchmark: {len(algorithms)}")

trained_models_dict = {}
benchmark_results = []
roc_curves = {}

for name, meta in algorithms.items():
    model = meta["model"]
    family = meta["family"]
    
    try:
        # Measure Training Time
        t0 = time.time()
        if name != "Random Forest":
            model.fit(X_train_scaled, y_train)
        fit_time_ms = round((time.time() - t0) * 1000, 2)
        
        # Save model in dictionary
        trained_models_dict[name] = model

        # Measure Inference Time on Test Set
        t_inf0 = time.time()
        preds = model.predict(X_test_scaled)
        inf_time_ms = round(((time.time() - t_inf0) / len(X_test)) * 1000, 4)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_test_scaled)
            probs = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-9)
        else:
            probs = preds.astype(float)

        # Performance Metrics
        acc = accuracy_score(y_test, preds) * 100
        prec = precision_score(y_test, preds, zero_division=0) * 100
        rec = recall_score(y_test, preds, zero_division=0) * 100
        f1 = f1_score(y_test, preds, zero_division=0) * 100
        auc_score = roc_auc_score(y_test, probs)

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0

        # 5-Fold Stratified Cross-Validation on Training Split
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
        cv_mean = cv_scores.mean() * 100
        cv_std = cv_scores.std() * 100

        # Store ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_curves[name] = (fpr, tpr, auc_score)

        benchmark_results.append({
            "Model": name,
            "Family": family,
            "Accuracy (%)": round(acc, 2),
            "Precision (%)": round(prec, 2),
            "Recall (Sensitivity) (%)": round(rec, 2),
            "Specificity (%)": round(spec, 2),
            "F1-Score (%)": round(f1, 2),
            "ROC-AUC": round(auc_score, 4),
            "5-Fold CV AUC Mean (%)": round(cv_mean, 2),
            "5-Fold CV AUC Std (%)": round(cv_std, 2),
            "Train Time (ms)": fit_time_ms,
            "Inference Time/Sample (ms)": inf_time_ms
        })
        print(f"  [✓] {name:<22} | Acc: {acc:.2f}% | Rec: {rec:.2f}% | AUC: {auc_score:.4f}")

    except Exception as e:
        print(f"  [✗] Failed to train {name}: {str(e)}")

benchmark_df = pd.DataFrame(benchmark_results)
benchmark_df = benchmark_df.sort_values(by=["ROC-AUC", "Recall (Sensitivity) (%)"], ascending=[False, False]).reset_index(drop=True)

print("\n" + "="*85)
print("7 ALGORITHMS BENCHMARK SUMMARY TABLE:")
print("="*85)
print(benchmark_df[["Model", "Family", "Accuracy (%)", "Recall (Sensitivity) (%)", "Precision (%)", "F1-Score (%)", "ROC-AUC"]].to_string(index=False))

# Save trained models bundle
joblib.dump(trained_models_dict, os.path.join(OUTPUT_DIR, "all_models.joblib"))
benchmark_df.to_csv(os.path.join(OUTPUT_DIR, "all_algorithms_benchmark.csv"), index=False)
print("\nSaved all models bundle to: all_models.joblib")
print("Saved benchmark results to: all_algorithms_benchmark.csv")

# ==============================================================================
# VISUALIZATIONS FOR THE 7 ALGORITHMS
# ==============================================================================

# Visualization 6: Confusion Matrices for All 7 Algorithms (2x4 Grid)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
cm_colors = ['Blues', 'Greens', 'Purples', 'Oranges', 'Reds', 'YlGnBu', 'PuRd']
model_order = [
    "Logistic Regression", "Naive Bayes", "KNN", "Decision Tree",
    "SVM", "Random Forest", "Gradient Boosting"
]

for idx, (m_name, color) in enumerate(zip(model_order, cm_colors)):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    m_obj = trained_models_dict[m_name]
    preds = m_obj.predict(X_test_scaled)
    cm_val = confusion_matrix(y_test, preds)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap=color, ax=ax, cbar=False,
                xticklabels=['Healthy', 'CAD'], yticklabels=['Healthy', 'CAD'])
    acc_val = accuracy_score(y_test, preds) * 100
    rec_val = recall_score(y_test, preds) * 100
    ax.set_title(f"{m_name}\nAcc: {acc_val:.1f}% | Recall: {rec_val:.1f}%", weight='bold', fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

# 8th subplot: Summary table card
axes[1, 3].axis('off')
summary_text = "🏆 7-Model Benchmark\n" + "-"*26 + "\n"
for _, r in benchmark_df.iterrows():
    summary_text += f"{r['Model'][:15]:<15}: {r['Accuracy (%)']:.1f}% Acc | {r['ROC-AUC']:.3f} AUC\n"
axes[1, 3].text(0.05, 0.5, summary_text, fontsize=9.5, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1'))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "6_confusion_matrices.png"), dpi=300)
plt.close()
print("Saved: 6_confusion_matrices.png")

# Visualization 7: Multi-Model ROC Curves for All 7 Algorithms
plt.figure(figsize=(10, 7.5))
palette = sns.color_palette("tab10", len(model_order))

for (m_name, c) in zip(model_order, palette):
    if m_name in roc_curves:
        fpr, tpr, auc_val = roc_curves[m_name]
        plt.plot(fpr, tpr, color=c, lw=2.2, label=f"{m_name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Baseline (0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Comparison (7 Algorithms)', weight='bold', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "7_roc_curves.png"), dpi=300)
plt.close()
print("Saved: 7_roc_curves.png")

# Visualization 8: Random Forest Gini Feature Importances
importances = best_rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
feat_imp.plot(kind='barh', color='#2980b9', edgecolor='black', alpha=0.85)
plt.title("Random Forest Feature Importances (MDI Gini Index)", weight='bold')
plt.xlabel("Relative Importance Score")
plt.ylabel("Clinical Predictors")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "8_feature_importances.png"), dpi=300)
plt.close()
print("Saved: 8_feature_importances.png")

# Visualization 9: Multi-Metric Model Comparison Bar Chart (All 7 Algorithms)
plot_df = benchmark_df.melt(
    id_vars=["Model"],
    value_vars=["Accuracy (%)", "Recall (Sensitivity) (%)", "Precision (%)", "F1-Score (%)"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(13, 7))
sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="Set2")
plt.title("Comparative Performance Across 7 Core Machine Learning Algorithms", weight='bold', fontsize=14)
plt.ylim(50, 102)
plt.ylabel("Performance Score (%)", fontsize=12)
plt.xticks(rotation=20, ha='right', fontsize=11)
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "9_model_comparison_bar.png"), dpi=300)
plt.close()
print("Saved: 9_model_comparison_bar.png")

# Visualization 10: Algorithm Family Performance Heatmap
plt.figure(figsize=(11, 6))
family_metrics = benchmark_df.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (Sensitivity) (%)", "Specificity (%)", "F1-Score (%)"]]
sns.heatmap(family_metrics, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Metric Score (%)'})
plt.title("Performance Matrix Across 7 Machine Learning Algorithms", weight='bold', fontsize=14, pad=12)
plt.ylabel("Algorithm", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10_algorithm_family_heatmap.png"), dpi=300)
plt.close()
print("Saved: 10_algorithm_family_heatmap.png")

# Save Updated Comprehensive JSON Summary
summary_payload = {
    "student_name": "Ayush",
    "prn": "202401040128",
    "course": "2304322L - Artificial Intelligence",
    "allocated_algorithm": "Random Forest",
    "total_models_evaluated": len(algorithms),
    "best_hyperparameters": grid_search.best_params_,
    "tuned_rf_metrics": {
        "accuracy": benchmark_df[benchmark_df["Model"] == "Random Forest"]["Accuracy (%)"].values[0],
        "precision": benchmark_df[benchmark_df["Model"] == "Random Forest"]["Precision (%)"].values[0],
        "recall": benchmark_df[benchmark_df["Model"] == "Random Forest"]["Recall (Sensitivity) (%)"].values[0],
        "f1": benchmark_df[benchmark_df["Model"] == "Random Forest"]["F1-Score (%)"].values[0],
        "roc_auc": benchmark_df[benchmark_df["Model"] == "Random Forest"]["ROC-AUC"].values[0]
    },
    "comparison_table": benchmark_df.to_dict(orient="records"),
    "available_models": list(trained_models_dict.keys())
}

with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary_payload, f, indent=4)

print("\nAll 7 algorithms trained, benchmarked, and serialized successfully!")
