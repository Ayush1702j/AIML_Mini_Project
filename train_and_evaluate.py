"""
CardioGuard AI - Comprehensive Multi-Algorithm Machine Learning Pipeline
Cardiovascular Disease Risk Prediction Benchmark across All Major ML Paradigms
Author: Ayush | PRN: 202401040128 | Course: 2304322L - Artificial Intelligence
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

# 1. Linear & Discriminant Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# 2. Bayesian & Distance/Geometric Models
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 3. Tree-based Models & Ensembles
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)

# 4. Advanced Boosting Algorithms
from xgboost import XGBClassifier

# 5. Neural Network
from sklearn.neural_network import MLPClassifier

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
print("SECTION F: HYPERPARAMETER TUNING ON RANDOM FOREST (ALLOCATED ALGORITHM)")
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

print("\n" + "="*85)
print("SECTION G: COMPREHENSIVE MULTI-ALGORITHM BENCHMARK ACROSS ALL MAJOR ML PARADIGMS")
print("="*85)

# Instantiate all major ML classifiers across all 5 paradigms
all_algorithms = {
    # 1. Linear & Discriminant Family
    "Logistic Regression (L2 Regularized)": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "family": "Linear & Discriminant"
    },
    "Logistic Regression (L1 Lasso)": {
        "model": LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        "family": "Linear & Discriminant"
    },
    "Ridge Classifier": {
        "model": CalibratedClassifierCV(RidgeClassifier(random_state=42), cv=3),
        "family": "Linear & Discriminant"
    },
    "Linear Discriminant Analysis (LDA)": {
        "model": LinearDiscriminantAnalysis(),
        "family": "Linear & Discriminant"
    },
    "SGD Classifier (Log Loss)": {
        "model": CalibratedClassifierCV(SGDClassifier(loss='log_loss', random_state=42), cv=3),
        "family": "Linear & Discriminant"
    },

    # 2. Bayesian & Distance / Geometric Family
    "Gaussian Naive Bayes": {
        "model": GaussianNB(),
        "family": "Bayesian & Distance"
    },
    "Bernoulli Naive Bayes": {
        "model": BernoulliNB(),
        "family": "Bayesian & Distance"
    },
    "K-Nearest Neighbors (KNN, k=7)": {
        "model": KNeighborsClassifier(n_neighbors=7),
        "family": "Bayesian & Distance"
    },
    "Support Vector Machine (Linear Kernel)": {
        "model": SVC(kernel='linear', probability=True, random_state=42),
        "family": "Support Vector Machines"
    },
    "Support Vector Machine (RBF Kernel)": {
        "model": SVC(kernel='rbf', probability=True, random_state=42),
        "family": "Support Vector Machines"
    },
    "Support Vector Machine (Polynomial Kernel)": {
        "model": SVC(kernel='poly', degree=3, probability=True, random_state=42),
        "family": "Support Vector Machines"
    },

    # 3. Decision Trees & Classical Tree Ensembles
    "Decision Tree Classifier (CART)": {
        "model": DecisionTreeClassifier(max_depth=5, random_state=42),
        "family": "Decision Trees & Ensembles"
    },
    "Baseline Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "family": "Decision Trees & Ensembles"
    },
    "Tuned Random Forest (Primary)": {
        "model": best_rf,
        "family": "Decision Trees & Ensembles"
    },
    "Extra Trees Classifier": {
        "model": ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=42),
        "family": "Decision Trees & Ensembles"
    },
    "AdaBoost Classifier": {
        "model": AdaBoostClassifier(n_estimators=100, algorithm='SAMME', random_state=42),
        "family": "Decision Trees & Ensembles"
    },
    "Gradient Boosting (GBM)": {
        "model": GradientBoostingClassifier(random_state=42),
        "family": "Gradient Boosting"
    },
    "Hist Gradient Boosting (Histogram-based)": {
        "model": HistGradientBoostingClassifier(random_state=42),
        "family": "Gradient Boosting"
    },

    # 4. State-of-the-Art Gradient Boosters
    "XGBoost Classifier": {
        "model": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=42),
        "family": "Gradient Boosting"
    },

    # 5. Neural Network
    "Multi-Layer Perceptron (Neural Net / MLP)": {
        "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
        "family": "Neural Networks"
    }
}

# Add Meta-Ensembles (Voting & Stacking)
base_estimators = [
    ('rf', best_rf),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=42)),
    ('lr', LogisticRegression(random_state=42, max_iter=1000))
]

all_algorithms["Soft Voting Ensemble"] = {
    "model": VotingClassifier(estimators=base_estimators, voting='soft'),
    "family": "Meta-Ensembles"
}

all_algorithms["Stacking Classifier (Meta-Learner)"] = {
    "model": StackingClassifier(
        estimators=[
            ('rf', best_rf),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=7)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=42))
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=3
    ),
    "family": "Meta-Ensembles"
}

print(f"Total Algorithms to Train and Benchmark: {len(all_algorithms)}")

trained_models_dict = {}
benchmark_results = []
roc_curves = {}

for name, meta in all_algorithms.items():
    model = meta["model"]
    family = meta["family"]
    
    try:
        # Measure Training Time
        t0 = time.time()
        if name != "Tuned Random Forest (Primary)":
            model.fit(X_train_scaled, y_train)
        fit_time_ms = round((time.time() - t0) * 1000, 2)
        
        # Save model in dictionary
        trained_models_dict[name] = model

        # Measure Inference Time on Test Set
        t_inf0 = time.time()
        y_pred = model.predict(X_test_scaled)
        inf_time_ms = round(((time.time() - t_inf0) / len(X_test_scaled)) * 1000, 3)

        # Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_test_scaled)
            y_prob = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min())
        else:
            y_prob = y_pred

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_prob)

        # Specificity from Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc_score)

        benchmark_results.append({
            "Model": name,
            "Family": family,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (Sensitivity) (%)": round(rec * 100, 2),
            "Specificity (%)": round(spec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2),
            "ROC-AUC": round(auc_score, 4),
            "False Negatives (FN)": int(fn),
            "Fit Time (ms)": fit_time_ms,
            "Inference Latency (ms)": inf_time_ms
        })
        print(f" ✓ Successfully evaluated: {name:40s} | Acc: {acc*100:.2f}% | Recall: {rec*100:.2f}% | AUC: {auc_score:.4f}")
    except Exception as e:
        print(f" ✕ Error evaluating {name}: {e}")

# Convert to DataFrame and sort by Accuracy and ROC-AUC
benchmark_df = pd.DataFrame(benchmark_results).sort_values(by=["Accuracy (%)", "ROC-AUC"], ascending=False)
benchmark_df.reset_index(drop=True, inplace=True)

print("\n" + "="*115)
print("EXHAUSTIVE MULTI-ALGORITHM BENCHMARK SUMMARY (N=61 Unseen Test Patients):")
print("="*115)
print(benchmark_df.to_string(index=False))

# Save all trained models bundle for dynamic Streamlit serving
joblib.dump(trained_models_dict, os.path.join(OUTPUT_DIR, "all_models.joblib"))
benchmark_df.to_csv(os.path.join(OUTPUT_DIR, "all_algorithms_benchmark.csv"), index=False)
print("\nSaved all models bundle to: all_models.joblib")
print("Saved benchmark results to: all_algorithms_benchmark.csv")

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

# Visualization 6: Confusion Matrices (Top 3 Model Architectures)
top_3_models = [
    ("Tuned Random Forest", best_rf),
    ("XGBoost Classifier", trained_models_dict["XGBoost Classifier"]),
    ("Logistic Regression (L2)", trained_models_dict["Logistic Regression (L2 Regularized)"])
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
colors_cm = ['Blues', 'Purples', 'Greens']

for ax, (m_name, m_obj), colormap in zip(axes, top_3_models, colors_cm):
    preds = m_obj.predict(X_test_scaled)
    cm_val = confusion_matrix(y_test, preds)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap=colormap, ax=ax, cbar=False,
                xticklabels=['Healthy', 'CAD'], yticklabels=['Healthy', 'CAD'])
    acc_val = accuracy_score(y_test, preds) * 100
    rec_val = recall_score(y_test, preds) * 100
    ax.set_title(f"{m_name}\nAcc: {acc_val:.1f}% | Recall: {rec_val:.1f}%", weight='bold', fontsize=11)
    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("True Ground Truth")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "6_confusion_matrices.png"), dpi=300)
plt.close()
print("Saved: 6_confusion_matrices.png")

# Visualization 7: Multi-Model ROC Curves (Top 8 Diverse Models)
highlight_models = [
    "Tuned Random Forest (Primary)",
    "XGBoost Classifier",
    "Support Vector Machine (RBF Kernel)",
    "Soft Voting Ensemble",
    "Stacking Classifier (Meta-Learner)",
    "Multi-Layer Perceptron (Neural Net / MLP)",
    "Extra Trees Classifier",
    "Logistic Regression (L2 Regularized)"
]

plt.figure(figsize=(10, 7.5))
palette = sns.color_palette("tab10", len(highlight_models))

for (m_name, c) in zip(highlight_models, palette):
    if m_name in roc_curves:
        fpr, tpr, auc_val = roc_curves[m_name]
        plt.plot(fpr, tpr, color=c, lw=2.2, label=f"{m_name} (AUC = {auc_val:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance Baseline (0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Multi-Algorithm Receiver Operating Characteristic (ROC) Benchmark', weight='bold', fontsize=14)
plt.legend(loc="lower right", fontsize=9.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "7_roc_curves.png"), dpi=300)
plt.close()
print("Saved: 7_roc_curves.png")

# Visualization 8: Random Forest Gini Feature Importances
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

# Visualization 9: Multi-Metric Model Comparison Bar Chart (Top 10 Models)
top_10 = benchmark_df.head(10).copy()
plot_df = top_10.melt(
    id_vars=["Model"],
    value_vars=["Accuracy (%)", "Recall (Sensitivity) (%)", "Precision (%)", "F1-Score (%)"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(13, 7))
sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="Set2")
plt.title("Comparative Performance of Top 10 Machine Learning Architectures", weight='bold', fontsize=14)
plt.ylim(65, 102)
plt.ylabel("Performance Score (%)", fontsize=12)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "9_model_comparison_bar.png"), dpi=300)
plt.close()
print("Saved: 9_model_comparison_bar.png")

# Visualization 10: Algorithm Family Performance Heatmap
plt.figure(figsize=(12, 10))
family_metrics = benchmark_df.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (Sensitivity) (%)", "Specificity (%)", "F1-Score (%)"]]
sns.heatmap(family_metrics, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Metric Score (%)'})
plt.title("Comprehensive Algorithm Performance Matrix (All Evaluated Models)", weight='bold', fontsize=14, pad=12)
plt.ylabel("Algorithm Architecture", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10_algorithm_family_heatmap.png"), dpi=300)
plt.close()
print("Saved: 10_algorithm_family_heatmap.png")

# Save Updated Comprehensive JSON Summary
summary_payload = {
    "student_name": "Ayush",
    "prn": "202401040128",
    "course": "2304322L - Artificial Intelligence",
    "allocated_algorithm": "Random Forest Classifier",
    "total_models_evaluated": len(all_algorithms),
    "best_hyperparameters": grid_search.best_params_,
    "tuned_rf_metrics": {
        "accuracy": benchmark_df[benchmark_df["Model"] == "Tuned Random Forest (Primary)"]["Accuracy (%)"].values[0],
        "precision": benchmark_df[benchmark_df["Model"] == "Tuned Random Forest (Primary)"]["Precision (%)"].values[0],
        "recall": benchmark_df[benchmark_df["Model"] == "Tuned Random Forest (Primary)"]["Recall (Sensitivity) (%)"].values[0],
        "f1": benchmark_df[benchmark_df["Model"] == "Tuned Random Forest (Primary)"]["F1-Score (%)"].values[0],
        "roc_auc": benchmark_df[benchmark_df["Model"] == "Tuned Random Forest (Primary)"]["ROC-AUC"].values[0]
    },
    "comparison_table": benchmark_df.to_dict(orient="records"),
    "available_models": list(trained_models_dict.keys())
}

with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary_payload, f, indent=4)

print("\nAll algorithms trained, benchmarked, and serialized successfully!")
