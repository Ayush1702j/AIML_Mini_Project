"""
Packages all practical exam deliverables into a clean submission zip archive.
"""

import os
import zipfile

OUTPUT_DIR = r"C:\Users\ayush\.gemini\antigravity\scratch\ai_practical_exam"
ZIP_PATH = os.path.join(OUTPUT_DIR, "202401040128_Ayush_AI_Practical_Exam.zip")

files_to_include = [
    "heart_disease_uci.csv",
    "train_and_evaluate.py",
    "heart_disease_practical_exam.ipynb",
    "app.py",
    "best_random_forest_model.joblib",
    "scaler.joblib",
    "feature_names.json",
    "evaluation_summary.json",
    "PROJECT_REPORT.pdf",
    "presentation.pptx",
    "PRESENTATION_SLIDES.md",
    "VIVA_VOCE_PREPARATION.md"
]

fig_dir = os.path.join(OUTPUT_DIR, "figures")

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in files_to_include:
        f_path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(f_path):
            zipf.write(f_path, arcname=f)
            print(f"Added: {f}")
        else:
            print(f"Warning: {f} not found!")

    if os.path.exists(fig_dir):
        for fig_file in os.listdir(fig_dir):
            fig_path = os.path.join(fig_dir, fig_file)
            zipf.write(fig_path, arcname=os.path.join("figures", fig_file))
            print(f"Added figure: figures/{fig_file}")

print("\nSuccessfully created submission zip:", ZIP_PATH)
print("File size:", os.path.getsize(ZIP_PATH), "bytes")
