"""
Generate a publication-quality Academic Research Paper PDF for CardioGuard AI
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

DIR = r"C:\Users\ayush\.gemini\antigravity\scratch\ai_practical_exam"
PDF_PATH = os.path.join(DIR, "RESEARCH_PAPER.pdf")
FIG_DIR = os.path.join(DIR, "figures")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (on pages after the first)
        if self._pageNumber > 1:
            self.drawString(54, 750, "CardioGuard AI: Comparative Machine Learning Framework for CAD Risk Stratification")
            self.drawRightString(612 - 54, 750, "Research Paper")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 35, page_text)
        self.drawString(54, 35, "Confidential - For Academic & Clinical Review Only")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=1, # Center
        spaceAfter=10
    )
    
    author_style = ParagraphStyle(
        'PaperAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        alignment=1,
        spaceAfter=15
    )
    
    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
        spaceAfter=4
    )

    abstract_text = ParagraphStyle(
        'AbstractText',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        alignment=4 # Justify
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
        alignment=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    table_cell = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    fig_caption = ParagraphStyle(
        'FigCaption',
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("CardioGuard AI: A Comparative Machine Learning Framework for Real-Time Clinical Decision Support in Coronary Artery Disease Risk Stratification", title_style))
    story.append(Paragraph("<b>Ayush</b><br/>Department of Computer Engineering | Artificial Intelligence & Machine Learning<br/>November 2026", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=10))

    # Abstract Box
    abstract_content = [
        Paragraph("<b>ABSTRACT</b>", abstract_heading),
        Paragraph(
            "Cardiovascular diseases (CVDs), primarily Coronary Artery Disease (CAD), constitute the leading cause of global mortality, "
            "accounting for 17.9 million deaths annually. While invasive coronary angiography remains the gold standard for diagnosis, "
            "its procedural cost, invasiveness, and logistical demands restrict its utility in rapid outpatient screening and emergency triage. "
            "This study presents <b>CardioGuard AI</b>, an end-to-end Machine Learning (ML) framework and interactive Clinical Decision Support "
            "System (CDSS) for non-invasive CAD risk stratification. Utilizing the benchmark UCI Cleveland dataset (303 patients, 14 attributes), "
            "we engineered a leak-free preprocessing pipeline incorporating domain-informed median/mode imputation, 99th-percentile Winsorization soft-capping, "
            "and nominal one-hot encoding. We conducted a systematic empirical comparative study across seven foundational ML classification algorithms: "
            "Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors (KNN), Decision Trees (CART), Support Vector Machines (SVM), "
            "Random Forest, and Gradient Boosting. On an independent unseen test cohort (N=61), <b>KNN (k=7)</b>, <b>SVM (RBF kernel)</b>, and "
            "<b>Tuned Random Forest</b> demonstrated superior diagnostic efficacy, each attaining <b>93.44% accuracy</b>. Crucially for clinical safety, "
            "KNN and SVM achieved <b>96.43% Recall (Sensitivity)</b>—missing only 1 out of 28 cardiac cases—with ROC-AUC scores of <b>0.9767 and 0.9675</b>, "
            "while Tuned Random Forest achieved <b>92.86% Precision</b> and <b>93.94% Specificity</b>. The validated pipeline is deployed in an interactive, "
            "sub-millisecond Streamlit application with ACC/AHA clinical triggers and ethical governance safeguards.",
            abstract_text
        ),
        Spacer(1, 4),
        Paragraph("<b>Keywords:</b> Coronary Artery Disease, Clinical Decision Support System, Random Forest, Support Vector Machines, KNN, ROC-AUC, Medical AI.", abstract_text)
    ]
    
    t_abs = Table([[abstract_content]], colWidths=[504])
    t_abs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(t_abs)
    story.append(Spacer(1, 14))

    # Section 1: Introduction
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Cardiovascular diseases (CVDs) represent the preeminent contributor to mortality worldwide, responsible for 32% of all global deaths. "
        "Coronary Artery Disease (CAD)—characterized by atherosclerotic plaque stenosis in the epicardial coronary arteries leading to myocardial ischemia "
        "or infarction—imposes an immense clinical burden. Conventional clinical scoring systems, such as the Framingham Risk Score, rely on rigid linear "
        "assumptions that frequently fail to capture complex, multi-factorial physiological interactions.", body_style
    ))
    story.append(Paragraph(
        "While invasive coronary angiography remains the definitive diagnostic modality, its invasive catheterization carries procedural complications "
        "and requires specialized cardiac catheterization laboratory infrastructure. Non-invasive machine learning models capable of synthesizing standard "
        "vital signs, exercise electrocardiography (ECG), lipid panels, and fluoroscopy into calibrated disease probabilities can substantially improve "
        "emergency triage and outpatient screening. This paper presents CardioGuard AI, implementing a leak-free 7-algorithm benchmark and deploying the optimal "
        "models inside a production-grade Clinical Decision Support System (CDSS).", body_style
    ))

    # Section 2: Materials & Methods
    story.append(Paragraph("2. Materials and Methods", h1_style))
    story.append(Paragraph("2.1 Clinical Dataset and Cohort Characteristics", h2_style))
    story.append(Paragraph(
        "The empirical investigation utilizes the authentic UCI Cleveland Cardiovascular Disease Database comprising 303 patient records with 14 clinical attributes. "
        "The ground truth binary outcome represents the presence (stenosis &ge; 50%) or absence of significant CAD.", body_style
    ))

    # Table 1: Features
    feat_data = [
        [Paragraph("Feature", table_header), Paragraph("Clinical Description", table_header), Paragraph("Type", table_header), Paragraph("Metric / Clinical Significance", table_header)],
        [Paragraph("age", table_cell_bold), Paragraph("Patient age", table_cell), Paragraph("Continuous", table_cell), Paragraph("Years (Range: 29 - 77)", table_cell)],
        [Paragraph("sex", table_cell_bold), Paragraph("Biological sex", table_cell), Paragraph("Binary", table_cell), Paragraph("1 = Male; 0 = Female", table_cell)],
        [Paragraph("cp", table_cell_bold), Paragraph("Chest pain type", table_cell), Paragraph("Nominal", table_cell), Paragraph("1: Typical, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic", table_cell)],
        [Paragraph("trestbps", table_cell_bold), Paragraph("Resting systolic blood pressure", table_cell), Paragraph("Continuous", table_cell), Paragraph("mmHg on hospital admission (94 - 200)", table_cell)],
        [Paragraph("chol", table_cell_bold), Paragraph("Total serum cholesterol", table_cell), Paragraph("Continuous", table_cell), Paragraph("mg/dl circulating cholesterol (126 - 564)", table_cell)],
        [Paragraph("fbs", table_cell_bold), Paragraph("Fasting blood sugar > 120 mg/dl", table_cell), Paragraph("Binary", table_cell), Paragraph("1 = Diabetic / elevated; 0 = Normal", table_cell)],
        [Paragraph("restecg", table_cell_bold), Paragraph("Resting electrocardiogram", table_cell), Paragraph("Nominal", table_cell), Paragraph("0: Normal, 1: ST-T wave changes, 2: LVH", table_cell)],
        [Paragraph("thalach", table_cell_bold), Paragraph("Max exercise heart rate", table_cell), Paragraph("Continuous", table_cell), Paragraph("Beats per minute achieved (71 - 202)", table_cell)],
        [Paragraph("exang", table_cell_bold), Paragraph("Exercise-induced angina", table_cell), Paragraph("Binary", table_cell), Paragraph("1 = Induced; 0 = None", table_cell)],
        [Paragraph("oldpeak", table_cell_bold), Paragraph("ST depression induced by exercise", table_cell), Paragraph("Continuous", table_cell), Paragraph("mm relative to baseline (0.0 - 6.2 mm)", table_cell)],
        [Paragraph("slope", table_cell_bold), Paragraph("Slope of peak exercise ST segment", table_cell), Paragraph("Nominal", table_cell), Paragraph("1: Upsloping, 2: Flat, 3: Downsloping", table_cell)],
        [Paragraph("ca", table_cell_bold), Paragraph("Fluoroscopy-colored major vessels", table_cell), Paragraph("Discrete", table_cell), Paragraph("Count (0 - 3) of visible coronary vessels", table_cell)],
        [Paragraph("thal", table_cell_bold), Paragraph("Thallium-201 myocardial scintigraphy", table_cell), Paragraph("Nominal", table_cell), Paragraph("3.0: Normal, 6.0: Fixed, 7.0: Reversible", table_cell)],
        [Paragraph("target", table_cell_bold), Paragraph("CAD Angiographic Diagnosis", table_cell), Paragraph("Binary", table_cell), Paragraph("0 = Healthy (<50% stenosis); 1 = CAD (≥50%)", table_cell)]
    ]
    t_feat = Table(feat_data, colWidths=[65, 150, 65, 224])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2.2 Preprocessing and Leak-Free Scaling Architecture", h2_style))
    story.append(Paragraph(
        "To guarantee clinical validity and avoid optimistic performance bias, a strictly leak-free preprocessing pipeline was implemented: "
        "<br/>• <b>Domain Imputation:</b> Missing attributes (ca: n=4, thal: n=2) were imputed using training set distribution median and mode. "
        "<br/>• <b>Winsorization Soft-Capping:</b> Right-tail extreme outliers in serum cholesterol (up to 564 mg/dl) and systolic blood pressure (up to 200 mmHg) were soft-capped at their respective 99th percentiles. "
        "<br/>• <b>Nominal One-Hot Encoding:</b> Multi-class categorical predictors were expanded into dummy indicators, yielding 19 orthogonal features. "
        "<br/>• <b>Stratified Holdout Split:</b> An 80/20 stratified split (Train N=242, Test N=61) was performed. StandardScaler was strictly fitted on X_train and transformed out-of-sample onto X_test.",
        body_style
    ))

    # Section 3: Machine Learning Models
    story.append(Paragraph("3. Machine Learning Classification Models", h1_style))
    story.append(Paragraph(
        "Seven classification algorithms representing distinct mathematical paradigms were implemented and evaluated: "
        "<br/>1. <b>Logistic Regression:</b> Linear log-odds decision boundary with L2 regularization via L-BFGS solver. "
        "<br/>2. <b>Gaussian Naive Bayes:</b> Probabilistic generative classifier assuming class-conditional feature independence. "
        "<br/>3. <b>K-Nearest Neighbors (KNN):</b> Instance-based non-parametric classifier operating with k=7 under Euclidean metric. "
        "<br/>4. <b>Decision Tree Classifier (CART):</b> Recursive binary partitioning optimizing Gini impurity with max_depth=4. "
        "<br/>5. <b>Support Vector Machine (SVM):</b> Non-linear RBF kernel maximizing the geometric margin in reproducing kernel Hilbert space. "
        "<br/>6. <b>Random Forest Classifier:</b> Bagging ensemble of 150 decorrelated trees tuned via 5-Fold Stratified GridSearchCV (max_depth=3, min_samples_leaf=2). "
        "<br/>7. <b>Gradient Boosting Classifier:</b> Sequential boosting ensemble of 100 stages optimizing cross-entropy pseudo-residuals (lr=0.08, max_depth=3).",
        body_style
    ))

    # Section 4: Experimental Results
    story.append(Paragraph("4. Experimental Results and Comparative Analysis", h1_style))
    story.append(Paragraph(
        "All models were evaluated on the previously unseen test partition (N=61: 33 healthy, 28 CAD patients). "
        "In clinical cardiology, <b>Recall (Sensitivity)</b> is prioritized alongside <b>ROC-AUC</b> to minimize fatal False Negatives.", body_style
    ))

    # Table 2: Benchmark
    bm_data = [
        [Paragraph("Model Architecture", table_header), Paragraph("Paradigm", table_header), Paragraph("Accuracy", table_header), Paragraph("Precision", table_header), Paragraph("Recall", table_header), Paragraph("Specificity", table_header), Paragraph("F1-Score", table_header), Paragraph("ROC-AUC", table_header), Paragraph("Latency", table_header)],
        [Paragraph("<b>KNN (k=7)</b>", table_cell), Paragraph("Distance-Based", table_cell), Paragraph("<b>93.44%</b>", table_cell), Paragraph("90.00%", table_cell), Paragraph("<b>96.43%</b>", table_cell), Paragraph("90.91%", table_cell), Paragraph("<b>93.10%</b>", table_cell), Paragraph("<b>0.9767</b>", table_cell), Paragraph("1.140 ms", table_cell)],
        [Paragraph("<b>SVM (RBF Kernel)</b>", table_cell), Paragraph("Kernel Methods", table_cell), Paragraph("<b>93.44%</b>", table_cell), Paragraph("90.00%", table_cell), Paragraph("<b>96.43%</b>", table_cell), Paragraph("90.91%", table_cell), Paragraph("<b>93.10%</b>", table_cell), Paragraph("<b>0.9675</b>", table_cell), Paragraph("0.033 ms", table_cell)],
        [Paragraph("<b>Random Forest (Tuned)</b>", table_cell), Paragraph("Bagging Ensemble", table_cell), Paragraph("<b>93.44%</b>", table_cell), Paragraph("<b>92.86%</b>", table_cell), Paragraph("92.86%", table_cell), Paragraph("<b>93.94%</b>", table_cell), Paragraph("92.86%", table_cell), Paragraph("0.9665", table_cell), Paragraph("0.049 ms", table_cell)],
        [Paragraph("Gradient Boosting", table_cell), Paragraph("Boosting Ensemble", table_cell), Paragraph("88.52%", table_cell), Paragraph("86.21%", table_cell), Paragraph("89.29%", table_cell), Paragraph("87.88%", table_cell), Paragraph("87.72%", table_cell), Paragraph("0.9307", table_cell), Paragraph("0.066 ms", table_cell)],
        [Paragraph("Logistic Regression", table_cell), Paragraph("Linear Models", table_cell), Paragraph("85.25%", table_cell), Paragraph("80.65%", table_cell), Paragraph("89.29%", table_cell), Paragraph("81.82%", table_cell), Paragraph("84.75%", table_cell), Paragraph("0.9567", table_cell), Paragraph("0.016 ms", table_cell)],
        [Paragraph("Decision Tree (CART)", table_cell), Paragraph("Tree-Based", table_cell), Paragraph("80.33%", table_cell), Paragraph("86.36%", table_cell), Paragraph("67.86%", table_cell), Paragraph("90.91%", table_cell), Paragraph("76.00%", table_cell), Paragraph("0.8923", table_cell), Paragraph("0.016 ms", table_cell)],
        [Paragraph("Naive Bayes (Gaussian)", table_cell), Paragraph("Probabilistic", table_cell), Paragraph("60.66%", table_cell), Paragraph("54.35%", table_cell), Paragraph("89.29%", table_cell), Paragraph("36.36%", table_cell), Paragraph("67.57%", table_cell), Paragraph("0.8474", table_cell), Paragraph("0.017 ms", table_cell)],
    ]
    t_bm = Table(bm_data, colWidths=[90, 72, 45, 45, 45, 45, 45, 47, 70])
    t_bm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ecfdf5"), colors.HexColor("#f0fdf4"), colors.HexColor("#f0fdf4"), colors.white, colors.white, colors.white, colors.HexColor("#fef2f2")]),
    ]))
    story.append(t_bm)
    story.append(Spacer(1, 10))

    # Add Figures
    fig_cm = os.path.join(FIG_DIR, "6_confusion_matrices.png")
    fig_roc = os.path.join(FIG_DIR, "7_roc_curves.png")
    if os.path.exists(fig_cm):
        story.append(KeepTogether([
            Image(fig_cm, width=490, height=245),
            Paragraph("<b>Figure 1:</b> Confusion matrices across the 7 Machine Learning classification models (Holdout Test Cohort, N=61).", fig_caption)
        ]))
    
    if os.path.exists(fig_roc):
        story.append(KeepTogether([
            Image(fig_roc, width=380, height=270),
            Paragraph("<b>Figure 2:</b> Multi-model Receiver Operating Characteristic (ROC) comparative curves demonstrating AUC discrimination.", fig_caption)
        ]))

    # Section 5: Clinical Deployment & Discussion
    story.append(Paragraph("5. Real-Time Clinical Decision Support System Deployment", h1_style))
    story.append(Paragraph(
        "The validated models were translated into an interactive, sub-millisecond Streamlit CDSS (<code>app.py</code>). "
        "The application provides real-time risk probability calculation, dynamic model switching, and automated safety alerts based on "
        "ACC/AHA clinical thresholds: Stage-2 Hypertension (&ge;140 mmHg), Hypercholesterolemia (&ge;240 mg/dl), Chronotropic Incompetence "
        "(<135 bpm), and Myocardial Ischemic ST Depression (&ge;2.0 mm). With sub-millisecond inference latency (0.016 ms - 1.14 ms), "
        "the architecture easily supports high-throughput clinical screening.", body_style
    ))

    # Add Feature Importance Figure
    fig_imp = os.path.join(FIG_DIR, "8_feature_importances.png")
    if os.path.exists(fig_imp):
        story.append(KeepTogether([
            Image(fig_imp, width=380, height=250),
            Paragraph("<b>Figure 3:</b> Random Forest Mean Decrease in Impurity (Gini Importance) across the 13 clinical predictors.", fig_caption)
        ]))

    story.append(Paragraph("6. Conclusion", h1_style))
    story.append(Paragraph(
        "Through rigorous leak-free cross-validation and holdout testing on the benchmark UCI Cleveland dataset, this investigation demonstrated "
        "that <b>K-Nearest Neighbors, Support Vector Machines (RBF), and Tuned Random Forest</b> deliver superior diagnostic efficacy (<b>93.44% accuracy</b>). "
        "KNN and SVM minimized fatal False Negatives to just 1 case (<b>96.43% Sensitivity, ROC-AUC 0.9767 and 0.9675</b>), while Random Forest "
        "provided the highest Precision (92.86%) and Specificity (93.94%). Deployed within an enterprise-grade Streamlit CDSS, CardioGuard AI offers "
        "a reliable, scalable, and ethically governed decision-support platform for non-invasive coronary artery disease screening.", body_style
    ))

    # References
    story.append(Paragraph("References", h1_style))
    refs = [
        "1. World Health Organization (WHO), 'Cardiovascular diseases (CVDs) Fact Sheet,' WHO Media Centre, 2021.",
        "2. R. Detrano et al., 'International application of a new probability algorithm for the diagnosis of coronary artery disease,' <i>Am. J. Cardiol.</i>, vol. 64, no. 5, pp. 304–310, 1989.",
        "3. L. Breiman, 'Random Forests,' <i>Machine Learning</i>, vol. 45, no. 1, pp. 5–32, 2001.",
        "4. C. Cortes and V. Vapnik, 'Support-vector networks,' <i>Machine Learning</i>, vol. 20, no. 3, pp. 273–297, 1995.",
        "5. T. Cover and P. Hart, 'Nearest neighbor pattern classification,' <i>IEEE Trans. Inf. Theory</i>, vol. 13, no. 1, pp. 21–27, 1967.",
        "6. J. H. Friedman, 'Greedy function approximation: A gradient boosting machine,' <i>Ann. Statist.</i>, vol. 29, no. 5, pp. 1189–1232, 2001.",
        "7. American College of Cardiology / American Heart Association (ACC/AHA), 'Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure,' <i>J. Am. Coll. Cardiol.</i>, vol. 71, no. 19, pp. e127–e248, 2018.",
        "8. F. Pedregosa et al., 'Scikit-learn: Machine Learning in Python,' <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011."
    ]
    for r in refs:
        story.append(Paragraph(r, bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Research Paper PDF successfully generated at: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
