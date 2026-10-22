"""
build_part_7_8_ppt.py
Generates the official IIT Roorkee PowerPoint presentation for:
Part 7 (Model Training) & Part 8 (Model Evaluation and Tuning)
in the Machine Learning Lifecycle.
Uses IITR_template.pptx as the master layout.
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def create_presentation():
    template_path = "docs/New folder/IITR_template_pptx(19062016)/IITR_template.pptx"
    out_path = "docs/New folder/Part_7_and_8_Model_Training_and_Tuning_PPT.pptx"

    prs = pptx.Presentation(template_path)

    # Palette
    NAVY = RGBColor(0x11, 0x2A, 0x46)
    STEEL_BLUE = RGBColor(0x1F, 0x4E, 0x79)
    DARK_TEXT = RGBColor(0x22, 0x22, 0x22)
    GREEN = RGBColor(0x10, 0x7C, 0x41)
    AMBER = RGBColor(0xD9, 0x77, 0x06)

    # -------------------------------------------------------------
    # SLIDE 0: Title Slide (Modify existing Slide 0)
    # -------------------------------------------------------------
    s0 = prs.slides[0]
    for shp in s0.shapes:
        if shp.name == "Title 1":
            shp.text_frame.word_wrap = True
            p = shp.text_frame.paragraphs[0]
            p.text = "Machine Learning Lifecycle:\nPart 7 & Part 8"
            p.font.name = "Calibri"
            p.font.size = Pt(28)
            p.font.bold = True
            p.font.color.rgb = NAVY
        elif shp.name == "Text Placeholder 2":
            shp.text_frame.word_wrap = True
            p = shp.text_frame.paragraphs[0]
            p.text = "Model Training, Evaluation & Multi-Task Calibration for SmartHVAC Hetero-GNN\n\nPresenter: Animesh Basak (Roll No. 26911001)\nCourse: CSL-505 Machine Learning / Data Mining\nDepartment of Computer Science and Engineering\nIndian Institute of Technology Roorkee"
            p.font.name = "Calibri"
            p.font.size = Pt(13)
            p.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 1: Context in ML Lifecycle (Modify existing Slide 1)
    # -------------------------------------------------------------
    s1 = prs.slides[1]
    for shp in s1.shapes:
        if shp.name == "Title 1":
            shp.text = "Context in the 10-Stage ML Lifecycle"
            shp.text_frame.paragraphs[0].font.color.rgb = NAVY
        elif "Content" in shp.name:
            tf = shp.text_frame
            tf.word_wrap = True
            tf.clear()
            items = [
                ("Assigned Scope: Part 7 (Model Training) and Part 8 (Model Evaluation & Tuning).", True),
                ("Bridges feature graph engineering (Steps 05-06) with enterprise deployment (Steps 09-10).", False),
                ("Pillar 1: Spatio-temporal demand forecasting across 40 Metropolitan and Village hubs.", False),
                ("Pillar 2: Appliance operating power regression and critical failure classification (180 SKUs).", False),
                ("Pillar 3: Sleeper customer churn classification & counter-seasonal link prediction (600 Residents).", False),
                ("Pillar 4: Dynamic spare parts inventory replenishment forecasting (50 Part types).", False),
                ("Target Platform: NVIDIA Tesla V100S GPU with PyTorch Automatic Mixed Precision (FP16 AMP).", True)
            ]
            for text, is_bold in items:
                p = tf.add_paragraph()
                p.text = "• " + text
                p.font.size = Pt(14)
                p.font.bold = is_bold
                p.font.color.rgb = STEEL_BLUE if is_bold else DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 2: Step 07 - GPU Hardware & Architecture (Modify Slide 2)
    # -------------------------------------------------------------
    s2 = prs.slides[2]
    for shp in s2.shapes:
        if shp.name == "Title 1":
            shp.text = "Step 07: High-Performance GPU Architecture"
            shp.text_frame.paragraphs[0].font.color.rgb = NAVY
        elif shp.name == "Text Placeholder 2":
            shp.text = "NVIDIA Tesla V100S Setup"
            shp.text_frame.paragraphs[0].font.color.rgb = STEEL_BLUE
            shp.text_frame.paragraphs[0].font.bold = True
        elif shp.name == "Content Placeholder 3":
            tf = shp.text_frame
            tf.clear()
            for line in [
                "• Dedicated Lab Server: 10.13.1.7 (Ubuntu 24.04).",
                "• Hardware: Tesla V100S-PCIE-32GB (640 Tensor Cores).",
                "• Memory Bandwidth: 1.13 TB/s HBM2 memory.",
                "• Software: PyTorch 2.6.0+cu124 & PyG 2.8.1.",
                "• Low VRAM Footprint: Peak 28.77 MB allocated."
            ]:
                p = tf.add_paragraph()
                p.text = line
                p.font.size = Pt(12)
                p.font.color.rgb = DARK_TEXT
        elif shp.name == "Text Placeholder 4":
            shp.text = "Automatic Mixed Precision (AMP)"
            shp.text_frame.paragraphs[0].font.color.rgb = STEEL_BLUE
            shp.text_frame.paragraphs[0].font.bold = True
        elif shp.name == "Content Placeholder 5":
            tf = shp.text_frame
            tf.clear()
            for line in [
                "• torch.amp.autocast('cuda') for FP16 operations.",
                "• torch.amp.GradScaler prevents gradient underflow.",
                "• Tensor Cores accelerate dense matrix math.",
                "• Master model weights maintained in FP32 precision.",
                "• Training speedup: > 3.8x faster than CPU baseline."
            ]:
                p = tf.add_paragraph()
                p.text = line
                p.font.size = Pt(12)
                p.font.color.rgb = DARK_TEXT

    # Drop existing slide 3 (Blank) and 4 (Last) to append new slides cleanly
    rId_last = prs.slides._sldIdLst[4].rId
    prs.part.drop_rel(rId_last)
    del prs.slides._sldIdLst[4]

    rId_blank = prs.slides._sldIdLst[3].rId
    prs.part.drop_rel(rId_blank)
    del prs.slides._sldIdLst[3]

    layout_content = prs.slide_layouts[1]
    layout_comp = prs.slide_layouts[2]
    layout_last = prs.slide_layouts[4]

    # -------------------------------------------------------------
    # SLIDE 3: Step 07 - Multi-Task GNN Training Formulation
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(layout_content)
    s3.shapes.title.text = "Step 07: Multi-Task Hetero-GNN Formulation"
    s3.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY
    tf3 = s3.placeholders[2].text_frame
    tf3.clear()
    for text, bold in [
        ("Inductive HeteroConv Backbone: 3 message-passing layers with GraphSAGE (SAGEConv) operator.", True),
        ("Dense 8-Dimensional Feature Spaces for all 4 node types (Location, Device, Resident, Part).", False),
        ("Continuous Cyclical Month Encoding: sin(2π(m-1)/12) and cos(2π(m-1)/12) preserve circular time.", True),
        ("Head 1 (Demand): Location-level continuous servicing demand regression (y_demand).", False),
        ("Head 2 (Power & Failure): Operating power kW regression (y_power) & binary fault risk (y_failure).", False),
        ("Head 3 (Sleeper Churn): Resident dormancy classification (y_sleeper) & dot-product cross-selling.", False),
        ("Head 4 (Inventory): Spare parts replenishment forecasting (y_inventory) across warehouses.", False)
    ]:
        p = tf3.add_paragraph()
        p.text = "• " + text
        p.font.size = Pt(13)
        p.font.bold = bold
        p.font.color.rgb = STEEL_BLUE if bold else DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 4: Step 07 - Self-Healing Engine & 120-Epoch Trajectory
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(layout_comp)
    s4.shapes.title.text = "Step 07: Pre-Training Trajectory & Self-Healing"
    s4.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY
    s4.placeholders[1].text = "Self-Healing Divergence Protection"
    s4.placeholders[1].text_frame.paragraphs[0].font.color.rgb = STEEL_BLUE
    s4.placeholders[1].text_frame.paragraphs[0].font.bold = True
    tf4_left = s4.placeholders[2].text_frame
    tf4_left.clear()
    for line in [
        "• Automated loss monitoring: inspects avg_loss each epoch.",
        "• Traps NaN, Inf, or > 10x sudden loss explosion.",
        "• Automatic remedy: Halves LR (1e-3 -> 5e-4) & restores checkpoint.",
        "• Gradient clipping (max_norm = 1.0) prevents runaway backprop.",
        "• Guarantees unattended 120-epoch training reliability."
    ]:
        p = tf4_left.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_TEXT

    s4.placeholders[3].text = "120-Epoch Pre-Training Convergence"
    s4.placeholders[3].text_frame.paragraphs[0].font.color.rgb = STEEL_BLUE
    s4.placeholders[3].text_frame.paragraphs[0].font.bold = True
    tf4_right = s4.placeholders[4].text_frame
    tf4_right.clear()
    for line in [
        "• Epoch 001: Initial Loss: 23,057.12 (Demand: 15,164.87).",
        "• Epoch 010: Rapid descent to 558.25 (Demand: 326.91).",
        "• Epoch 040: Early plateau at 253.39 (Maint: 0.75, Churn: 0.57).",
        "• Epoch 120: Final pre-training loss: 310.93.",
        "• Checkpoint saved: best_hetero_gnn_v100.pt (Best: 235.23)."
    ]:
        p = tf4_right.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 5: Step 08 - Evaluation & The 3 Audit Bottlenecks
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(layout_content)
    s5.shapes.title.text = "Step 08: Model Evaluation & Audit Bottlenecks"
    s5.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY
    tf5 = s5.placeholders[2].text_frame
    tf5.clear()
    for text, bold in [
        ("Rigorous evaluation of pre-trained weights exposed 3 critical bottlenecks:", True),
        ("Bottleneck 1: Multi-Task Gradient Dominance — Demand MSE (~115) and Inventory MSE (~20) produced gradients 200x larger than Churn BCE (~0.50) and Failure BCE (~0.70), starving classification heads.", False),
        ("Bottleneck 2: Under-Calibration in Monsoon & Transition Months — Months 02, 03, 07, 08, 09, 10, and 11 reported 0 high-risk equipment failures, dangerously ignoring moisture and grid surge faults.", False),
        ("Bottleneck 3: False-Positive Churn & Late-Epoch Oscillation — Fixed learning rates caused loss oscillations (253 -> 310), and 518/600 customers were falsely flagged as dormant churners.", False),
        ("Mandate: Implement principled multi-task calibration and real empirical dataset fine-tuning.", True)
    ]:
        p = tf5.add_paragraph()
        p.text = "• " + text
        p.font.size = Pt(13)
        p.font.bold = bold
        p.font.color.rgb = STEEL_BLUE if bold else DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 6: Step 08 - Kendall Uncertainty Weighting & Huber Loss
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(layout_content)
    s6.shapes.title.text = "Step 08: Uncertainty Weighting & Huber Loss"
    s6.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY
    tf6 = s6.placeholders[2].text_frame
    tf6.clear()
    for text, bold in [
        ("Homoscedastic Task Uncertainty Weighting (Kendall et al., CVPR 2018):", True),
        ("L_total = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ], where s_i = log(sigma_i^2) are learnable task log-variances.", False),
        ("Automatically scales down regression tasks by exp(-4.5) ≈ 0.011, matching classification BCE gradients (exp(-0.5) ≈ 0.606).", True),
        ("SmoothL1 (Huber) Regression Loss (beta = 4.0):", True),
        ("Huber(e) = 0.5 * e^2 for |e| < beta, and beta * |e| - 0.5 * beta^2 otherwise. Prevents demand outliers from dominating backprop.", False),
        ("Class-Balanced BCE Loss: pos_weight = 2.5 for rare equipment faults and 1.8 for sleeper churn.", False),
        ("Optimization: AdamW (lr = 2.5e-4) + CosineAnnealingLR (T_max = 60, eta_min = 1e-5) + Gradient Clipping (1.0).", True)
    ]:
        p = tf6.add_paragraph()
        p.text = "• " + text
        p.font.size = Pt(13)
        p.font.bold = bold
        p.font.color.rgb = STEEL_BLUE if bold else DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 7: Step 08 - Real Benchmark Datasets Ingested
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(layout_content)
    s7.shapes.title.text = "Step 08: Real Benchmark Dataset Ingestion"
    s7.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY
    tf7 = s7.placeholders[2].text_frame
    tf7.clear()
    for text, bold in [
        ("Dataset 4: UCI AI4I 2020 Predictive Maintenance (dataset/601):", True),
        ("10,000 industrial machine records. Ingested Heat Dissipation Failure (HDF) & Power Failure (PWF) modes (2.07% fault rate). Mapped torque * RPM to mechanical power kW (1.2 - 3.6 kW).", False),
        ("Dataset 1: UCI Online Retail II (dataset/502):", True),
        ("525,461 transactions across 4,312 customers. Extracted Recency, Frequency, Monetary (RFM). Recency > 90d sets ground-truth sleeper churn (33.28% baseline). Log-monetary maps to income tier.", False),
        ("Dataset 3: Kaggle HVAC Power Consumption (felixpradoh):", True),
        ("8,760 hourly thermal telemetry readings across 12 climate regimes. Ingested ambient temp, indoor temp, and power kW to calibrate seasonal thermodynamic load curves.", False),
        ("Hardware Execution: fine_tune_v100.py completed in 17.0s on Tesla V100 GPU (60 epochs).", True)
    ]:
        p = tf7.add_paragraph()
        p.text = "• " + text
        p.font.size = Pt(13)
        p.font.bold = bold
        p.font.color.rgb = STEEL_BLUE if bold else DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 8: Step 08 - Quantitative Before vs. After Benchmark
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(layout_comp)
    s8.shapes.title.text = "Step 08: Before vs. After Calibration Ledger"
    s8.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY
    s8.placeholders[1].text = "Pre-Fine-Tune Baseline"
    s8.placeholders[1].text_frame.paragraphs[0].font.color.rgb = AMBER
    s8.placeholders[1].text_frame.paragraphs[0].font.bold = True
    tf8_left = s8.placeholders[2].text_frame
    tf8_left.clear()
    for line in [
        "• Overall Dossier Loss: 137.04",
        "• Pillar 1 Demand MSE: 115.61 (MAE: 8.77)",
        "• Pillar 2 Power MSE: 0.8364 (MAE: 0.749)",
        "• Pillar 2 Failure F1: 0.540 (154 flagged)",
        "• Pillar 3 Churn F1: 0.397 (518 flagged)",
        "• Pillar 4 Inventory MSE: 19.13 (MAE: 3.45)",
        "• Annual Revenue: ₹2.40 Cr | Net Profit: ₹2.10 Cr"
    ]:
        p = tf8_left.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_TEXT

    s8.placeholders[3].text = "Post-Fine-Tune Calibrated"
    s8.placeholders[3].text_frame.paragraphs[0].font.color.rgb = GREEN
    s8.placeholders[3].text_frame.paragraphs[0].font.bold = True
    tf8_right = s8.placeholders[4].text_frame
    tf8_right.clear()
    for line in [
        "• Overall Dossier Loss: 46.80 (-65.85% Error Reduction)",
        "• Pillar 1 Demand MSE: 33.13 (-71.34% Error Reduction)",
        "• Pillar 2 Power MSE: 0.0654 (-92.18% Error Reduction)",
        "• Pillar 2 Failure F1: 0.697 (+0.157 gain, 432 units)",
        "• Pillar 3 Churn F1: 0.646 (+0.249 gain, 278 flag / 276 rct)",
        "• Pillar 4 Inventory MSE: 12.61 (-34.11% Error Reduction)",
        "• Annual Revenue: ₹2.61 Cr | Net Profit: ₹2.27 Cr"
    ]:
        p = tf8_right.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = GREEN

    # -------------------------------------------------------------
    # SLIDE 9: Step 08 - 12-Month Audited Operational Ledger & P&L
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(layout_content)
    s9.shapes.title.text = "Step 08: 12-Month Audited Financial Ledger"
    s9.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY
    tf9 = s9.placeholders[2].text_frame
    tf9.clear()
    for text, bold in [
        ("100% Monthly Profitability: All 12 calendar snapshots demonstrate positive Net Operating Profit.", True),
        ("Gross Operating Revenue: ₹2,61,01,851.84 (~₹2.61 Cr, exact row-sum match, +8.65% over baseline).", False),
        ("Operating Expenditure (OPEX): ₹34,00,492.40 (~₹34.00 L, 100% exact match across all 12 rows).", False),
        ("Net Operating Profit: ₹2,27,01,359.44 (~₹2.27 Cr, +₹17.45 L / +8.33% margin gain over baseline).", True),
        ("Cost-Efficiency Ratio: 7.764x (unweighted mean) / 7.676x (aggregate annual ratio).", False),
        ("Proactive Savings Gain: +22.64% avg (Peak: +28.4% in winter & summer).", False),
        ("High-Risk Units Flagged: Exactly 432 units across all months (zero monsoon blind spots).", True)
    ]:
        p = tf9.add_paragraph()
        p.text = "• " + text
        p.font.size = Pt(13)
        p.font.bold = bold
        p.font.color.rgb = STEEL_BLUE if bold else DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 10: Summary of Contributions (Part 7 & 8)
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(layout_content)
    s10.shapes.title.text = "Summary of Contributions (Part 7 & 8)"
    s10.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY
    tf10 = s10.placeholders[2].text_frame
    tf10.clear()
    for text, bold in [
        ("Hardware Pipeline: Orchestrated V100 GPU FP16 AMP training with self-healing divergence guards.", True),
        ("Multi-Task Loss Balancing: Solved gradient dominance via Kendall homoscedastic uncertainty weighting.", False),
        ("Real Data Integration: Grounded GNN in 3 canonical benchmarks (UCI AI4I 2020, UCI 502, Kaggle Power).", False),
        ("Failure Detection: Lifted F1 score from 0.540 to 0.697 and flagged 432 critical units (monsoon calibrated).", True),
        ("Customer Churn: Lifted F1 score from 0.397 to 0.646, eliminating 240 false-positive churn flags.", False),
        ("Audited Business Impact: Delivered ₹2.27 Cr net profit with audited zero arithmetic roll-up variance.", True),
        ("Deliverables Generated: fine_tuned_hetero_gnn_v100.pt, fine_tune_telemetry.json, ARGOS.md, ARGOS_BROWSER_RESULTS.html.", False)
    ]:
        p = tf10.add_paragraph()
        p.text = "• " + text
        p.font.size = Pt(13)
        p.font.bold = bold
        p.font.color.rgb = STEEL_BLUE if bold else DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 11: Thank You Slide (Add Last Slide Layout)
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(layout_last)
    s11.shapes.title.text = "Thank You!\nQuestions & Discussion"
    s11.shapes.title.text_frame.paragraphs[0].font.color.rgb = NAVY

    prs.save(out_path)
    print(f"Successfully generated PowerPoint presentation: {out_path} ({len(prs.slides)} slides)")

if __name__ == "__main__":
    create_presentation()
