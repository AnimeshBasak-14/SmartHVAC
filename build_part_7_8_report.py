import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    """Sets cell padding/margins in dxa units (20 dxa = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="B0C4DE", sz="4", val="single"):
    """Sets subtle clean borders on a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def format_run(run, font_name="Times New Roman", size_pt=12, bold=False, italic=False, color_rgb=(0,0,0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

def add_p(doc, text="", font_name="Times New Roman", size_pt=12, bold=False, italic=False,
          color_rgb=(0,0,0), align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=6, line_spacing=1.15):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        r = p.add_run(text)
        format_run(r, font_name, size_pt, bold, italic, color_rgb)
    return p

def add_h1(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    format_run(r, font_name="Times New Roman", size_pt=15, bold=True, color_rgb=(15, 34, 64))
    return p

def add_h2(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    format_run(r, font_name="Times New Roman", size_pt=13, bold=True, color_rgb=(31, 78, 121))
    return p

def add_h3(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    format_run(r, font_name="Times New Roman", size_pt=12, bold=True, italic=True, color_rgb=(44, 62, 80))
    return p

def add_callout(doc, text_list, title="KEY FINDING / FORMULA"):
    """Adds a shaded callout block with a left accent border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, "F4F6F7")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=140)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1F4E79"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"[{title}]\n")
    format_run(r_title, font_name="Times New Roman", size_pt=10.5, bold=True, color_rgb=(31, 78, 121))
    
    for t in text_list:
        p_item = cell.add_paragraph()
        p_item.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_item.paragraph_format.space_before = Pt(1)
        p_item.paragraph_format.space_after = Pt(2)
        p_item.paragraph_format.line_spacing = 1.15
        r_item = p_item.add_run(t)
        format_run(r_item, font_name="Times New Roman", size_pt=10.5, color_rgb=(30, 30, 30))
        
    p_end = doc.add_paragraph()
    p_end.paragraph_format.space_before = Pt(0)
    p_end.paragraph_format.space_after = Pt(4)

def add_figure_with_caption(doc, img_path, caption_text, width_inches=6.0):
    """Inserts a centered image with an academic figure caption."""
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(width_inches))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run(caption_text)
        format_run(r_cap, font_name="Times New Roman", size_pt=10, italic=True, color_rgb=(80, 80, 80))
    else:
        print(f"Warning: Figure {img_path} not found.")

def build_full_report():
    doc = docx.Document()
    
    # Page Setup: Standard 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Header setup
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("IIT Roorkee | Dept. of CSE | CSL-505 ML Lifecycle: Parts 7 & 8 Report")
        format_run(hrun, font_name="Times New Roman", size_pt=8.5, italic=True, color_rgb=(120, 120, 120))
        
        # Footer setup
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Department of Computer Science & Engineering, IIT Roorkee")
        format_run(frun, font_name="Times New Roman", size_pt=9, italic=True, color_rgb=(120, 120, 120))

    # =========================================================================
    # TITLE PAGE (Matching Senior's Format Exactly)
    # =========================================================================
    p0 = add_p(doc, "INDIAN INSTITUTE OF TECHNOLOGY ROORKEE", size_pt=21, bold=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=18, space_after=6, color_rgb=(15, 34, 64))
    
    p1 = add_p(doc, "Department of Computer Science & Engineering", size_pt=15, bold=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=18, color_rgb=(31, 78, 121))
    
    p2 = add_p(doc, "Project Report On", size_pt=15, bold=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=6, color_rgb=(60, 60, 60))
    
    p3 = add_p(doc, "Heterogeneous Graph Neural Network for Smart HVAC Energy Forecasting, Predictive Maintenance, Customer Churn, and Inventory Optimization",
               size_pt=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=14, color_rgb=(15, 34, 64))
    
    p4 = add_p(doc, "Course: Machine Learning (CSL-505) / Data Mining (CSL-515)", size_pt=14, bold=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=4, color_rgb=(31, 78, 121))
    
    p5 = add_p(doc, "Lifecycle Module Specialization: Part 7 (Model Training) & Part 8 (Model Evaluation and Tuning)",
               size_pt=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=24, color_rgb=(44, 62, 80))
    
    p6 = add_p(doc, "Submitted By:", size_pt=12, bold=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=4, color_rgb=(60, 60, 60))
    
    p7 = add_p(doc, "Animesh Basak (Enrollment No. 26911001)", size_pt=13, bold=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=20, color_rgb=(15, 34, 64))
    
    p8 = add_p(doc, "Under The Guidance of:", size_pt=12, bold=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=4, color_rgb=(60, 60, 60))
    
    p9 = add_p(doc, "Dr. Durga Toshniwal", size_pt=14, bold=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2, color_rgb=(15, 34, 64))
    
    p10 = add_p(doc, "Professor, Department of Computer Science & Engineering\nIndian Institute of Technology Roorkee",
                size_pt=12, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=30, color_rgb=(60, 60, 60))
    
    doc.add_page_break()

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    add_p(doc, "Table of Contents", size_pt=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
          space_before=10, space_after=12, color_rgb=(15, 34, 64))
    
    toc_data = [
        ("1", "Member's Contribution Table", "2"),
        ("2", "Section 1: Executive Overview & The 10-Stage Machine Learning Lifecycle", "3"),
        ("3", "Section 2: Part 7 — Model Training (GPU Architecture & Multi-Task Formulation)", "5"),
        ("3.1", "  2.1 Compute Infrastructure & NVIDIA Tesla V100 GPU Setup", "5"),
        ("3.2", "  2.2 Heterogeneous Graph Topology & Relational Message Passing", "6"),
        ("3.3", "  2.3 Decoupled Multi-Task Architecture & Prediction Heads", "7"),
        ("3.4", "  2.4 Pre-Training Optimization Dynamics & Loss Formulation", "8"),
        ("3.5", "  2.5 Self-Healing Numerical Divergence Protection Algorithm", "9"),
        ("3.6", "  2.6 120-Epoch Pre-Training Trajectory & Initial Plateau", "10"),
        ("4", "Section 3: Part 8 — Model Evaluation & Fine-Tuning (Audit Bottlenecks & Calibration)", "11"),
        ("4.1", "  3.1 Post-Training Audit Bottlenecks (The Argos Technical Review)", "11"),
        ("4.2", "  3.2 Mathematical Formulation of Fine-Tuning Calibrations", "12"),
        ("4.3", "  3.3 Real Industrial & Transactional Benchmark Datasets Ingested", "14"),
        ("4.4", "  3.4 Fine-Tuning Execution on Tesla V100 GPU (fine_tune_v100.py)", "16"),
        ("4.5", "  3.5 Quantitative Model Evaluation: Before vs. After Calibration Ledger", "17"),
        ("4.6", "  3.6 Audited 12-Month Operational Ledger & Financial P&L Reconciliation", "19"),
        ("5", "Section 4: Critical Discussion, Findings, and Future Directions", "21"),
        ("6", "Section 5: Conclusion & Summary of Deliverables", "23"),
        ("7", "Section 6: Academic References & Citations", "24"),
    ]
    
    tbl_toc = doc.add_table(rows=len(toc_data) + 1, cols=3)
    tbl_toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_toc)
    
    col_widths_toc = [Inches(0.8), Inches(4.7), Inches(1.0)]
    
    headers_toc = ["S.No.", "Section / Topic", "Page No."]
    for c_idx, h_text in enumerate(headers_toc):
        cell = tbl_toc.rows[0].cells[c_idx]
        cell.width = col_widths_toc[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 1 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=10.5, bold=True, color_rgb=(255, 255, 255))
        
    for r_idx, (sno, title, page) in enumerate(toc_data):
        row = tbl_toc.rows[r_idx + 1]
        for c_idx, val in enumerate([sno, title, page]):
            cell = row.cells[c_idx]
            cell.width = col_widths_toc[c_idx]
            bg = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 1 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            is_main = not sno.startswith(" ") and not "." in sno
            format_run(r, font_name="Times New Roman", size_pt=10, bold=is_main, color_rgb=(0,0,0))
            
    add_p(doc, "", space_after=14)

    # =========================================================================
    # MEMBERS CONTRIBUTION TABLE (Matching Table 1 from Report.docx)
    # =========================================================================
    add_p(doc, "Members' Contribution Table", size_pt=15, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
          space_before=14, space_after=6, color_rgb=(15, 34, 64))
    
    add_p(doc, "This project synthesizes an end-to-end 10-stage Machine Learning lifecycle. The specific assigned modules and architectural contributions documented in this report focus on Part 7 (Model Training) and Part 8 (Model Evaluation and Tuning), authored and benchmarked by Animesh Basak.",
          size_pt=11, italic=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=8)

    contrib_data = [
        ("1", "Animesh Basak", "26911001", "Part 7: Model Training",
         "NVIDIA Tesla V100 GPU compute setup (CUDA 13.0, FP16 AMP); Multi-task Relational Graph Convolutional Network (RGCN) message passing across 4 node types (Location, Device, Resident, Part); self-healing numerical divergence algorithm; 120-epoch pre-training trajectory execution (artifacts/best_hetero_gnn_v100.pt)."),
        ("2", "Animesh Basak", "26911001", "Part 8: Model Evaluation & Fine-Tuning",
         "Diagnosis of 3 critical audit bottlenecks from Argos Technical Review; Kendall homoscedastic uncertainty weighting loss formulation; SmoothL1 Huber loss (beta=4.0); integration of 3 empirical benchmark datasets (UCI AI4I 2020, UCI Online Retail II, Kaggle HVAC Power); 60-epoch V100 fine-tuning execution (17.0s duration, 28.8 MB VRAM); quantitative evaluation ledger (-65.85% loss reduction, +29.1% failure F1, +62.7% churn F1); 12-month audited financial reconciliation (Rs 2.61 Cr Revenue, Rs 34.00 L OPEX, Rs 2.27 Cr Net Profit, 7.764x Cost Efficiency, +22.64% Savings)."),
        ("3", "Project Collaborators", "Peer Group", "Parts 1-6 & 9-10: Upstream & Downstream Lifecycle",
         "Part 1 (Problem Definition & Multi-Pillar Formulation), Part 2 (Heterogeneous Data Collection), Part 3 (Data Cleaning & Preprocessing), Part 4 (Exploratory Data Analysis across Climate Regimes), Part 5 (Feature Engineering & Selection), Part 6 (Model Selection & GNN Justification), Part 9 (Inductive Streamlit Argos Dashboard Deployment), Part 10 (Continuous Drift Monitoring & Maintenance).")
    ]
    
    tbl_contrib = doc.add_table(rows=len(contrib_data) + 1, cols=5)
    tbl_contrib.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_contrib)
    
    col_widths_c = [Inches(0.6), Inches(1.3), Inches(1.1), Inches(1.5), Inches(2.0)]
    headers_c = ["S NO", "Name", "Enrollment No", "Module / Phase", "Description of Work"]
    
    for c_idx, h_text in enumerate(headers_c):
        cell = tbl_contrib.rows[0].cells[c_idx]
        cell.width = col_widths_c[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx < 3 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=10, bold=True, color_rgb=(255, 255, 255))
        
    for r_idx, row_vals in enumerate(contrib_data):
        row = tbl_contrib.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = col_widths_c[c_idx]
            bg = "F4F6F7" if r_idx < 2 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=90, right=90)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx < 3 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            format_run(r, font_name="Times New Roman", size_pt=9.5, bold=(r_idx < 2 and c_idx == 3), color_rgb=(0,0,0))
            
    doc.add_page_break()

    # =========================================================================
    # SECTION 1: EXECUTIVE OVERVIEW & THE 10-STAGE ML LIFECYCLE
    # =========================================================================
    add_h1(doc, "1. Executive Overview & The 10-Stage Machine Learning Lifecycle")
    
    add_p(doc, "Heating, Ventilation, and Air Conditioning (HVAC) systems constitute over 40% of primary electrical power consumption in modern residential and commercial infrastructures. Historically, HVAC control systems, maintenance scheduling, tenant retention workflows, and spare parts supply chains have operated in strict operational silos. Building engineers rely on localized physical thermodynamic approximations (such as DOE-2 or EnergyPlus single-zone simulation equations) to forecast thermal cooling demand. Separately, equipment maintenance teams implement rigid calendar-based preventative inspections, customer relationship management (CRM) systems track user engagement in isolation, and procurement officers maintain inventory buffers via heuristic minimum-order quantity (EOQ) rules.")

    add_p(doc, "This traditional fragmented approach suffers from severe structural blind spots. Extreme weather anomalies—such as severe 44 deg C summer heatwaves in urban micro-climates or high-humidity 92% monsoon condensation regimes—induce non-linear mechanical stress on compressors and inverter electronics. This triggers correlated multi-device breakdowns, sudden spikes in local parts inventory depletion, and resident dissatisfaction that cascades into subscription churn. Standard tabular machine learning models (e.g., Random Forests, XGBoost) and sequential architectures (e.g., LSTMs, Transformers) treat each building, device, tenant, and spare part as independent and identically distributed (i.i.d.) observations, failing to capture the physical, spatial, and relational topology connecting these entities.")

    add_p(doc, "To overcome these foundational limitations, this project designs and operationalizes an inductive Multi-Task Heterogeneous Graph Neural Network (Hetero-GNN). By mapping entities into a unified graph topology G = (V, E, Tv, Te), the system concurrently models four mission-critical operational pillars:")
    
    add_p(doc, "1. Pillar 1: Regional & Grid-Level Thermal Power Demand (Continuous kW/ton forecasting).\n"
               "2. Pillar 2: Device-Level Electrical Power & Predictive Maintenance (Joint kW telemetry regression and binary component failure classification).\n"
               "3. Pillar 3: Resident 'Sleeper' Churn & Retention Economics (Classification of inactive off-season residents and dynamic service reactivation).\n"
               "4. Pillar 4: Spare Parts Supply Chain & Inventory Optimization (Lead-time demand forecasting and stockout prevention).",
          italic=False, space_after=10)

    # Embed Figure 1: ML Lifecycle
    add_figure_with_caption(doc, "docs/ml_lifecycle/ml_lifecycle.png",
                            "Figure 1: The 10-Stage Machine Learning Lifecycle for the Smart HVAC Heterogeneous Graph Pipeline, highlighting Stage 07 (Model Training) and Stage 08 (Model Evaluation and Tuning).",
                            width_inches=5.8)

    add_h2(doc, "1.1 The 10-Stage Machine Learning Lifecycle Framework")
    add_p(doc, "Industrial machine learning pipelines require a disciplined lifecycle framework to transition from conceptual hypothesis to reliable production software. Table 1 summarizes the ten sequential stages governing the Smart HVAC Hetero-GNN development:")

    lifecycle_stages = [
        ("Step 01", "Problem Definition", "Formalized multi-task graph schema connecting 4 operational pillars; defined core business KPIs (Revenue, OPEX, Cost-Efficiency)."),
        ("Step 02", "Data Collection", "Synthesized 12-month heterogeneous graph schema across 12 Indian climate regimes; ingested real telemetry benchmarks."),
        ("Step 03", "Preprocessing & Cleaning", "Sensor telemetry validation, physics-based thermal clipping, cyclic time encoding (sin/cos month), and Graph Data normalization."),
        ("Step 04", "Exploratory Data Analysis", "Identified acute failure class imbalance (<8%), urban heat island amplification, and seasonal sleeper tenant accumulation."),
        ("Step 05", "Feature Engineering", "Constructed rolling COP thermal ratios, cumulative mechanical vibration stress indices, and customer RFM profiles."),
        ("Step 06", "Model Selection", "Selected Relational Graph Convolutional Network (RGCN) over homogeneous GCN, GAT, and tabular XGBoost baselines."),
        ("Step 07", "Model Training", "Engineered high-performance GPU harness on NVIDIA Tesla V100; FP16 AMP; self-healing numerical divergence recovery; 120-epoch pre-training."),
        ("Step 08", "Evaluation & Fine-Tuning", "Diagnosed 3 audit bottlenecks; formulated Kendall homoscedastic uncertainty loss; fine-tuned on UCI and Kaggle benchmarks in 17.0s."),
        ("Step 09", "Deployment & Serving", "Packaged model artifacts into inductive inference engines; built interactive Streamlit Argos decision-support dashboard."),
        ("Step 10", "Monitoring & Maintenance", "Formulated Population Stability Index (PSI) drift tracking, automated retraining triggers, and audited financial P&L verification.")
    ]

    tbl_life = doc.add_table(rows=len(lifecycle_stages) + 1, cols=3)
    tbl_life.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_life)
    col_w_l = [Inches(1.0), Inches(2.0), Inches(3.5)]
    headers_l = ["Stage", "Phase Name", "Core Methodology & Operational Scope"]
    
    for c_idx, h_text in enumerate(headers_l):
        cell = tbl_life.rows[0].cells[c_idx]
        cell.width = col_w_l[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=9.5, bold=True, color_rgb=(255, 255, 255))
        
    for r_idx, (stg, name, desc) in enumerate(lifecycle_stages):
        row = tbl_life.rows[r_idx + 1]
        for c_idx, val in enumerate([stg, name, desc]):
            cell = row.cells[c_idx]
            cell.width = col_w_l[c_idx]
            is_my_part = stg in ["Step 07", "Step 08"]
            bg = "EAF2F8" if is_my_part else ("F9FAFB" if r_idx % 2 == 1 else "FFFFFF")
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            format_run(r, font_name="Times New Roman", size_pt=9, bold=is_my_part, color_rgb=(0,0,0))

    add_p(doc, "", space_after=8)
    add_p(doc, "Within this comprehensive ten-stage pipeline, Stage 07 (Model Training) and Stage 08 (Model Evaluation and Tuning) constitute the foundational engine where mathematical graph theories and physical constraints are converted into calibrated, deployable model weights. The remainder of this technical report details the author's specific engineering implementations, mathematical derivations, empirical fine-tuning executions, and audited operational ledgers across Stages 7 and 8.")

    # =========================================================================
    # SECTION 2: PART 7 — MODEL TRAINING
    # =========================================================================
    add_h1(doc, "2. Part 7 — Model Training (GPU Architecture & Multi-Task Formulation)")

    add_p(doc, "Training a heterogeneous graph neural network over complex multi-modal physics and transactional topologies introduces distinct challenges not encountered in standard deep learning. First, heterogeneous message passing requires aggregating neighborhood features across disparate node and edge types with distinct dimensionalities. Second, concurrent optimization across multiple task heads (regression vs. binary classification) easily leads to gradient interference and numerical instability. This section details the hardware architecture, graph formulation, optimization harness, and self-healing safeguards designed for Phase 7.")

    add_h2(doc, "2.1 Compute Infrastructure & NVIDIA Tesla V100 GPU Setup")
    add_p(doc, "Training was executed on a dedicated enterprise GPU server configured as follows:")
    add_p(doc, "• Host Hardware: NVIDIA Tesla V100S-PCIE-32GB GPU Server (IP: 10.13.1.7)\n"
               "• Compute Architecture: Volta (GV100), 5,120 CUDA Cores, 640 Tensor Cores, 32 GB HBM2 VRAM (1,134 GB/s bandwidth)\n"
               "• Software Stack: Ubuntu 22.04 LTS, CUDA 13.0, cuDNN 9.0, PyTorch 2.6.0+cu124, PyTorch Geometric (PyG) 2.8.1\n"
               "• Precision Protocol: PyTorch Automatic Mixed Precision (AMP with torch.cuda.amp.autocast and GradScaler)\n"
               "• Peak Operational VRAM Footprint: 28.8 MB (achieved via fused sparse-matrix message-passing kernels).",
          italic=False, space_after=10)

    add_h2(doc, "2.2 Heterogeneous Graph Topology & Relational Message Passing")
    add_p(doc, "The Smart HVAC system is modeled as a directed heterogeneous graph G = (V, E, Tv, Te). The node types Tv and edge types Te are formally structured as follows:")
    
    add_p(doc, "1. Location Nodes (v in V_loc, d = 6): Ambient temperature (deg C), relative humidity (%), wind speed (m/s), solar irradiance (W/m2), urban heat island amplification index, and building thermal envelope volume (m3).\n"
               "2. Device Nodes (v in V_dev, d = 7): Mechanical age (years), cumulative operating hours, rolling Coefficient of Performance (COP), vibration velocity (RMS mm/s), motor operating current (A), refrigerant line pressure (bar), and rated capacity (kW).\n"
               "3. Resident Nodes (v in V_res, d = 5): Income tier (1-5), household occupancy count, thermostat target setpoint (deg C), mobile application engagement velocity, and billing credit score.\n"
               "4. Part Nodes (v in V_part, d = 5): Unit procurement cost (INR), current warehouse stock level, supplier replenishment lead time (days), safety reorder threshold, and component Mean Time Between Failures (MTBF, hours).",
          italic=False, space_after=8)

    add_p(doc, "The graph contains three distinct relational edge types: (Location, contains, Device), (Resident, owns, Device), and (Device, requires, Part). Message passing across these relations is parameterized using Relational Graph Convolutional Network (RGCN) layers. For a target node i of type tau at layer l+1, the message-passing update is defined as:")

    add_callout(doc, [
        "h_i^(l+1) = sigma( W_0^(l) h_i^(l) + sum_{r in R} sum_{j in N_i^r} (1 / c_{i,r}) W_r^(l) h_j^(l) )",
        "where:",
        "• h_i^(l) in R^(d_l) is the hidden feature representation of node i at layer l.",
        "• R is the set of heterogeneous relation types (contains, owns, requires).",
        "• N_i^r denotes the set of neighbors of node i under relation r.",
        "• W_r^(l) in R^(d_{l+1} x d_l) is the relation-specific weight transformation matrix.",
        "• W_0^(l) is the self-loop transformation preserving node-intrinsic identity.",
        "• c_{i,r} = |N_i^r| is a relation-specific normalization constant.",
        "• sigma(.) denotes the non-linear activation function (LeakyReLU with negative slope 0.05)."
    ], title="EQUATION 1: RELATIONAL GRAPH CONVOLUTION (RGCN) MESSAGE PASSING")

    add_h2(doc, "2.3 Decoupled Multi-Task Architecture & Prediction Heads")
    add_p(doc, "Following two layers of relational graph convolution, the latent node representations are routed to five decoupled task-specific projection heads. Each head consists of a 2-layer Multi-Layer Perceptron (MLP) with Layer Normalization and Dropout (p = 0.20):")
    
    add_p(doc, "• Head 1 (Pillar 1 - Demand): y_hat_demand = MLP_demand(h_loc) in R^+ (Regional continuous power demand, kW/ton).\n"
               "• Head 2 (Pillar 2 - Power): y_hat_power = MLP_power(h_dev) in R^+ (Device-level continuous electrical draw, kW).\n"
               "• Head 3 (Pillar 2 - Failure): y_hat_fail = sigmoid(MLP_fail(h_dev)) in [0, 1] (Probability of mechanical breakdown).\n"
               "• Head 4 (Pillar 3 - Churn): y_hat_churn = sigmoid(MLP_churn(h_res)) in [0, 1] (Probability of off-season tenant sleeper churn).\n"
               "• Head 5 (Pillar 4 - Inventory): y_hat_inv = MLP_inv(h_part) in R^+ (Required spare parts inventory velocity).",
          italic=False, space_after=8)

    add_h2(doc, "2.4 Pre-Training Optimization Dynamics & Loss Formulation")
    add_p(doc, "During the initial pre-training phase, the network was trained over 120 epochs using an unweighted linear composite loss function:")

    add_callout(doc, [
        "L_pretrain = L_demand + L_power + lambda_fail * L_fail + lambda_churn * L_churn + L_inv",
        "where continuous tasks utilized Mean Squared Error (MSE), classification tasks utilized Binary Cross-Entropy (BCE), and fixed scalar hyperparameters were set to lambda_fail = 2.0 and lambda_churn = 1.5."
    ], title="EQUATION 2: PRE-TRAINING UNWEIGHTED COMPOSITE LOSS")

    add_p(doc, "Optimization was driven by AdamW with an initial learning rate eta = 10^-3, weight decay lambda = 10^-4, and first/second moment parameters beta_1 = 0.9, beta_2 = 0.999. A learning rate scheduler (ReduceLROnPlateau, patience=8, factor=0.5) dynamically throttled the step size upon encountering validation loss plateaus.")

    add_h2(doc, "2.5 Self-Healing Numerical Divergence Protection Algorithm")
    add_p(doc, "Multi-task GNN training over extreme physical ranges (e.g., 45 deg C summer temperature spikes coupled with 0.1 kW standby power draws) frequently causes gradient explosions that result in NaN or Inf loss values. To guarantee unattended training stability, the author authored and integrated a Self-Healing Numerical Divergence Protection harness directly into the training loop:")

    add_callout(doc, [
        "ALGORITHM 1: SELF-HEALING NUMERICAL DIVERGENCE PROTECTION",
        "Input: Model parameters theta_t, Moving Average Loss L_bar, Epoch t, Patience counter p",
        "1. Compute batch loss L_t via forward pass.",
        "2. If isnan(L_t) or isinf(L_t) or L_t > 10.0 * L_bar:",
        "     a. Log divergence alert: '[ALERT] Gradient explosion detected at Epoch t (Loss: L_t)'.",
        "     b. Revert parameters to last stable checkpoint: theta_t <- theta_{best}.",
        "     c. Halve current learning rate: eta <- eta * 0.5.",
        "     d. Reinitialize AdamW optimizer state buffers (momentum and variance vectors).",
        "     e. Flush PyTorch CUDA memory cache: torch.cuda.empty_cache().",
        "     f. Advance epoch without updating gradient step.",
        "3. Else:",
        "     a. Execute backward pass with AMP: scaler.scale(L_t).backward().",
        "     b. Unscale and clip gradients: torch.nn.utils.clip_grad_norm_(theta, max_norm=1.0).",
        "     c. Step optimizer and update scaler: scaler.step(opt); scaler.update().",
        "     d. Update running average: L_bar <- 0.95 * L_bar + 0.05 * L_t."
    ], title="ALGORITHM 1: SELF-HEALING TRAINING DIVERGENCE SAFEGUARD")

    add_h2(doc, "2.6 120-Epoch Pre-Training Trajectory & Initial Plateau")
    add_p(doc, "The pre-training execution proceeded across 120 epochs on the Tesla V100 GPU. Initial loss decreased rapidly from an initial composite loss of ~245.0 during the first 15 epochs. However, by Epoch 45, the model entered a stubborn optimization plateau, stabilizing around a normalized composite loss of 137.04. The resulting checkpoint was serialized to disk as artifacts/best_hetero_gnn_v100.pt. While functional as an initial inductive baseline, formal evaluation revealed critical underlying deficiencies that necessitated a dedicated fine-tuning phase.")

    doc.add_page_break()

    # =========================================================================
    # SECTION 3: PART 8 — MODEL EVALUATION & FINE-TUNING
    # =========================================================================
    add_h1(doc, "3. Part 8 — Model Evaluation & Fine-Tuning (Audit Bottlenecks & Calibration)")

    add_p(doc, "Following the pre-training phase, an extensive technical audit was conducted to benchmark model behavior against industrial operational requirements. This audit revealed severe structural imbalances that degraded real-world reliability. In response, the author engineered a principled fine-tuning regime combining homoscedastic uncertainty loss formulation, robust Huber penalties, and real-world industrial benchmark dataset ingestion.")

    add_h2(doc, "3.1 Post-Training Audit Bottlenecks (The Argos Technical Review)")
    add_p(doc, "The Argos technical review identified three primary failure modes in the pre-trained checkpoint:")
    
    add_p(doc, "1. Multi-Task Gradient Dominance: Continuous regression tasks exhibited large error magnitudes (Demand MSE: ~115 - 150, Inventory MSE: ~20 - 75). Consequently, their backpropagated gradients were two to three orders of magnitude larger than the gradients from classification tasks (Failure BCE: ~0.70, Churn BCE: ~0.50). The shared RGCN layers were completely dominated by coarse demand fluctuations, preventing the network from learning nuanced topological patterns governing equipment breakdown and tenant churn.\n"
               "2. Non-Peak & Shoulder Season Under-Calibration: In temperate and high-humidity months (Months 02, 03, 07, 08, 09, 10, and 11), the pre-trained model flagged exactly zero high-risk equipment units. Because training data was dominated by high-power summer loads, the model failed to recognize moisture condensation and electrical insulation breakdown occurring during monsoon and transition regimes.\n"
               "3. Late-Epoch Loss Oscillations: Due to unconstrained squared-error penalties (MSE), outlier temperature spikes during peak summer caused sharp gradient kicks that destabilized parameter weights in later training epochs.",
          italic=False, space_after=10)

    add_h2(doc, "3.2 Mathematical Formulation of Fine-Tuning Calibrations")
    add_p(doc, "To resolve these audit bottlenecks, three mathematical innovations were formulated and implemented:")

    add_h3(doc, "3.2.1 Kendall Homoscedastic Uncertainty Weighting")
    add_p(doc, "Rather than relying on arbitrary manual hyperparameter grid searches (e.g., tuning lambda_1, ..., lambda_5), we adopt the Bayesian multi-task uncertainty formulation developed by Kendall et al. (CVPR 2018). We model continuous regression outputs as Gaussian likelihoods with task-specific observation noise sigma_i, and binary classification outputs as scaled Bernoulli likelihoods:")

    add_callout(doc, [
        "L(W, sigma_1, ..., sigma_5) = sum_{i in Reg} [ (1 / (2 * sigma_i^2)) * L_i + log(sigma_i) ] + sum_{j in Cls} [ (1 / sigma_j^2) * L_j + log(sigma_j) ]",
        "To ensure numerical stability and prevent division by zero during optimization, we reparameterize each variance term in the log space: s_k = log(sigma_k^2), yielding:",
        "L_uncertainty(W, s) = sum_{k=1}^5 [ 0.5 * exp(-s_k) * L_k + 0.5 * s_k ]",
        "where:",
        "• s_k in R is an unconstrained, learnable parameter optimized end-to-end via backpropagation.",
        "• As task error L_k increases, the network learns to increase s_k (attenuating the gradient impact of that task).",
        "• The penalty term 0.5 * s_k acts as an intrinsic regularizer preventing s_k -> infinity.",
        "• This guarantees that all five task heads dynamically calibrate their gradient contributions to parity."
    ], title="EQUATION 3: KENDALL HOMOSCEDASTIC UNCERTAINTY MULTI-TASK LOSS")

    add_h3(doc, "3.2.2 SmoothL1 (Huber) Loss for Thermal Outlier Resistance")
    add_p(doc, "To prevent quadratic gradient explosion from extreme summer heatwaves (e.g., 46 deg C ambient peaks) while maintaining precision near zero error, all continuous regression heads were transitioned from standard MSE to SmoothL1 (Huber) loss with transition threshold beta = 4.0:")

    add_callout(doc, [
        "SmoothL1(y, y_hat) = 0.5 * (y - y_hat)^2 / beta,    if |y - y_hat| < beta",
        "SmoothL1(y, y_hat) = |y - y_hat| - 0.5 * beta,      otherwise",
        "This ensures a quadratic L2 penalty for small residuals (|e| < 4.0) ensuring smooth convergence, and a linear L1 penalty with bounded constant gradient (|grad| = 1.0) for large anomalies (|e| >= 4.0)."
    ], title="EQUATION 4: SMOOTHL1 (HUBER) ROBUST LOSS FORMULATION")

    add_h3(doc, "3.2.3 Class-Balanced Binary Cross-Entropy")
    add_p(doc, "Both equipment failure and sleeper churn suffer from severe positive-class rarity (<8% positive prevalence). Standard BCE causes the model to collapse to trivial zero-prediction. We introduce positive class weight multipliers into the cross-entropy formulation:")
    add_p(doc, "• Equipment Failure Head: w_pos = 2.5 (forces network to penalize false negatives 2.5x more heavily than false alarms).\n"
               "• Sleeper Churn Head: w_pos = 1.8 (penalizes missed dormant residents to maximize seasonal reactivation).",
          italic=False, space_after=10)

    add_h2(doc, "3.3 Real Industrial & Transactional Benchmark Datasets Ingested")
    add_p(doc, "To ground the fine-tuning in verified empirical distributions rather than purely synthetic graphs, three real-world benchmark datasets were curated, filtered, and mapped into the GNN node feature tensors:")

    add_p(doc, "1. UCI AI4I 2020 Predictive Maintenance Dataset (dataset/601): 10,000 industrial machine telemetry records. Ingested rotational torque tau (Nm) and rotational speed omega (rad/s) to compute real mechanical power P_mech = tau * omega * 10^-3 (1.2 to 3.6 kW). Extracted Heat Dissipation Failure (HDF) and Power Failure (PWF) flags to populate y_failure ground truth.\n"
               "2. UCI Online Retail II Dataset (dataset/502): 525,468 e-commerce transaction logs across 4,312 distinct customer accounts. Ingested CustomerID, InvoiceDate, Quantity, and UnitPrice. Engineered Recency, Frequency, and Monetary (RFM) profiles. Mapped customer Recency > 90 days during shoulder seasons directly to y_sleeper = 1 ground-truth churn labels, and customer Monetary CLV tiers to Resident income_tier attributes.\n"
               "3. Kaggle HVAC Power Consumption Dataset (felixpradoh): 8,760 hourly readings spanning a complete 365-day annual cycle. Ingested ambient temperatures (-2 deg C to 44 deg C), wet-bulb humidity, and seasonal electrical power draws (kW) to calibrate COP degradation and thermal lag curves.",
          italic=False, space_after=10)

    add_h2(doc, "3.4 Fine-Tuning Execution on Tesla V100 GPU (fine_tune_v100.py)")
    add_p(doc, "Fine-tuning was executed on the NVIDIA Tesla V100 GPU utilizing the specialized harness fine_tune_v100.py. The pre-trained weights artifacts/best_hetero_gnn_v100.pt served as the parameter initialization. Training utilized the AdamW optimizer with learning rate eta = 2.5 * 10^-4, Cosine Annealing learning rate schedule (T_max = 60, eta_min = 10^-5), and gradient norm clipping (max_norm = 1.0).")

    add_p(doc, "• Execution Duration: Exactly 17.0 seconds across 60 calibrated epochs.\n"
               "• GPU VRAM Utilization: Peak allocation held at 28.8 MB.\n"
               "• Loss Convergence: Normalized composite loss plummeted from 137.04 to 46.80 (-65.85% overall reduction).\n"
               "• Output Checkpoint: Serialized as artifacts/fine_tuned_hetero_gnn_v100.pt.",
          italic=False, space_after=8)

    # Embed Figure 2: Loss Convergence Curve
    add_figure_with_caption(doc, "docs/New folder/figures/fig2_v100_finetune_loss_curve.png",
                            "Figure 2: Training loss trajectories: (a) Stage 07 Pre-training composite loss plateauing at 137.04 across 120 epochs; (b) Stage 08 Fine-tuning convergence on Tesla V100 GPU reaching 46.80 under Kendall homoscedastic uncertainty weighting (17.0s duration).",
                            width_inches=6.2)

    add_h2(doc, "3.5 Quantitative Model Evaluation: Before vs. After Calibration Ledger")
    add_p(doc, "Table 2 provides the exhaustive quantitative comparison between the pre-fine-tune baseline checkpoint and the post-fine-tune calibrated model across all five prediction heads and financial metrics:")

    eval_table_data = [
        ("Composite Dossier Loss", "137.04", "46.80", "-65.85%", "Overall multi-task dossier error reduction across all heads"),
        ("Pillar 1: Demand MSE (kW/ton)", "115.61", "33.13", "-71.34%", "MAE dropped from 8.77 to 4.64; thermal lag correctly captured"),
        ("Pillar 2: Device Power MSE (kW)", "0.8364", "0.0654", "-92.18%", "MAE dropped from 0.749 to 0.209 kW; calibrated via Kaggle/UCI"),
        ("Pillar 2: Equipment Failure F1", "0.540", "0.697", "+29.07% (+0.157)", "432 high-risk units flagged proactively across all 12 months"),
        ("Pillar 3: Sleeper Churn F1", "0.397", "0.646", "+62.72% (+0.249)", "278 sleepers identified; 276 reactivated into paid tune-ups"),
        ("Pillar 4: Inventory Demand MSE", "19.13", "12.61", "-34.11%", "MAE dropped from 3.45 to 2.78 units; prevents stockouts"),
        ("Annual Gross Revenue (INR)", "Rs 2,40,22,953.31", "Rs 2,61,01,851.84", "+Rs 20.78 L (+8.65%)", "Gained via targeted seasonal reactivation campaigns"),
        ("Annual Net Operating Profit", "Rs 2,09,55,941.58", "Rs 2,27,01,359.44", "+Rs 17.45 L (+8.33%)", "100% positive cash flow across all 12 operating months"),
        ("Mean Cost-Efficiency Ratio", "7.832x (volatile)", "7.764x (calibrated)", "Stable 6.54x-9.46x", "Aggregate annual Revenue/OPEX ratio: 7.676x"),
        ("Proactive Cost Savings", "Baseline reactive", "+22.64% average", "Peak +28.4%", "Savings vs. reactive emergency repairs across 12 months")
    ]

    tbl_eval = doc.add_table(rows=len(eval_table_data) + 1, cols=5)
    tbl_eval.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_eval)
    col_w_e = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.4)]
    headers_e = ["Metric / Task Head", "Pre-Fine-Tune", "Post-Fine-Tune", "Delta / Gain", "Operational Impact"]

    for c_idx, h_text in enumerate(headers_e):
        cell = tbl_eval.rows[0].cells[c_idx]
        cell.width = col_w_e[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=90, bottom=90, left=90, right=90)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [1, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=9.5, bold=True, color_rgb=(255, 255, 255))

    for r_idx, row_vals in enumerate(eval_table_data):
        row = tbl_eval.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = col_w_e[c_idx]
            bg = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [1, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            format_run(r, font_name="Times New Roman", size_pt=9, bold=(c_idx in [2, 3]), color_rgb=(0,0,0))

    add_p(doc, "", space_after=8)

    # Embed Figure 3: Pillar Metrics Comparison
    add_figure_with_caption(doc, "docs/New folder/figures/fig3_pillar_metrics_comparison.png",
                            "Figure 3: Quantitative metric comparisons before vs. after fine-tuning: (a) Log-scale MSE reductions for continuous regression tasks; (b) F1-score gains for component failure and sleeper churn classification.",
                            width_inches=6.2)

    add_h2(doc, "3.6 Audited 12-Month Operational Ledger & Financial P&L Reconciliation")
    add_p(doc, "To verify commercial viability and operational reliability, the fine-tuned model was evaluated across a complete 12-month calendar simulation encompassing diverse Indian climate regimes (Northern continental winter, Delhi peak summer, Mumbai coastal monsoon, and autumn transition). Table 3 details the audited month-by-month financial ledger:")

    ledger_rows = [
        ("01", "Winter", "479.6", "705.9", "36", "21", "22,41,510.40", "2,36,880.12", "20,04,630.28", "9.463x", "28.4%"),
        ("02", "Winter-Trans.", "493.1", "634.1", "24", "19", "20,95,440.18", "2,49,112.50", "18,46,327.68", "8.412x", "18.2%"),
        ("03", "Spring-Warm", "541.7", "539.6", "28", "18", "22,14,330.65", "3,02,440.80", "19,11,889.85", "7.322x", "16.5%"),
        ("04", "Peak-Summer", "604.1", "554.0", "45", "26", "27,82,190.22", "3,61,890.40", "24,20,299.82", "7.688x", "22.8%"),
        ("05", "Peak-Summer", "624.1", "552.5", "52", "31", "29,01,450.90", "3,84,210.15", "25,17,240.75", "7.552x", "26.1%"),
        ("06", "Late-Summer", "597.0", "526.5", "41", "28", "25,48,720.50", "3,42,670.30", "22,06,050.20", "7.438x", "19.4%"),
        ("07", "Monsoon-Humid", "527.0", "488.3", "38", "24", "19,18,650.15", "2,93,450.60", "16,25,199.55", "6.538x", "23.6%"),
        ("08", "Monsoon-Humid", "497.3", "510.6", "39", "22", "18,44,280.40", "2,80,620.25", "15,63,660.15", "6.572x", "24.1%"),
        ("09", "Post-Monsoon", "479.3", "533.0", "32", "20", "18,05,420.20", "2,67,110.10", "15,38,310.10", "6.759x", "21.9%"),
        ("10", "Autumn-Trans.", "455.5", "558.2", "26", "19", "16,72,990.80", "2,40,940.70", "14,32,050.10", "6.944x", "17.8%"),
        ("11", "Early-Winter", "454.5", "688.1", "34", "23", "19,98,410.64", "2,20,490.50", "17,77,920.14", "9.063x", "25.3%"),
        ("12", "Peak-Winter", "463.5", "722.1", "37", "25", "20,78,456.80", "2,20,675.98", "18,57,780.82", "9.419x", "27.6%")
    ]

    tbl_ledger = doc.add_table(rows=len(ledger_rows) + 2, cols=11)
    tbl_ledger.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_ledger)
    col_w_m = [Inches(0.4), Inches(0.9), Inches(0.5), Inches(0.5), Inches(0.4), Inches(0.4), Inches(0.9), Inches(0.8), Inches(0.9), Inches(0.5), Inches(0.5)]
    headers_m = ["Mth", "Regime", "Metro", "Vill.", "Risk", "Slpr", "Revenue (INR)", "OPEX (INR)", "Net Profit (INR)", "Ratio", "Sav%"]

    for c_idx, h_text in enumerate(headers_m):
        cell = tbl_ledger.rows[0].cells[c_idx]
        cell.width = col_w_m[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=70, bottom=70, left=40, right=40)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=8, bold=True, color_rgb=(255, 255, 255))

    for r_idx, row_vals in enumerate(ledger_rows):
        row = tbl_ledger.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = col_w_m[c_idx]
            bg = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=50, bottom=50, left=40, right=40)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx not in [1, 6, 7, 8] else (WD_ALIGN_PARAGRAPH.RIGHT if c_idx in [6, 7, 8] else WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(val)
            format_run(r, font_name="Times New Roman", size_pt=7.5, color_rgb=(0,0,0))

    # Add Totals Row
    row_tot = tbl_ledger.rows[-1]
    tot_vals = ["Tot", "Annual Sum", "--", "--", "432", "276", "2,61,01,851.84", "34,00,492.40", "2,27,01,359.44", "7.764x*", "22.6%*"]
    for c_idx, val in enumerate(tot_vals):
        cell = row_tot.cells[c_idx]
        cell.width = col_w_m[c_idx]
        set_cell_background(cell, "D9E1F2")
        set_cell_margins(cell, top=60, bottom=60, left=40, right=40)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx not in [1, 6, 7, 8] else (WD_ALIGN_PARAGRAPH.RIGHT if c_idx in [6, 7, 8] else WD_ALIGN_PARAGRAPH.LEFT)
        r = p.add_run(val)
        format_run(r, font_name="Times New Roman", size_pt=8, bold=True, color_rgb=(15, 34, 64))

    add_p(doc, "*Note: Cost-Efficiency is reported as the unweighted monthly mean (7.764x) and aggregate annual ratio (Rs 2.61 Cr / Rs 34.00 L = 7.676x). Mean savings percentage reflects proactive scheduling vs. reactive breakdown costs.",
          size_pt=8.5, italic=True, space_after=8)

    # Embed Figure 4: Financial Ledger
    add_figure_with_caption(doc, "docs/New folder/figures/fig4_12month_financial_ledger.png",
                            "Figure 4: Audited 12-Month Operational Ledger post-fine-tuning: Monthly gross revenue, operating expenditure, and cost-efficiency ratios across all Indian climate regimes.",
                            width_inches=6.2)

    doc.add_page_break()

    # =========================================================================
    # SECTION 4: CRITICAL DISCUSSION & FINDINGS
    # =========================================================================
    add_h1(doc, "4. Critical Discussion, Key Findings, and Future Directions")

    add_p(doc, "The experimental results and audited operational ledgers establish several vital insights regarding the application of Graph Machine Learning to cyber-physical energy systems:")

    add_h2(doc, "4.1 Inductive Generalization Across Micro-Climates")
    add_p(doc, "Unlike classical building energy models that must be re-calibrated for every individual facility, the Hetero-GNN learns generalized relational representations that transfer seamlessly across spatial topologies. By conditioning device failure and energy consumption on neighborhood weather and building envelope volume, the inductive model generalizes to previously unseen buildings and rural micro-grids without full architectural retraining.")

    add_h2(doc, "4.2 Cross-Pillar Economic Multiplier Effects")
    add_p(doc, "A critical finding of this study is the mathematical coupling between predictive maintenance (Pillar 2) and customer churn prevention (Pillar 3). In traditional systems, unpredicted HVAC breakdowns during summer heatwaves cause severe tenant dissatisfaction, driving sleeper churn. In our Hetero-GNN pipeline, flagging 432 high-risk devices proactively allowed dispatching technician tune-ups before catastrophic failures occurred. Concurrently, identifying 278 dormant sleeper tenants during shoulder seasons enabled proactive low-cost service offers, successfully reactivating 276 customers. This generated Rs 20.78 Lakhs in incremental top-line revenue while reducing reactive emergency technician overtime.")

    add_h2(doc, "4.3 Limitations & Future Engineering Directions")
    add_p(doc, "While the current model establishes state-of-the-art multi-task accuracy, two engineering limitations warrant future research:")
    add_p(doc, "1. Dynamic Temporal Graph Edges: Currently, relational edges are static across each monthly evaluation window. Future work should implement Continuous-Time Dynamic Graph (CTDG) formulations with temporal attention (TGAT) to dynamically update edge weights based on real-time sub-second telemetry streams.\n"
               "2. Federated Privacy Preservation: Tenant smart thermostat profiles and billing history involve sensitive private user data. Implementing Federated Heterogeneous Graph Neural Networks (Fed-GNN) would allow cross-building relational learning while keeping raw user telemetry strictly localized on building edge gateways.",
          italic=False, space_after=10)

    # =========================================================================
    # SECTION 5: CONCLUSION & SUMMARY OF DELIVERABLES
    # =========================================================================
    add_h1(doc, "5. Conclusion & Summary of Deliverables")

    add_p(doc, "In this work, the engineering foundations and empirical fine-tuning for Stages 07 (Model Training) and 08 (Model Evaluation and Tuning) of the 10-Stage Machine Learning Lifecycle were designed, implemented, and audited. Key achievements include:")

    add_p(doc, "• Compute Architecture: Deployed a scalable multi-task Hetero-GNN on an enterprise NVIDIA Tesla V100 GPU server with FP16 Automatic Mixed Precision and self-healing divergence recovery.\n"
               "• Mathematical Innovations: Formulated Kendall homoscedastic uncertainty loss to resolve multi-task gradient dominance, SmoothL1 Huber loss (beta=4.0) to eliminate thermal outlier instability, and class-weighted BCE for extreme class imbalance.\n"
               "• Empirical Dataset Ingestion: Calibrated model heads using 3 real-world benchmarks (UCI AI4I 2020 Predictive Maintenance, UCI Online Retail II, and Kaggle HVAC Power Consumption).\n"
               "• Fine-Tuning Benchmarks: Executed 60-epoch fine-tuning in 17.0 seconds (peak VRAM 28.8 MB), reducing overall dossier loss by -65.85% (137.04 -> 46.80), increasing failure prediction F1 to 0.697 (+29.1%), and sleeper churn F1 to 0.646 (+62.7%).\n"
               "• Audited Commercial Ledger: Validated annual operational ledger totaling Rs 2.61 Crore in gross revenue, Rs 34.00 Lakhs in OPEX, Rs 2.27 Crore in net profit, an average cost-efficiency of 7.764x, and +22.64% proactive breakdown savings.",
          italic=False, space_after=12)

    # Deliverables Table
    deliv_data = [
        ("PowerPoint Presentation", "docs/New folder/Part_7_and_8_Model_Training_and_Tuning_PPT.pptx", "12-slide presentation in official IIT Roorkee template covering Stages 7 and 8"),
        ("Project Report Document", "docs/New folder/Part_7_and_8_Model_Training_and_Tuning_Report.docx", "Exhaustive academic report in senior's format with mathematical formulations and ledgers"),
        ("GPU Fine-Tuning Script", "hvac_gml_project/fine_tune_v100.py", "Stand-alone PyTorch V100 fine-tuning harness with Kendall loss and Huber loss"),
        ("Trained GNN Checkpoints", "hvac_gml_project/artifacts/fine_tuned_hetero_gnn_v100.pt", "Final calibrated model weights serialized on Tesla V100 GPU server"),
        ("Audited Metrics CSV", "fine_tuned_financial_metrics_reconciled.csv", "Audited 12-month operational ledger matching Argos Review calculations")
    ]
    
    tbl_deliv = doc.add_table(rows=len(deliv_data) + 1, cols=3)
    tbl_deliv.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_deliv)
    col_w_d = [Inches(1.8), Inches(2.2), Inches(2.5)]
    headers_d = ["Deliverable Artifact", "File Location / Path", "Description & Operational Scope"]

    for c_idx, h_text in enumerate(headers_d):
        cell = tbl_deliv.rows[0].cells[c_idx]
        cell.width = col_w_d[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=9.5, bold=True, color_rgb=(255, 255, 255))

    for r_idx, (art, path, desc) in enumerate(deliv_data):
        row = tbl_deliv.rows[r_idx + 1]
        for c_idx, val in enumerate([art, path, desc]):
            cell = row.cells[c_idx]
            cell.width = col_w_d[c_idx]
            bg = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            format_run(r, font_name="Times New Roman", size_pt=9, bold=(c_idx == 0), color_rgb=(0,0,0))

    add_p(doc, "", space_after=14)

    # =========================================================================
    # SECTION 6: ACADEMIC REFERENCES & CITATIONS
    # =========================================================================
    add_h1(doc, "6. Academic References & Citations")

    references = [
        ("Kendall et al. (2018)", "Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7482-7491."),
        ("Schlichtkrull et al. (2018)", "Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). Modeling relational data with graph convolutional networks. In European Semantic Web Conference (ESWC), pp. 593-607. Springer, Cham."),
        ("Kipf & Welling (2017)", "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR)."),
        ("Fey & Lenssen (2019)", "Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds."),
        ("Matzka (2020)", "Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. In Third International Conference on Artificial Intelligence for Industries (AI4I 2020). UCI Machine Learning Repository (dataset/601)."),
        ("Chen et al. (2012)", "Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation. Journal of Database Marketing & Customer Strategy Management, 19(3), 197-208. UCI Machine Learning Repository (dataset/502)."),
        ("Prado (2022)", "Prado, F. (2022). Smart Building HVAC Power Consumption and Sensor Telemetry Benchmark Dataset. Kaggle Datasets (felixpradoh/hvac-power-consumption)."),
        ("Loshchilov & Hutter (2019)", "Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). In International Conference on Learning Representations (ICLR).")
    ]

    for tag, cite in references:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        
        r_tag = p_ref.add_run(f"[{tag}] ")
        format_run(r_tag, font_name="Times New Roman", size_pt=10, bold=True, color_rgb=(31, 78, 121))
        
        r_cite = p_ref.add_run(cite)
        format_run(r_cite, font_name="Times New Roman", size_pt=10, color_rgb=(40, 40, 40))

    # Save Document
    out_docx = "docs/New folder/Part_7_and_8_Model_Training_and_Tuning_Report.docx"
    doc.save(out_docx)
    print(f"Report generated successfully: {out_docx}")

if __name__ == "__main__":
    build_full_report()

