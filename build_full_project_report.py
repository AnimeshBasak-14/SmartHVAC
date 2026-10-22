import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
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

def add_callout(doc, text_list, title="ENGINEERING FORMULA"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, "F4F6F7")
    set_cell_margins(cell, top=110, bottom=110, left=160, right=130)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1F4E79"/>'
        f'<w:top w:val="none"/><w:bottom w:val="none"/><w:right w:val="none"/>'
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
        format_run(r_item, font_name="Times New Roman", size_pt=10, color_rgb=(30, 30, 30))
        
    p_end = doc.add_paragraph()
    p_end.paragraph_format.space_before = Pt(0)
    p_end.paragraph_format.space_after = Pt(4)

def add_figure_with_caption(doc, img_path, caption_text, width_inches=6.0):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(3)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(width_inches))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run(caption_text)
        format_run(r_cap, font_name="Times New Roman", size_pt=9.5, italic=True, color_rgb=(70, 70, 70))
    else:
        print(f"Warning: Image {img_path} not found.")

def build_full_report():
    doc = docx.Document()
    
    # 1.0 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("IIT Roorkee | Dept. of CSE | CSL-505 Project Report")
        format_run(hrun, font_name="Times New Roman", size_pt=8.5, italic=True, color_rgb=(120, 120, 120))
        
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Department of Computer Science & Engineering, IIT Roorkee")
        format_run(frun, font_name="Times New Roman", size_pt=9, italic=True, color_rgb=(120, 120, 120))

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    add_p(doc, "INDIAN INSTITUTE OF TECHNOLOGY ROORKEE", size_pt=21, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=16, space_after=6, color_rgb=(15, 34, 64))
    
    add_p(doc, "Department of Computer Science & Engineering", size_pt=15, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=16, color_rgb=(31, 78, 121))
    
    add_p(doc, "Project Report On", size_pt=14, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=4, color_rgb=(60, 60, 60))
    
    add_p(doc, "Heterogeneous Graph Neural Network for Smart HVAC Energy Forecasting, Predictive Maintenance, Customer Churn, and Inventory Optimization",
          size_pt=19, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=14, color_rgb=(15, 34, 64))
    
    add_p(doc, "Course: Machine Learning (CSL-505) / Data Mining (CSL-515)", size_pt=13, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=20, color_rgb=(31, 78, 121))
    
    add_p(doc, "Group Members:", size_pt=12, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=4, color_rgb=(60, 60, 60))
    
    add_p(doc, "Kriti | Dhruv | Ankit | Animesh Basak (26911001) | Saurav", size_pt=13, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=20, color_rgb=(15, 34, 64))
    
    add_p(doc, "Under The Guidance of:", size_pt=12, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=4, color_rgb=(60, 60, 60))
    
    add_p(doc, "Dr. Durga Toshniwal", size_pt=14, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2, color_rgb=(15, 34, 64))
    
    add_p(doc, "Professor, Department of Computer Science & Engineering\nIndian Institute of Technology Roorkee",
          size_pt=12, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=28, color_rgb=(60, 60, 60))
    
    doc.add_page_break()

    # =========================================================================
    # TABLE OF CONTENTS (Matching user's exact outline)
    # =========================================================================
    add_p(doc, "Table of Contents", size_pt=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
          space_before=10, space_after=12, color_rgb=(15, 34, 64))
    
    toc_data = [
        ("1", "Members' Contribution", "2"),
        ("2", "Introduction", "3"),
        ("3", "Problem Statement (The 4 Operational Pillars)", "4"),
        ("4", "The Dataset & Heterogeneous Graph Schema", "6"),
        ("5", "Dataset Preprocessing & Exploratory Data Analysis", "9"),
        ("6", "Feature Engineering and Selection", "12"),
        ("7", "Model Selection & Relational Architecture", "15"),
        ("8", "Core Lifecycle Execution: Model Training, Evaluation, & Deployment", "18"),
        ("8.1", "  8.1 Model Training on NVIDIA Tesla V100 GPU", "18"),
        ("8.2", "  8.2 Model Evaluation & Multi-Task Fine-Tuning", "21"),
        ("8.3", "  8.3 Model Deployment & Interactive Command Center", "26"),
        ("8.4", "  8.4 Model Monitoring & Drift Maintenance", "28"),
        ("9", "Results and Key Findings", "30"),
        ("10", "Challenges and Future Work", "33"),
        ("11", "Conclusion", "35"),
    ]
    
    tbl_toc = doc.add_table(rows=len(toc_data) + 1, cols=3)
    tbl_toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_toc)
    col_w_toc = [Inches(0.8), Inches(4.7), Inches(1.0)]
    
    headers_toc = ["Chapter", "Section Title", "Page No."]
    for c_idx, h_text in enumerate(headers_toc):
        cell = tbl_toc.rows[0].cells[c_idx]
        cell.width = col_w_toc[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=90, bottom=90, left=110, right=110)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 1 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=10, bold=True, color_rgb=(255, 255, 255))
        
    for r_idx, (sno, title, page) in enumerate(toc_data):
        row = tbl_toc.rows[r_idx + 1]
        for c_idx, val in enumerate([sno, title, page]):
            cell = row.cells[c_idx]
            cell.width = col_w_toc[c_idx]
            bg = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 1 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            is_main = not sno.startswith(" ") and not "." in sno
            format_run(r, font_name="Times New Roman", size_pt=9.5, bold=is_main, color_rgb=(0,0,0))
            
    add_p(doc, "", space_after=14)

    # =========================================================================
    # CHAPTER 1: MEMBERS' CONTRIBUTION
    # =========================================================================
    add_h1(doc, "Chapter 1: Members' Contribution")
    
    add_p(doc, "Our team divided the ten stages of the machine learning lifecycle across five members based on individual focus areas. Each member took direct ownership of specific lifecycle phases, from raw data ingestion to production monitoring:")

    contrib_data = [
        ("1", "Kriti", "Problem Definition &\nData Collection", "Ch 3 & 4",
         "Formulated the 4 operational business pillars (thermal demand, device failure, sleeper churn, inventory velocity). Collected and assembled the heterogeneous data schema across 12 monthly climate snapshots, defining 40 location nodes, 180 device nodes, 600 resident nodes, and 50 spare part SKUs."),
        ("2", "Dhruv", "Data Cleaning, Preprocessing\n& Exploratory Analysis (EDA)", "Ch 5",
         "Sanitized raw sensor telemetry by applying median filtering and thermodynamic capacity bounds. Implemented cyclic month encoding (sin and cos). Analyzed extreme positive class rarity in failure events (<8%) and examined temperature variations across metro and rural village hubs."),
        ("3", "Ankit", "Feature Engineering &\nModel Selection", "Ch 6 & 7",
         "Engineered rolling Coefficient of Performance (COP) degradation ratios, cumulative mechanical vibration stress metrics, and customer RFM (Recency, Frequency, Monetary) profiles. Selected Relational Graph Convolution (RGCN / HeteroConv) over standard tabular baselines."),
        ("4", "Animesh Basak\n(26911001)", "Model Training, Evaluation\n& Multi-Task Fine-Tuning", "Ch 8.1 & 8.2",
         "Built the GPU training harness on an NVIDIA Tesla V100 GPU with FP16 AMP and self-healing divergence recovery. Diagnosed audit bottlenecks (gradient dominance and non-peak under-calibration), applied Kendall homoscedastic uncertainty loss and Huber penalties, and fine-tuned on real UCI AI4I 2020, UCI Online Retail II, and Kaggle datasets (-65.85% loss reduction)."),
        ("5", "Saurav", "Model Deployment, Monitoring\n& Maintenance", "Ch 8.3 & 8.4",
         "Deployed the trained pipeline to Streamlit Cloud with an in-memory PyVis graph renderer, eliminating blank screen crashes. Formulated drift detection protocols (Kolmogorov-Smirnov test and Population Stability Index) and set up automated incremental retraining triggers.")
    ]

    tbl_c = doc.add_table(rows=len(contrib_data) + 1, cols=5)
    tbl_c.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_c)
    col_w_c = [Inches(0.5), Inches(1.4), Inches(1.5), Inches(0.8), Inches(2.3)]
    headers_c = ["S.No.", "Name", "Assigned Lifecycle Phase", "Chapter", "Concrete Work Done"]

    for c_idx, h_text in enumerate(headers_c):
        cell = tbl_c.rows[0].cells[c_idx]
        cell.width = col_w_c[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=90, bottom=90, left=90, right=90)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 3] else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=9.5, bold=True, color_rgb=(255, 255, 255))

    for r_idx, row_vals in enumerate(contrib_data):
        row = tbl_c.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = col_w_c[c_idx]
            bg = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 3] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            format_run(r, font_name="Times New Roman", size_pt=9, bold=(c_idx == 1), color_rgb=(0,0,0))

    add_p(doc, "", space_after=10)

    # Embed Figure 1: 10-Stage ML Lifecycle
    add_figure_with_caption(doc, "visualization/01_ml_lifecycle_10_stages.png",
                            "Figure 1: The 10-Stage Machine Learning Lifecycle framework structuring our group project implementation.",
                            width_inches=5.8)

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 2: INTRODUCTION
    # =========================================================================
    add_h1(doc, "Chapter 2: Introduction")

    add_p(doc, "Heating, ventilation, and air conditioning systems are responsible for roughly 40% of the total electricity consumed in modern residential townships and commercial complexes. Despite their massive energy footprint, these systems are managed in a remarkably fragmented way. In practice, three separate departments run their operations in complete isolation from one another:")

    add_p(doc, "First, facility engineers calculate cooling and heating demand using standalone building energy simulations like EnergyPlus. These models treat each building as an isolated thermodynamic box, ignoring neighboring buildings and grid conditions.\n\n"
               "Second, field technicians handle appliance maintenance using fixed calendar schedules. A crew visits a compressor once every six months, regardless of whether the machine is running smoothly or shaking violently due to bearing wear.\n\n"
               "Third, customer support and inventory managers work in separate software systems entirely. Support teams only learn about equipment trouble when an angry tenant calls during a 45°C heatwave, while warehouse teams stock replacement capacitors using static buffer rules without checking local weather forecasts.")

    add_p(doc, "When an extreme weather event hits—such as a week of 45°C peak summer heat in Delhi or high-humidity monsoon downpours in Mumbai—this fragmented setup falls apart. High ambient temperatures force air conditioning compressors to run continuously at maximum load. As refrigerant lines heat up and compressor capacitors degrade, dozens of cooling units break down simultaneously across a neighborhood. Because maintenance teams have no predictive warning, warehouses immediately run out of replacement parts. Technicians cannot service broken units in time, tenants get frustrated by days of heat, and many cancel their service subscriptions entirely.")

    add_p(doc, "Our team built this project to fix this disconnect. We frame the entire problem as a Heterogeneous Graph Neural Network (Hetero-GNN). Instead of treating buildings, appliances, residents, and spare parts as isolated rows in separate tables, a graph lets us connect them directly. When extreme weather warms a location, the graph naturally propagates that stress to the connected air conditioners, alerts the warehouse to reserve replacement capacitors, and flags which residents are at risk of churning if their unit fails.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 3: PROBLEM STATEMENT (KRITI)
    # =========================================================================
    add_h1(doc, "Chapter 3: Problem Statement (The 4 Operational Pillars) — Kriti")

    add_p(doc, "To build an effective system, we had to move away from isolated single-task models. A model that only predicts temperature does not help a technician, and a model that only tracks customer churn does not help an inventory manager. We organized the operational requirements into four interconnected pillars:")

    add_h2(doc, "Pillar 1: Regional and Grid-Level Thermal Power Demand")
    add_p(doc, "The system needs to predict the aggregate thermal cooling and heating demand (measured in kW per ton) for every geographic location across twelve calendar months. In India, demand is sharply bimodal: metropolitan urban centers experience severe cooling demand spikes during summer (April to June), while rural and northern village hubs experience heating demand surges during winter (December to January) driven by space heaters and water geysers. The model must capture both regimes without requiring separate models for each region.")

    add_h2(doc, "Pillar 2: Device Health and Predictive Maintenance")
    add_p(doc, "For every individual cooling unit and heating appliance, the system must predict two quantities simultaneously: (1) its continuous operating electrical power draw (kW), and (2) its binary breakdown probability over the next 30 days. When an appliance's operating current climbs while its cooling efficiency drops, the model must flag that unit as high-risk before catastrophic mechanical lockup occurs.")

    add_h2(doc, "Pillar 3: Resident 'Sleeper' Churn Economics")
    add_p(doc, "Subscription churn in HVAC servicing behaves differently from standard SaaS churn. During mild transition months (such as February or October), residents stop opening their climate control app simply because the weather is comfortable. If an energy provider mistakes this seasonal quietness for active customer loss, they waste money sending aggressive marketing messages. Conversely, if a resident stays inactive for over 90 days into the summer season, they are truly 'sleeping' and likely switching to an unmonitored local technician. The system must accurately distinguish seasonal inactivity from real churn, and compute customer lifetime value to prioritize retention offers.")

    add_h2(doc, "Pillar 4: Spare Parts Supply Chain Optimization")
    add_p(doc, "Whenever an air conditioner breaks down, a technician needs specific physical components: run capacitors, fan motors, expansion valves, or inverter PCBs. If a warehouse runs out of stock, mean time to repair jumps from 4 hours to 5 days. Pillar 4 forecasts monthly replenishment demand for 50 distinct spare part SKUs, ensuring parts arrive before seasonal failure waves begin.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 4: THE DATASET & GRAPH SCHEMA (KRITI)
    # =========================================================================
    add_h1(doc, "Chapter 4: The Dataset & Heterogeneous Graph Schema — Kriti")

    add_p(doc, "To connect these four pillars, we constructed a multi-relational heterogeneous graph schema containing four node types and three relational edge types across twelve monthly calendar snapshots.")

    add_h2(doc, "4.1 Node Types and Physical Feature Vectors")
    add_p(doc, "Each node represents a distinct physical or commercial entity, parameterized by realistic engineering features:")

    add_p(doc, "1. Location Nodes (40 nodes, 6 features):\n"
               "Each location represents an urban ward or village cluster. Features include ambient baseline temperature (°C), relative humidity (%), wind speed (m/s), solar irradiance (W/m²), an urban heat island factor (ranging from 0.0 in villages to 1.0 in dense city centers), and total building volume envelope (m³).\n\n"
               "2. Device Nodes (180 nodes, 7 features):\n"
               "Each device represents an installed cooling air conditioner or heating appliance. Features include mechanical age in years, cumulative lifetime operating hours, rolling Coefficient of Performance (COP), vibration velocity (RMS mm/s), motor operating current (A), refrigerant line pressure (bar), and rated nameplate capacity (kW).\n\n"
               "3. Resident Nodes (600 nodes, 5 features):\n"
               "Each resident represents a residential tenant or commercial account. Features include income tier (deciles 1 to 5), household occupancy count, preferred thermostat setpoint (°C), monthly mobile app engagement frequency, and historical billing credit score.\n\n"
               "4. Part Nodes (50 SKUs, 5 features):\n"
               "Each part represents a tracked component in the regional warehouse. Features include unit procurement cost in INR (₹350 to ₹8,500), current on-hand warehouse inventory, supplier replenishment lead time in days, minimum safety reorder threshold, and component Mean Time Between Failures (MTBF in hours).")

    add_h2(doc, "4.2 Relational Edges and Connectivity Topology")
    add_p(doc, "We connect these nodes through three directed relation types:")
    add_p(doc, "• (Location)-[:CONTAINS]->(Device): Links every appliance to the micro-climate envelope of the building it operates in.\n"
               "• (Resident)-[:OWNS]->(Device): Links each customer to their installed appliances, capturing personal usage behavior and thermostat habits.\n"
               "• (Device)-[:REQUIRES]->(Part): Links each appliance model to its required repair components in the warehouse catalog.")

    # Embed Figure 2: Graph Schema
    add_figure_with_caption(doc, "visualization/02_heterogeneous_graph_schema.png",
                            "Figure 2: Heterogeneous Graph Schema showing node attributes and the three relational edge types.",
                            width_inches=6.0)

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 5: DATASET PREPROCESSING & EDA (DHRUV)
    # =========================================================================
    add_h1(doc, "Chapter 5: Dataset Preprocessing & Exploratory Data Analysis — Dhruv")

    add_p(doc, "Raw engineering data collected from IoT sensors and utility databases is rarely clean. Before passing data into the graph neural network, we applied several preprocessing steps to filter sensor errors and normalize features.")

    add_h2(doc, "5.1 Data Cleaning and Thermodynamic Sanitization")
    add_p(doc, "Field telemetry often contains sensor dropouts, transmission glitches, and physical impossibilities. We applied three cleaning rules:")
    add_p(doc, "1. Thermodynamic Clipping: Electrical power sensors occasionally report negative spikes during grid surges or values exceeding hardware ratings. We clipped power draws to the physical nameplate bounds of the equipment (0.0 kW to 3.5 kW) and removed unrealistic ambient temperature readings outside the -5°C to 50°C range.\n\n"
               "2. Spatial Imputation for Missing Telemetry: When an outdoor temperature or humidity sensor dropped offline, we imputed its values using the distance-weighted average of neighboring location nodes rather than naive column means.\n\n"
               "3. Graph Referential Integrity: We verified that every device node connected to an existing location, resident, and replacement part SKU. Any orphaned node references were pruned to prevent graph index mismatches.")

    add_h2(doc, "5.2 Cyclical Temporal Encoding")
    add_p(doc, "Because seasons repeat on a 12-month calendar loop, passing the calendar month as a linear integer (1 to 12) misleads the network into thinking January (month 1) and December (month 12) are maximally distant. We transformed calendar months into two continuous cyclical coordinates:")

    add_callout(doc, [
        "theta_m = 2 * pi * (m - 1) / 12",
        "Month_Sin = sin(theta_m)",
        "Month_Cos = cos(theta_m)",
        "This ensures December and January are adjacent in feature space, preserving smooth seasonal transitions."
    ], title="CYCLICAL MONTH ENCODING")

    add_h2(doc, "5.3 Exploratory Data Analysis Across 12 Climate Regimes")
    add_p(doc, "Analyzing the processed telemetry across twelve Indian climate regimes highlighted three major operational patterns:")

    add_p(doc, "1. Severe Failure Class Imbalance: Breakdown events represent fewer than 8% of all monthly device observations (averaging 2.07% to 7.8% across seasons). Standard classifiers trained on this distribution collapse into predicting zero failures for every machine.\n\n"
               "2. Divergent Regional Demands: Metropolitan wards experience intense cooling peaks in May (624 demand units) when ambient temperatures reach 44°C. In contrast, rural village hubs peak in December (722 demand units) driven by electric geysers and space heaters.\n\n"
               "3. High-Humidity Monsoon Stress: During July and August, relative humidity stays above 88%. While ambient temperatures moderate to 31°C, moisture condensation accelerates electrical insulation degradation and causes inverter board shorts.")

    # Embed Figure 3: Climate Regimes
    add_figure_with_caption(doc, "visualization/09_seasonal_cooling_heating_regimes.png",
                            "Figure 3: 12-Month Climate Regimes and Demand Characteristics across Metropolitan and Rural Village hubs.",
                            width_inches=6.0)

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 6: FEATURE ENGINEERING AND SELECTION (ANKIT)
    # =========================================================================
    add_h1(doc, "Chapter 6: Feature Engineering and Selection — Ankit")

    add_p(doc, "Raw sensor readings like voltage and temperature are not enough on their own to predict subtle machine degradation or resident churn. We engineered domain-specific features that capture the physical thermodynamics of HVAC equipment and customer engagement dynamics.")

    add_h2(doc, "6.1 Physical Thermodynamics and Stress Features")
    add_p(doc, "We constructed three primary mechanical features:")

    add_p(doc, "1. Rolling Coefficient of Performance (COP):\n"
               "COP measures the ratio of useful heating or cooling delivered to the electrical energy consumed: COP = Q_thermal / P_electrical. For a healthy air conditioner, COP ranges between 2.8 and 3.8. When heat exchanger coils become dirty or refrigerant leaks, the compressor must work harder to deliver the same cooling, causing COP to drop below 2.0. We computed a 3-month rolling COP trend; a steady negative slope provides early warning of equipment wear weeks before failure.\n\n"
               "2. Cumulative Mechanical Vibration Stress:\n"
               "Motor bearing wear produces high-frequency physical vibrations. We calculated an accumulated stress index by scaling RMS vibration velocity (mm/s) by total operational run hours. Machines exceeding 4.5 mm/s RMS vibration under sustained load are prioritized for preventative bearing replacement.\n\n"
               "3. Urban Micro-Climate Heat Index:\n"
               "Due to concrete density and asphalt heat retention, urban city centers remain 2.5°C warmer at night than surrounding rural areas. We combined ambient temperature, humidity, and population density into an effective thermal stress index that adjusts cooling load predictions for urban heat islands.")

    add_h2(doc, "6.2 Customer Behavioral and Supply Chain Features")
    add_p(doc, "On the customer and supply chain side, we engineered:")

    add_p(doc, "1. Recency, Frequency, and Monetary (RFM) Scores:\n"
               "From historical transaction logs, we computed: (R) days since last service booking, (F) number of completed service requests, and (M) total annual maintenance spend. Customers with R > 90 days during shoulder seasons are tagged as 'Sleepers'.\n\n"
               "2. Warehouse Stockout Vulnerability Index:\n"
               "For each replacement part SKU, we computed the ratio of expected lead-time demand to current warehouse inventory: Stockout_Risk = (Lead_Time * Demand_Velocity) / Current_Stock. SKUs with a ratio greater than 1.0 face imminent stockout if another breakdown occurs.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 7: MODEL SELECTION (ANKIT)
    # =========================================================================
    add_h1(doc, "Chapter 7: Model Selection & Relational Architecture — Ankit")

    add_p(doc, "Before committing to a Graph Neural Network, we evaluated standard machine learning baselines including Gradient Boosted Decision Trees (XGBoost), Random Forests, and Multi-Layer Perceptrons.")

    add_h2(doc, "7.1 Why Standard Tabular Models Failed")
    add_p(doc, "Tabular algorithms treat every sample as an independent and identically distributed (i.i.d.) observation. When tabular models attempt to predict whether an air conditioner will fail, they only inspect that single air conditioner's row. They cannot see that four adjacent cooling units in the same building are already drawing excessive current, or that the local building transformer is experiencing voltage sags. Tabular models also cannot propagate information across entities—for example, knowing that an incoming heatwave at a location node will increase device power draws, which in turn will exhaust specific capacitor inventory at the connected part node.")

    add_h2(doc, "7.2 Relational Graph Convolutional Networks (RGCN)")
    add_p(doc, "To capture these interdependencies, we selected a Relational Graph Convolutional Network (RGCN) implemented via PyTorch Geometric's HeteroConv module. In an RGCN, message passing uses distinct weight transformation matrices W_r for each relational edge type:")

    add_callout(doc, [
        "h_i^(l+1) = LeakyReLU( W_0^(l) h_i^(l) + sum_{r in Relations} sum_{j in Neighbors(r)} (1 / |N_i^r|) * W_r^(l) h_j^(l) )",
        "where W_0 preserves the node's own feature identity, while W_r transforms incoming messages from neighbor j based on relation r (contains, owns, or requires)."
    ], title="EQUATION: RELATIONAL MESSAGE PASSING (RGCN)")

    add_p(doc, "Following two layers of relational message passing (hidden dimensions: 64 and 32, with LayerNorm and Dropout p = 0.20), the latent node embeddings are routed to five task-specific Multi-Layer Perceptron (MLP) prediction heads:")
    add_p(doc, "• Head 1 (Demand): Continuous regression predicting regional thermal demand (kW/ton).\n"
               "• Head 2 (Power): Continuous regression predicting device electrical draw (kW).\n"
               "• Head 3 (Failure): Binary classification predicting 30-day component breakdown probability.\n"
               "• Head 4 (Churn): Binary classification predicting tenant sleeper churn probability.\n"
               "• Head 5 (Inventory): Continuous regression predicting required spare parts replenishment velocity.")

    # Embed Figure 4: Architecture
    add_figure_with_caption(doc, "visualization/03_hetero_gnn_model_architecture.png",
                            "Figure 4: End-to-end MultiTaskHVACHeteroGNN architecture, showing relational convolution layers and decoupled task heads.",
                            width_inches=6.0)

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 8: CORE LIFECYCLE EXECUTION (ANIMESH & SAURAV)
    # =========================================================================
    add_h1(doc, "Chapter 8: Core Lifecycle Execution (Model Training, Evaluation, & Deployment)")

    add_p(doc, "This chapter covers the computational implementation, fine-tuning calibration, production serving, and continuous monitoring of our pipeline. Sections 8.1 and 8.2 detail the training harness and fine-tuning innovations led by Animesh Basak, while Sections 8.3 and 8.4 detail the interactive deployment and drift monitoring led by Saurav.")

    # -------------------------------------------------------------------------
    # 8.1 Model Training (Animesh)
    # -------------------------------------------------------------------------
    add_h2(doc, "8.1 Model Training on NVIDIA Tesla V100 GPU — Animesh Basak")

    add_p(doc, "Training a multi-task heterogeneous graph over multi-modal physics and business data requires significant computational throughput. We deployed our training harness on a dedicated enterprise GPU server:")
    add_p(doc, "• Hardware: NVIDIA Tesla V100S-PCIE-32GB GPU Server (IP: 10.13.1.7)\n"
               "• Specifications: 5,120 CUDA Cores, 640 Tensor Cores, 32 GB HBM2 VRAM (1,134 GB/s memory bandwidth)\n"
               "• Software Stack: Ubuntu 22.04 LTS, CUDA 13.0, PyTorch 2.6.0+cu124, PyTorch Geometric 2.8.1\n"
               "• Memory Optimization: Automatic Mixed Precision (AMP via torch.cuda.amp.autocast and GradScaler)\n"
               "• Peak VRAM Footprint: 28.8 MB (achieved by using sparse matrix PyG message-passing kernels)")

    add_h3(doc, "Self-Healing Numerical Divergence Protection")
    add_p(doc, "During unattended training runs, sudden numerical spikes in power telemetry (such as a 45°C ambient heatwave combined with a near-zero standby current) can cause gradient explosions, resulting in NaN or Inf loss values. Rather than allowing training to crash, Animesh authored an automated Self-Healing Training Harness directly into the optimization loop:")

    add_callout(doc, [
        "ALGORITHM: SELF-HEALING NUMERICAL DIVERGENCE PROTECTION",
        "1. Compute batch loss L_t via forward pass.",
        "2. If isnan(L_t) or isinf(L_t) or L_t > 10.0 * Running_Average_Loss:",
        "     a. Log divergence alert: '[ALERT] Gradient explosion detected at Epoch t'.",
        "     b. Immediately revert model parameters to the last known stable checkpoint.",
        "     c. Halve the learning rate: lr = lr * 0.5.",
        "     d. Reinitialize AdamW optimizer momentum and variance buffers.",
        "     e. Flush CUDA cache: torch.cuda.empty_cache().",
        "     f. Advance epoch without updating corrupted gradients.",
        "3. Else: Execute scaled backward pass, clip gradient norm to 1.0, and step optimizer."
    ], title="SELF-HEALING TRAINING SAFEGUARD")

    # Embed Figure 5: Self Healing Flowchart
    add_figure_with_caption(doc, "visualization/05_self_healing_training_flowchart.png",
                            "Figure 5: Algorithmic flowchart of the Self-Healing Numerical Divergence Protection harness.",
                            width_inches=5.6)

    add_p(doc, "During the initial pre-training phase, the network was trained over 120 epochs using an unweighted composite loss function. Training loss dropped rapidly from an initial composite loss of ~245.0 before stabilizing at a plateau of 137.04 (saved to disk as artifacts/best_hetero_gnn_v100.pt). While functional, our post-training audit revealed critical deficiencies.")

    # -------------------------------------------------------------------------
    # 8.2 Model Evaluation & Multi-Task Fine-Tuning (Animesh)
    # -------------------------------------------------------------------------
    add_h2(doc, "8.2 Model Evaluation & Multi-Task Fine-Tuning — Animesh Basak")

    add_p(doc, "An independent technical audit of the pre-trained model identified three major performance bottlenecks:")

    add_p(doc, "1. Multi-Task Gradient Dominance:\n"
               "Unweighted squared error losses for continuous tasks produced massive numerical gradients (Demand MSE was around 115, Inventory MSE around 20). In contrast, binary cross-entropy losses for component failure and sleeper churn were small (0.50 to 0.70). As a result, the optimizer focused almost entirely on tiny fluctuations in demand while ignoring critical component breakdowns.\n\n"
               "2. Non-Peak Regime Under-Reporting:\n"
               "In temperate and monsoon months (February, March, July to November), the pre-trained model flagged exactly zero high-risk equipment units. Because the baseline was overwhelmed by summer heat loads, it completely missed moisture-driven PCB corrosion during the rainy season.\n\n"
               "3. Late-Epoch Loss Oscillations:\n"
               "Unconstrained squared errors (MSE) amplified summer heatwave outlier readings, kicking gradients violently and causing validation loss to oscillate between epochs 40 and 120.")

    add_h3(doc, "Mathematical Calibrations: Kendall Uncertainty Loss and Huber Loss")
    add_p(doc, "To resolve these audit bottlenecks without arbitrary trial-and-error hyperparameter tuning, Animesh formulated and integrated two mathematical improvements:")

    add_p(doc, "First, we implemented Kendall Homoscedastic Uncertainty Weighting (Kendall et al., CVPR 2018). We treat each task head as having an inherent observation noise parameter sigma_k. Reparameterizing in log-variance space s_k = log(sigma_k^2) to guarantee numerical stability yields:")

    add_callout(doc, [
        "L_uncertainty(W, s) = sum_{k=1}^5 [ 0.5 * exp(-s_k) * L_k + 0.5 * s_k ]",
        "where s_k is a learnable parameter updated via backpropagation. When a task has high error L_k, the network automatically increases s_k, dampening that task's gradient impact and preventing it from drowning out the other heads."
    ], title="KENDALL HOMOSCEDASTIC MULTI-TASK LOSS")

    add_p(doc, "Second, we replaced Mean Squared Error on all continuous regression heads with SmoothL1 (Huber) Loss using a transition threshold of beta = 4.0:")

    add_callout(doc, [
        "SmoothL1(y, y_hat) = 0.5 * (y - y_hat)^2 / beta,    if |y - y_hat| < beta",
        "SmoothL1(y, y_hat) = |y - y_hat| - 0.5 * beta,      otherwise",
        "For small errors, the penalty is quadratic (L2) ensuring smooth convergence. For large outliers (like extreme 45°C heatwaves), the penalty switches to linear (L1) with a constant gradient of 1.0, eliminating gradient explosions."
    ], title="SMOOTHL1 (HUBER) ROBUST REGRESSION LOSS")

    add_h3(doc, "Fine-Tuning on Real Benchmark Datasets")
    add_p(doc, "We fine-tuned the model (fine_tune_v100.py) by ingesting three real-world industrial and commercial benchmark datasets:")
    add_p(doc, "• UCI AI4I 2020 Predictive Maintenance (dataset/601): 10,000 industrial machine records used to calibrate Heat Dissipation Failure (HDF) and Power Failure (PWF) fault rates.\n"
               "• UCI Online Retail II (dataset/502): 525,468 transactions aggregated into 4,312 customer RFM profiles to calibrate sleeper churn labels (Recency > 90 days).\n"
               "• Kaggle HVAC Power Consumption (felixpradoh): 8,760 hourly thermal telemetry curves used to calibrate continuous electrical power draws across all twelve months.")

    add_p(doc, "Executing this fine-tuning on the Tesla V100 GPU took exactly 17.0 seconds across 60 calibrated epochs (peak VRAM: 28.8 MB). Overall dossier loss dropped from 137.04 to 46.80 (-65.85% reduction), saved as artifacts/fine_tuned_hetero_gnn_v100.pt.")

    # Embed Figure 6 & 7: Loss Trajectory & Metrics
    add_figure_with_caption(doc, "visualization/06_pretrain_vs_finetune_loss_trajectory.png",
                            "Figure 6: Loss convergence curves showing pre-training plateau (137.04) and rapid V100 fine-tuning convergence to 46.80.",
                            width_inches=6.0)

    add_figure_with_caption(doc, "visualization/07_pillar_performance_comparison_before_after.png",
                            "Figure 7: Quantitative performance comparison before vs. after fine-tuning across all four operational pillars.",
                            width_inches=6.0)

    # -------------------------------------------------------------------------
    # 8.3 Model Deployment (Saurav)
    # -------------------------------------------------------------------------
    add_h2(doc, "8.3 Model Deployment & Interactive Dashboard — Saurav")

    add_p(doc, "To make the fine-tuned graph model useful to facility managers, technicians, and operations executives, Saurav deployed the system across two tiers:")

    add_p(doc, "1. Live Streamlit Cloud Command Center:\n"
               "Deployed at https://smarthvac-advisor.streamlit.app/ with continuous GitHub integration. To eliminate blank-screen crashes caused by temporary file permission restrictions on cloud containers, Saurav engineered an in-memory PyVis graph renderer (net.generate_html()). The interface is organized into five focused operational tabs:\n"
               "  • Tab 1: 🌐 Interactive Hetero-Graph (visualizes spatial clusters and component edges)\n"
               "  • Tab 2: ⚡ Device Failure Telemetry (operating load vs. failure risk scatter plots)\n"
               "  • Tab 3: 👥 Sleeper Retention Engine (customer RFM deciles and retention campaign lists)\n"
               "  • Tab 4: 📦 Spare Parts Forecasting (warehouse stockout risks and reorder timelines)\n"
               "  • Tab 5: 💰 Executive Financial Ledger (audited monthly P&L and cost-efficiency ratios)\n\n"
               "2. Standalone Offline Browser Dashboard (ARGOS_BROWSER_RESULTS.html):\n"
               "A self-contained HTML5 and Chart.js dashboard that requires no server, Python runtime, or internet connection. Users can double-click the file in any browser to inspect interactive revenue curves, model loss decays, and full before-vs-after ledgers.")

    # Embed Figure 9: System Pipeline
    add_figure_with_caption(doc, "visualization/04_end_to_end_system_pipeline_flowchart.png",
                            "Figure 8: End-to-end system architecture from multi-modal data ingestion to production dashboard serving.",
                            width_inches=6.0)

    # -------------------------------------------------------------------------
    # 8.4 Model Monitoring & Drift Maintenance (Saurav)
    # -------------------------------------------------------------------------
    add_h2(doc, "8.4 Model Monitoring & Maintenance — Saurav")

    add_p(doc, "Machine learning models operating in cyber-physical environments face performance degradation over time due to weather shifts and hardware aging. Saurav implemented monitoring protocols for two distinct forms of drift:")

    add_p(doc, "1. Covariate (Feature) Climate Drift:\n"
               "Tracks statistical distribution changes in outdoor weather telemetry. If a 14-day rolling window exhibits temperatures or humidity significantly outside historical distributions (Kolmogorov-Smirnov test p < 0.01 or Population Stability Index PSI > 0.25), an operational alert is triggered.\n\n"
               "2. Concept Appliance Wear Drift:\n"
               "Tracks changes in the underlying physical relationship between electrical power and cooling output. As compressors age, their electrical draw rises while COP falls. If power prediction MSE exceeds 0.20 kW or sleeper churn F1 drops below 0.55, the pipeline automatically triggers incremental fine-tuning on the Tesla V100 GPU using the latest 60-day telemetry window.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 9: RESULTS AND KEY FINDINGS
    # =========================================================================
    add_h1(doc, "Chapter 9: Results and Key Findings")

    add_p(doc, "We evaluated the calibrated model across all five prediction heads and simulated a complete 12-month calendar cycle to audit its real-world business viability.")

    add_h2(doc, "9.1 Quantitative Performance Improvements")
    add_p(doc, "Table 2 summarizes the quantitative results before and after fine-tuning:")

    res_table_data = [
        ("Composite Dossier Loss", "137.04", "46.80", "-65.85%", "Overall multi-task error reduction"),
        ("Pillar 1: Demand MSE (kW/ton)", "115.61", "33.13", "-71.34%", "MAE dropped from 8.77 to 4.64"),
        ("Pillar 2: Device Power MSE (kW)", "0.8364", "0.0654", "-92.18%", "MAE dropped from 0.749 to 0.209 kW"),
        ("Pillar 2: Equipment Failure F1", "0.540", "0.697", "+29.07%", "432 high-risk units flagged proactively"),
        ("Pillar 3: Sleeper Churn F1", "0.397", "0.646", "+62.72%", "278 sleepers identified, 276 reactivated"),
        ("Pillar 4: Inventory Demand MSE", "19.13", "12.61", "-34.11%", "MAE dropped from 3.45 to 2.78 units"),
        ("Annual Gross Revenue", "₹2,40,22,953.31", "₹2,61,01,851.84", "+8.65%", "+₹20.78 Lakhs via sleeper campaigns"),
        ("Annual Net Operating Profit", "₹2,09,55,941.58", "₹2,27,01,359.44", "+8.33%", "+₹17.45 Lakhs profit gain"),
        ("Mean Cost-Efficiency Ratio", "7.83x (volatile)", "7.764x (calibrated)", "Stable", "Aggregate annual ratio: 7.676x"),
        ("Proactive Cost Savings", "Baseline reactive", "+22.64% average", "Peak +28.4%", "Savings vs. emergency repair overtime")
    ]

    tbl_r = doc.add_table(rows=len(res_table_data) + 1, cols=5)
    tbl_r.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_r)
    col_w_r = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.4)]
    headers_r = ["Metric / Task Head", "Before Fine-Tune", "After Fine-Tune", "Improvement", "Operational Impact"]

    for c_idx, h_text in enumerate(headers_r):
        cell = tbl_r.rows[0].cells[c_idx]
        cell.width = col_w_r[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=90, bottom=90, left=90, right=90)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [1, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=9.5, bold=True, color_rgb=(255, 255, 255))

    for r_idx, row_vals in enumerate(res_table_data):
        row = tbl_r.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = col_w_r[c_idx]
            bg = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [1, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            format_run(r, font_name="Times New Roman", size_pt=9, bold=(c_idx in [2, 3]), color_rgb=(0,0,0))

    add_p(doc, "", space_after=10)

    # Embed Figure 8: ROC Curves
    add_figure_with_caption(doc, "visualization/10_failure_and_churn_roc_curves.png",
                            "Figure 9: ROC curves demonstrating calibrated classification performance for equipment failure (AUC = 0.882) and customer churn (AUC = 0.843).",
                            width_inches=6.0)

    add_h2(doc, "9.2 Audited 12-Month Financial Ledger")
    add_p(doc, "To verify commercial viability, we audited monthly revenue, operating expenses, and net profit across all twelve calendar regimes. Table 3 presents the audited monthly ledger:")

    ledger_rows = [
        ("01", "Winter", "36", "21", "22,41,510.40", "2,36,880.12", "20,04,630.28", "9.463x", "28.4%"),
        ("02", "Winter-Trans.", "24", "19", "20,95,440.18", "2,49,112.50", "18,46,327.68", "8.412x", "18.2%"),
        ("03", "Spring-Warm", "28", "18", "22,14,330.65", "3,02,440.80", "19,11,889.85", "7.322x", "16.5%"),
        ("04", "Peak-Summer", "45", "26", "27,82,190.22", "3,61,890.40", "24,20,299.82", "7.688x", "22.8%"),
        ("05", "Peak-Summer", "52", "31", "29,01,450.90", "3,84,210.15", "25,17,240.75", "7.552x", "26.1%"),
        ("06", "Late-Summer", "41", "28", "25,48,720.50", "3,42,670.30", "22,06,050.20", "7.438x", "19.4%"),
        ("07", "Monsoon-Humid", "38", "24", "19,18,650.15", "2,93,450.60", "16,25,199.55", "6.538x", "23.6%"),
        ("08", "Monsoon-Humid", "39", "22", "18,44,280.40", "2,80,620.25", "15,63,660.15", "6.572x", "24.1%"),
        ("09", "Post-Monsoon", "32", "20", "18,05,420.20", "2,67,110.10", "15,38,310.10", "6.759x", "21.9%"),
        ("10", "Autumn-Trans.", "26", "19", "16,72,990.80", "2,40,940.70", "14,32,050.10", "6.944x", "17.8%"),
        ("11", "Early-Winter", "34", "23", "19,98,410.64", "2,20,490.50", "17,77,920.14", "9.063x", "25.3%"),
        ("12", "Peak-Winter", "37", "25", "20,78,456.80", "2,20,675.98", "18,57,780.82", "9.419x", "27.6%")
    ]

    tbl_l = doc.add_table(rows=len(ledger_rows) + 2, cols=9)
    tbl_l.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_l)
    col_w_l = [Inches(0.5), Inches(1.1), Inches(0.5), Inches(0.5), Inches(1.0), Inches(0.9), Inches(1.0), Inches(0.5), Inches(0.5)]
    headers_l = ["Mth", "Regime", "Risk", "Slpr", "Revenue (INR)", "OPEX (INR)", "Net Profit (INR)", "Ratio", "Sav%"]

    for c_idx, h_text in enumerate(headers_l):
        cell = tbl_l.rows[0].cells[c_idx]
        cell.width = col_w_l[c_idx]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=70, bottom=70, left=40, right=40)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        format_run(r, font_name="Times New Roman", size_pt=8, bold=True, color_rgb=(255, 255, 255))

    for r_idx, row_vals in enumerate(ledger_rows):
        row = tbl_l.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = col_w_l[c_idx]
            bg = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=50, bottom=50, left=40, right=40)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx not in [1, 4, 5, 6] else (WD_ALIGN_PARAGRAPH.RIGHT if c_idx in [4, 5, 6] else WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(val)
            format_run(r, font_name="Times New Roman", size_pt=7.5, color_rgb=(0,0,0))

    # Add Totals Row
    row_tot = tbl_l.rows[-1]
    tot_vals = ["Tot", "Annual Sum", "432", "276", "2,61,01,851.84", "34,00,492.40", "2,27,01,359.44", "7.764x", "22.6%"]
    for c_idx, val in enumerate(tot_vals):
        cell = row_tot.cells[c_idx]
        cell.width = col_w_l[c_idx]
        set_cell_background(cell, "D9E1F2")
        set_cell_margins(cell, top=60, bottom=60, left=40, right=40)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx not in [1, 4, 5, 6] else (WD_ALIGN_PARAGRAPH.RIGHT if c_idx in [4, 5, 6] else WD_ALIGN_PARAGRAPH.LEFT)
        r = p.add_run(val)
        format_run(r, font_name="Times New Roman", size_pt=8, bold=True, color_rgb=(15, 34, 64))

    add_p(doc, "", space_after=8)

    # Embed Figure 10: Financial Ledger Chart
    add_figure_with_caption(doc, "visualization/08_12month_audited_financial_ledger.png",
                            "Figure 10: Audited 12-Month Financial Ledger showing monthly revenue, OPEX, and cost-efficiency ratios.",
                            width_inches=6.0)

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 10: CHALLENGES AND FUTURE WORK
    # =========================================================================
    add_h1(doc, "Chapter 10: Challenges and Future Work")

    add_p(doc, "Building and deploying this multi-task heterogeneous graph model presented several practical engineering hurdles:")

    add_h2(doc, "10.1 Key Challenges Encountered and Overcome")
    add_p(doc, "1. Balancing Conflicting Gradient Updates:\n"
               "Early in training, we ran into severe multi-task gradient interference. The unweighted demand regression head generated massive squared-error gradients that completely overpowered the binary failure classification head. Applying Kendall's Bayesian uncertainty weighting allowed the model to balance continuous regression and classification heads dynamically without manual hyperparameter search.\n\n"
               "2. Thermal Outlier Instability:\n"
               "During peak summer heatwaves, ambient temperatures reach 45°C. When a single prediction had a large residual, standard Mean Squared Error squared that error, kicking gradients violently and destabilizing model weights. Replacing MSE with SmoothL1 (Huber) loss using a threshold of beta = 4.0 bounded the gradients to a constant 1.0, restoring training stability.\n\n"
               "3. Cloud File Permission Restrictions:\n"
               "When we first deployed to Streamlit Cloud, the app crashed with a blank screen because PyVis attempted to write temporary HTML graph files to a restricted local filesystem. Rewriting the graph visualization pipeline to generate HTML entirely in memory resolved the deployment crash.")

    add_h2(doc, "10.2 Future Research and Engineering Directions")
    add_p(doc, "1. Continuous-Time Dynamic Graphs (CTDG):\n"
               "Our current implementation constructs twelve discrete monthly graph snapshots. In future work, we plan to implement continuous-time dynamic graph neural networks using temporal graph attention (TGAT) to update edge weights continuously as real-time sub-second telemetry streams arrive from smart thermostats.\n\n"
               "2. Federated Privacy-Preserving Learning:\n"
               "Smart thermostat setpoints and billing histories represent private customer information. Deploying Federated GNN edge gateways on local building controllers would allow models to learn shared equipment degradation patterns across buildings without transmitting private raw resident telemetry off-site.")

    # =========================================================================
    # CHAPTER 11: CONCLUSION
    # =========================================================================
    add_h1(doc, "Chapter 11: Conclusion")

    add_p(doc, "In this project, our team designed, trained, evaluated, and deployed an inductive Multi-Task Heterogeneous Graph Neural Network for smart HVAC energy forecasting, predictive maintenance, customer churn prevention, and inventory supply chain management.")

    add_p(doc, "By replacing isolated single-building thermodynamic equations with an integrated graph topology, the model unifies physical equipment sensor feeds with commercial customer economics. Training on an NVIDIA Tesla V100 GPU with automatic mixed precision achieved an overall loss reduction of -65.85% (137.04 to 46.80), improved component failure prediction F1 to 0.697 (+29.07%), and increased customer sleeper churn detection F1 to 0.646 (+62.72%).")

    add_p(doc, "Audited across a complete 12-month calendar cycle, the system delivered ₹2.61 Crore in gross revenue, ₹34.00 Lakhs in operating expenditure, and ₹2.27 Crore in net operating profit, maintaining positive cash flow in all twelve months with an average cost-efficiency ratio of 7.764x and +22.64% proactive maintenance cost savings over reactive emergency repairs.")

    add_p(doc, "The complete codebase, trained model checkpoints, interactive Streamlit dashboard, and documentation are publicly available on GitHub: https://github.com/AnimeshBasak-14/SmartHVAC, with live cloud serving at: https://smarthvac-advisor.streamlit.app/.")

    # Save Document
    out_docx = "docs/New folder/SmartHVAC_Full_Project_Report.docx"
    doc.save(out_docx)
    print(f"Full Project Report generated successfully: {out_docx}")

if __name__ == "__main__":
    build_full_report()
