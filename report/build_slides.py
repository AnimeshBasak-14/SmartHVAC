"""
build_slides.py - builds report/HVAC_Service_Planner_Presentation.pptx (16:9) from the report figures and result tables.

    python report/build_slides.py
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "hvac_ml" / "outputs" / "figures"
RES = ROOT / "hvac_ml" / "outputs" / "results"
OUT = ROOT / "report" / "HVAC_Service_Planner_Presentation.pptx"

NAVY, GREY = RGBColor(0x1F, 0x4E, 0x79), RGBColor(0x66, 0x66, 0x66)
PEOPLE = {
    1: ("Kriti Vajpayee", RGBColor(0x8E, 0x44, 0xAD)), 2: ("Kriti Vajpayee", RGBColor(0x8E, 0x44, 0xAD)),
    3: ("Dhruv Aggarwal", RGBColor(0x2E, 0x86, 0xAB)), 4: ("Dhruv Aggarwal", RGBColor(0x2E, 0x86, 0xAB)),
    5: ("Ankit Kumar", RGBColor(0x3A, 0x9D, 0x5D)), 6: ("Ankit Kumar", RGBColor(0x3A, 0x9D, 0x5D)),
    7: ("Animesh Basak", RGBColor(0xE0, 0x7A, 0x1F)), 8: ("Animesh Basak", RGBColor(0xE0, 0x7A, 0x1F)),
    9: ("Sourav Kumar Boro", RGBColor(0xC0, 0x39, 0x2B)), 10: ("Sourav Kumar Boro", RGBColor(0xC0, 0x39, 0x2B)),
    0: ("Whole group", NAVY),
}
STAGE_NAMES = {0: "Summary", 1: "Problem definition", 2: "Data collection", 3: "Data cleaning", 4: "Data exploration", 5: "Feature engineering",
               6: "Model selection", 7: "Model training", 8: "Evaluation and tuning", 9: "Deployment", 10: "Monitoring"}
SUMMARY = pd.read_csv(RES / "summary_mean_std.csv")
SP3 = "unseen_customers_later_period"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
slide_no = [0]


def txt(slide, x, y, w, h, text, size=14, bold=False, color=RGBColor(0x22, 0x22, 0x22), align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.italic = Pt(size), bold, italic
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return tb


def base(title, stage, notes=""):
    s = prs.slides.add_slide(BLANK)
    slide_no[0] += 1
    name, color = PEOPLE[stage]
    txt(s, 0.4, 0.2, 10.3, 0.75, title, 26 if len(title) <= 52 else 23 if len(title) <= 62 else 20, True, NAVY, anchor=MSO_ANCHOR.MIDDLE)
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(0.95), Inches(12.45), Inches(0.05))
    line.fill.solid(); line.fill.fore_color.rgb = color; line.line.fill.background()
    chip = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.75), Inches(0.28), Inches(2.2), Inches(0.58))
    chip.fill.solid(); chip.fill.fore_color.rgb = color; chip.line.fill.background()
    tf = chip.text_frame
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = (f"Stage {stage}: " if stage else "") + STAGE_NAMES[stage]
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(11), True, RGBColor(255, 255, 255), "Calibri"
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = name
    r2.font.size, r2.font.color.rgb, r2.font.name = Pt(10), RGBColor(255, 255, 255), "Calibri"
    txt(s, 0.4, 7.12, 8, 0.3, "HVAC Service Planner  |  Group 1  |  CSC-505 Machine Learning, IIT Roorkee", 9, False, GREY)
    txt(s, 12.0, 7.12, 0.95, 0.3, str(slide_no[0]), 10, False, GREY, PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def find(name):
    for d in (FIG / "report", FIG):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def picture(slide, name, box):
    x, y, w, h = box
    path = find(name)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def image_slide(title, stage, names, notes="", caption=None):
    s = base(title, stage, notes)
    names = names if isinstance(names, list) else [names]
    top, bottom = 1.1, 7.05 - (0.4 if caption else 0)
    if len(names) == 1:
        with Image.open(find(names[0])) as im:
            aspect = im.size[0] / im.size[1]
        ph = 12.63 / aspect
        sentences = [t.strip() + "." for t in notes.replace("..", ".").split(". ") if t.strip()]
        sentences = [t.rstrip(".") + "." for t in sentences][:3]
        room = bottom - top - ph
        if room >= 1.2 and sentences and not caption:
            picture(s, names[0], (0.35, top, 12.63, ph))
            size = 18 if room >= 2.3 else 16
            y = top + ph + 0.2
            for sent in sentences:
                if y > 6.6:
                    break
                dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(y + 0.1), Inches(0.14), Inches(0.14))
                dot.fill.solid(); dot.fill.fore_color.rgb = PEOPLE[stage][1]; dot.line.fill.background()
                tb = txt(s, 0.85, y - 0.02, 12.0, 0.9, sent, size)
                y += 0.46 * (1 + len(sent) // 120) + (0.12 if size == 18 else 0.05)
        else:
            picture(s, names[0], (0.35, top, 12.63, bottom - top))
    else:
        h = (bottom - top) / len(names)
        for i, n in enumerate(names):
            picture(s, n, (0.35, top + i * h, 12.63, h - 0.05))
    if caption:
        txt(s, 0.45, 6.72, 12.4, 0.4, caption, 13, False, RGBColor(0x33, 0x33, 0x33), italic=True)
    return s


def table_slide(title, stage, header, rows, col_w=None, font=13, notes="", first_bold=True, note_line=None, row_h=0.42, zebra=True, center_from=1):
    s = base(title, stage, notes)
    n_rows, n_cols = len(rows) + 1, len(header)
    avail_w = 12.5
    col_w = col_w or [1.0] * n_cols
    tot = sum(col_w)
    col_w = [c * avail_w / tot for c in col_w]
    top = 1.2
    shape = s.shapes.add_table(n_rows, n_cols, Inches(0.4), Inches(top), Inches(avail_w), Inches(row_h * n_rows))
    tbl = shape.table
    tblPr = shape._element.graphic.graphicData.tbl.tblPr
    tblPr.set("bandRow", "0"); tblPr.set("firstRow", "0")
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    for i in range(n_rows):
        tbl.rows[i].height = Inches(row_h)
        for j in range(n_cols):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            text = header[j] if i == 0 else str(rows[i - 1][j])
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            if j >= center_from and len(text) < 22:
                p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = text
            r.font.size = Pt(font); r.font.name = "Calibri"
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = NAVY
                r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                cell.fill.fore_color.rgb = RGBColor(0xF2, 0xF5, 0xF8) if (zebra and i % 2 == 0) else RGBColor(255, 255, 255)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
                r.font.bold = bool(first_bold and j == 0)
    if note_line:
        txt(s, 0.45, 6.68, 12.4, 0.4, note_line, 12.5, False, RGBColor(0x33, 0x33, 0x33), italic=True)
    return s


def sv(p, split, model, col, digits=3, std=True):
    r = SUMMARY[(SUMMARY["pillar"] == p) & (SUMMARY["split"] == split) & (SUMMARY["model"] == model)].iloc[0]
    m, sd = r[f"{col}_mean"], r[f"{col}_std"]
    return f"{m:.{digits}f} ± {sd:.{digits}f}" if std and pd.notna(sd) else f"{m:.{digits}f}"


E, B = " (engineered features)", " (basic features)"

# ============================================================================ 1. title
s = prs.slides.add_slide(BLANK)
slide_no[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(3.5))
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
txt(s, 0.7, 0.55, 12, 0.5, "Indian Institute of Technology Roorkee  |  Department of Computer Science and Engineering", 16, False, RGBColor(0xDD, 0xE6, 0xF0))
txt(s, 0.7, 1.15, 12, 1.0, "HVAC Service Planner", 52, True, RGBColor(255, 255, 255))
txt(s, 0.7, 2.25, 12, 0.6, "Using machine learning for HVAC system providers such as Aire Serv", 24, False, RGBColor(255, 255, 255), italic=True)
txt(s, 0.7, 2.85, 12, 0.5, "Demand forecasting, failure prediction and customer retention", 16, False, RGBColor(0xDD, 0xE6, 0xF0))
txt(s, 0.7, 3.8, 6, 0.4, "CSC-505 Machine Learning  |  Group 1", 18, True, NAVY)
rows = [("Kriti Vajpayee", "25931008", "Stages 1 and 2"), ("Dhruv Aggarwal", "25911019", "Stages 3 and 4"), ("Ankit Kumar", "26911002", "Stages 5 and 6"),
        ("Animesh Basak", "26911001", "Stages 7 and 8"), ("Sourav Kumar Boro", "26535026", "Stages 9 and 10")]
for i, (n, e, st) in enumerate(rows):
    col = PEOPLE[1 + 2 * i][1]
    chip = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(4.3 + i * 0.5), Inches(0.12), Inches(0.38))
    chip.fill.solid(); chip.fill.fore_color.rgb = col; chip.line.fill.background()
    txt(s, 0.95, 4.27 + i * 0.5, 3.6, 0.45, n, 18, True)
    txt(s, 4.4, 4.27 + i * 0.5, 1.8, 0.45, e, 18, False, GREY)
    txt(s, 6.3, 4.27 + i * 0.5, 2.5, 0.45, st, 18, False, GREY)
txt(s, 9.2, 4.3, 3.8, 0.4, "Under the guidance of", 14, False, GREY)
txt(s, 9.2, 4.65, 3.8, 0.5, "Dr. Durga Toshniwal", 20, True)
txt(s, 9.2, 5.1, 3.8, 0.8, "Professor, Department of Computer Science and Engineering, IIT Roorkee", 14, False, GREY)
txt(s, 9.2, 6.3, 3.8, 0.4, "October 2026", 14, False, GREY)
s.notes_slide.notes_text_frame.text = "Introduce the group and the product: three decisions an HVAC franchise makes every week, answered with three machine learning tools."

# ============================================================================ stage 1
image_slide("Ten lifecycle stages, five authors, one product", 0, "s1_01_lifecycle_owners.png",
            "We follow the machine learning lifecycle from problem definition to monitoring. Each member wrote two stages and built the matching part of the project.")
image_slide("One dashboard, three decisions", 1, "s1_02_product_overview.png",
            "Demand Planner predicts power for tomorrow, Failure Radar ranks machines for inspection, Retention List ranks customers to call. Each tool has its own dataset and model.")
table_slide("Each question becomes a precise learning task", 1, ["", "Pillar 1: demand", "Pillar 2: failure", "Pillar 3: retention"],
            [["Task type", "Regression (a number)", "Binary classification, rare event", "Binary classification"],
             ["Unit of prediction", "One building-hour", "One machine", "One active customer at a cutoff date"],
             ["Inputs", "Outdoor and indoor temperature, humidity, hour, month, mode", "Air and process temperature, speed, torque, tool wear, type", "Purchase history before the cutoff"],
             ["Output", "Power in kW", "Failure yes or no, risk score", "Lapses yes or no, risk score"],
             ["Looking ahead", "Same hour, weather forecast as input", "The recorded failure label", "90 days after the cutoff"],
             ["Main metric", "Mean absolute error in kW", "PR-AUC; failures found in the top 10%", "ROC-AUC; lapsers found in the top 10%"],
             ["Why this metric", "Average miss, easy to read", "Failures are 3.4% of machines, so accuracy is meaningless", "A small call budget means the top of the list matters"]],
            [1.5, 3, 3.2, 3.2], 15, "We fixed these definitions before touching any data. For failure, a model that always says no failure is 96.6% accurate and finds nothing, so we use PR-AUC and top-10% capture.", row_h=0.7, center_from=9)
image_slide("The bar to beat: what a rule of thumb achieves", 1, "s1_03_rule_of_thumb.png",
            "For each pillar we picked the simplest sensible rule a company could use today. A model must beat it, and also beat a statistical and a classic ML baseline.")
# ============================================================================ stage 2
image_slide("Three public datasets, 1.7 MB in total", 2, "s2_01_data_sources.png",
            "None of the datasets comes from an HVAC company. They are stand-ins, and each card ends with its main caveat.")
table_slide("The datasets at a glance, with sources", 2, ["Pillar", "Dataset", "Rows", "Size", "Target", "Source"],
            [["1. Demand", "HVAC power consumption, hourly 2025", "8,760", "0.49 MB", "Power in kW", "kaggle.com/datasets/felixpradoh/hvac-power-consumption"],
             ["2. Failure", "AI4I 2020 Predictive Maintenance", "10,000", "0.52 MB", "Machine failure, 3.39%", "archive.ics.uci.edu/dataset/601"],
             ["3. Churn", "Online Retail II (one row per invoice)", "44,870 invoices", "0.76 MB", "No purchase in the next 90 days (built by us)", "archive.ics.uci.edu/dataset/502"]],
            [1.2, 3.2, 1.5, 1.1, 2.6, 3.6], 16, "The Kaggle link was rebuilt from the dataset name we recorded, so check that it opens before submitting.",
            note_line="Caveats: the HVAC file looks simulated, AI4I is synthetic data about machine tools, and Online Retail II is a UK online gift shop.", row_h=1.25, center_from=9)
image_slide("A year of hourly readings that move in monthly steps", 2, "s2_02_hvac_year_overview.png",
            "Temperature steps up or down at the start of nearly every month, which is why we think the file is simulated. 824 hours reach 40 degrees or more, which gives us the heatwave test.")
image_slide("AI4I: 339 failures in 10,000 machines", 2, "s2_03_ai4i_composition.png",
            "Low-quality machines fail most. Heat dissipation, overstrain and power failures dominate. The failure-type columns are never used as inputs.")
image_slide("Online Retail II: seasonal peaks, mostly UK customers", 2, "s2_04_retail_activity.png",
            "Invoices and active customers peak in October and November. December 2011 is partial. This seasonality affects how we read churn.")
# ============================================================================ stage 3
image_slide("From 1.07 million lines to 44,870 invoices", 3, "s3_01_retail_cleaning_funnel.png",
            "We drop lines without a customer id (22.8%) and lines with a price of zero or less, then group by invoice. The right panel shows rows per churn snapshot and the lapse rate.")
image_slide("The churn target is built from the future", 3, "s3_02_churn_table_timeline.png",
            "Features use only invoices before the cutoff. The target is no purchase in the 90 days after. Five training cutoffs, one later test cutoff.")
image_slide("Every test hides data the model never saw", 3, "s3_03_split_designs.png",
            "Demand has four tests: random hours, unseen months, a forecast of months 10 to 12 and a heatwave at 40 degrees or more. Failure uses a stratified split. Churn splits both customers and time.")
table_slide("Seven guards against leakage", 3, ["Risk", "What we did", "Check"],
            [["Features that use the future (churn)", "Features look only at invoices before the cutoff", "Automated test deletes later invoices"],
             ["Target built from the past (churn)", "Target looks only at invoices after the cutoff", "Automated test"],
             ["Same customer in training and test", "Customers are split before rows are built", "Automated test"],
             ["Overlapping windows", "Last training window ends before the test cutoff", "Automated test"],
             ["Answer hidden in the inputs (failure)", "The five failure-type columns are never inputs", "Automated test"],
             ["Test statistics in the pipeline", "Imputing and scaling sit inside the model pipeline", "By design"],
             ["Threshold chosen on test data", "Threshold comes from out-of-fold training scores", "By design"]],
            [3.4, 5.4, 3.2], 15, "Leakage makes scores look excellent and fail in real use. Six of the seven guards are automated tests; 17 tests in all.", row_h=0.62, center_from=9)
# ============================================================================ stage 4
image_slide("Power depends on the month first, the hour second", 4, "s4_01_hvac_heatmap.png",
            "May to August are dark all day. February, March, October and November are mostly idle. The monthly steps in the data show up here.")
image_slide("Mode follows temperature; power follows the gap", 4, "s4_02_hvac_modes.png",
            "Idle 0.19 kW, heating 1.84 kW, cooling 2.73 kW. The 824 hours at 40 degrees or more are all in cooling mode.")
image_slide("The absolute gap explains power better than temperature", 4, "s4_03_hvac_corr.png",
            "Correlation with power: absolute gap 0.85, outdoor temperature 0.63, indoor 0.52, humidity 0.16, hour 0.14.")
image_slide("Failing machines sit at both ends of torque and power", 4, "s4_04_ai4i_distributions.png",
            "A U-shape: a straight line cannot capture it, trees can. This predicts the gap between linear and tree models in stage 8.")
image_slide("Customers who order often and regularly rarely lapse", 4, "s4_07_retail_lapse_drivers.png",
            "Lapse falls from 68% with one order to 5% with 17 or more. A short order rhythm predicts staying. Recency, the basis of the usual rule, is the weakest signal.")
table_slide("What the exploration told us to do", 4, ["Pillar", "Finding", "Number", "Consequence"],
            [["1", "Power follows the gap between outdoor and indoor temperature", "correlation 0.85", "Add the absolute gap; a linear model on it can extrapolate"],
             ["1", "Idle hours draw 0.19 kW, cooling 2.73 kW", "by mode", "Keep mode as an input"],
             ["1", "824 hours reach 40 °C or more, all in cooling mode", "9.4% of hours", "Use them as the heatwave test"],
             ["2", "Failure is U-shaped in torque, speed and power", "17% to 18% in top decile, against 3.4%", "Trees should beat linear models"],
             ["2", "No single input correlates strongly with failure", "at most 0.19", "Combine inputs; use PR-AUC"],
             ["3", "Lapse falls from 68% to 5% as order count rises", "frequency", "The signal is in the order history"],
             ["3", "Recency is the weakest of the main signals", "correlation 0.16", "The recency rule should be a weak baseline"],
             ["3", "Money columns are very skewed", "top 1% hold 32% of spend", "Use log versions"]],
            [0.8, 5.2, 3, 4], 14, "", row_h=0.58, center_from=9)
# ============================================================================ stage 5
image_slide("Engineered inputs come from the physics and the business", 5, "s5_01_feature_pipeline.png",
            "Pillar 1 goes from 10 to 13 inputs, Pillar 2 from 8 to 11, Pillar 3 from 3 to 16. Every new input has a one-sentence reason.")
image_slide("Folding the V: absolute gap turns two slopes into one", 5, "s5_02_hvac_gap_features.png",
            "Power against temperature is a V. Against the absolute gap it becomes a single rising band, which a straight line follows well.")
image_slide("Engineered failure inputs expose the thresholds", 5, "s5_03_ai4i_engineered.png",
            "The lowest temperature-gap tenth fails 12% of the time, the top mechanical-power tenth 17% and the top wear-times-torque tenth 15%.")
image_slide("Engineering halves Ridge error, and does little for trees", 5, "s5_05_basic_vs_engineered.png",
            "Linear models need the gap handed to them. XGBoost builds such shapes itself by splitting.")
table_slide("Basic against engineered inputs, on unseen data", 5, ["Test", "Ridge, basic", "Ridge, engineered", "XGBoost, basic", "XGBoost, engineered"],
            [[lab, sv(p, sp, "Ridge" + B, c, 3, False), sv(p, sp, "Ridge" + E, c, 3, False), sv(p, sp, "XGBoost" + B, c, 3, False), sv(p, sp, "XGBoost" + E, c, 3, False)]
             for lab, p, sp, c in (("Demand, random split (MAE, kW)", "demand", "random", "mae"), ("Demand, unseen months (MAE, kW)", "demand", "unseen_months", "mae"),
                                   ("Demand, forecast (MAE, kW)", "demand", "forecast", "mae"), ("Demand, heatwave (MAE, kW)", "demand", "heatwave", "mae"),
                                   ("Failure (PR-AUC)", "failure", "stratified_20pct", "pr_auc"), ("Churn (ROC-AUC)", "churn", SP3, "roc_auc"), ("Churn (PR-AUC)", "churn", SP3, "pr_auc"))],
            [3.6, 2, 2, 2, 2], 18, "Lower is better for demand, higher is better for failure and churn.", row_h=0.68)
# ============================================================================ stage 6
table_slide("Five kinds of predictor, from a rule of thumb to boosted trees", 6, ["Model", "Role in this project", "How it fits", "Beyond the training range?", "Main point"],
            [["Rule of thumb", "No learning", "Average by hour; most worn tool first; longest since last purchase", "Not applicable", "What a company does today"],
             ["Linear / logistic regression", "Statistical baseline", "One weight per input, no penalty", "Yes (straight line)", "Explainable; cannot bend"],
             ["Ridge", "Our model 1", "Same, with an L2 penalty chosen by cross-validation", "Yes (straight line)", "Stable with many inputs"],
             ["Random Forest", "Classic ML baseline", "300 trees on random subsets, averaged", "No (flat beyond training range)", "Few settings; handles thresholds"],
             ["XGBoost", "Our model 2", "400 or 300 small trees, each fixing the errors of the previous ones", "No (flat beyond training range)", "Most accurate on tables"]],
            [2.3, 2.1, 3.9, 2.5, 2.5], 15, "The column about extrapolation decides the heatwave test.", row_h=0.95, center_from=9)
image_slide("Why trees go flat outside their training range", 6, "s6_02_extrapolation_toy.png",
            "A toy example: the linear model continues the trend, Random Forest and XGBoost stay at the last value they saw. Our heatwave hours are above anything in training.")
image_slide("Every model on every test", 6, "s6_03_performance_matrix.png",
            "Trees are green on ordinary demand and on failure. In the heatwave the linear models are green and the trees are four to five times worse. Churn is a tie.")
image_slide("Tie rule: the simplest model within one std wins", 6, "s6_04_selection_rule.png",
            "For failure, Random Forest and XGBoost tie. For churn, five models tie, so the simplest, logistic regression, wins. We ship XGBoost for failure and Ridge for churn, which are inside the same band.")
# ============================================================================ stage 7
image_slide("One protocol for every model, repeated with five seeds", 7, "s7_01_training_protocol.png",
            "Split, features, tune the penalty by cross-validation on training rows, fit once, choose the threshold from out-of-fold scores, test once. Seeds 42 to 46.")
table_slide("Settings fixed in advance", 7, ["Setting", "Value", "Applies to"],
            [["Seeds", "42, 43, 44, 45, 46", "All experiments"], ["Held-out share", "20% of rows, customers or hours", "All pillars"],
             ["Held-out months / forecast months", "3, 6, 9, 12 / 10, 11, 12", "Demand"], ["Heatwave threshold", "40 °C", "Demand"],
             ["Churn cutoffs", "1 Jun, 1 Sep, 1 Dec 2010; 1 Mar, 1 Jun 2011; test 9 Sep 2011", "Churn"],
             ["Ridge penalty candidates", "0.001 to 1,000 (alpha); 0.01 to 100 (C); 5-fold cross-validation", "Demand; failure and churn"],
             ["XGBoost, demand", "400 trees, depth 4, learning rate 0.05, 80% rows and columns", "Pillar 1"],
             ["XGBoost, failure and churn", "300 trees, depth 3, learning rate 0.05, 80% rows and columns", "Pillars 2 and 3"],
             ["Random Forest", "300 trees, at least 5 rows per leaf", "All pillars"], ["Linear and logistic regression", "No penalty", "Statistical baseline"]],
            [3.4, 6, 2.6], 14, "Nothing is tuned on test data. Cross-validation picked alpha 0.1 for demand and C 0.1 for failure and churn.", row_h=0.5, center_from=9)
image_slide("Trees need more data than Ridge, and for churn never pass it", 7, "s7_02_learning_curves.png",
            "Ridge learns its straight line early. XGBoost climbs for failure and keeps improving slowly. For churn both end at about 0.74 ROC-AUC.")
image_slide("The results do not hinge on a lucky setting", 7, "s7_03_sensitivity.png",
            "Ridge penalty, number of trees, tree depth and churn penalty all move the score only a little. In the heatwave no number of trees fixes XGBoost.")
image_slide("Training costs seconds, not hours", 7, "s7_04_fit_times.png",
            "Every model fits in under 1.2 seconds on a laptop. The whole experiment takes about 11 minutes. One prediction takes 1.4 to 2.4 milliseconds.")
# ============================================================================ stage 8
dem = ["Reference: training mean", "Reference: average by hour of day", "Linear regression" + E, "Ridge" + B, "Ridge" + E, "Random Forest" + E, "XGBoost" + B, "XGBoost" + E]
dem_lab = ["Rule: training mean", "Rule: average by hour", "Linear regression", "Ridge, basic", "Ridge", "Random Forest", "XGBoost, basic", "XGBoost"]
table_slide("Demand: error in kW on unseen data", 8, ["Test"] + dem_lab,
            [[lab] + [sv("demand", sp, m, "mae", 3, False) for m in dem] for lab, sp in (("Random 20% of hours", "random"), ("Unseen months", "unseen_months"), ("Forecast, months 10-12", "forecast"), ("Heatwave, 40 °C and above", "heatwave"))],
            [2.4, 1.4, 1.4, 1.4, 1.2, 1.1, 1.4, 1.3, 1.1], 13, "Lower is better. Mean over 5 seeds; the std is 0.001 to 0.002 kW on the random test and 0.009 or less in the heatwave. Rules of thumb are wrong by about half of the 1.9 kW average load.",
            row_h=0.8, note_line="Heatwave: Ridge 0.077 kW, XGBoost 0.336 kW, Random Forest 0.365 kW. Trees cannot predict above the temperatures they saw.")
image_slide("In the heatwave XGBoost under-predicts by up to 0.8 kW", 8, "s8_01_p1_residuals.png",
            "Forecast months: both models are centred on zero, XGBoost is narrower. Heatwave: XGBoost's residual grows with temperature while Ridge stays near zero. That is why the dashboard switches to Ridge at 40 degrees.")
image_slide("Error by hour and by month", 8, "s8_02_p1_error_by_time.png",
            "Ridge struggles in the afternoon and in March. XGBoost stays between 0.06 and 0.085 kW in every held-out month.")
fl = [("Rule: most worn tool first", "Reference: most worn tool first"), ("Logistic regression", "Logistic regression" + E), ("Ridge, basic", "Ridge" + B), ("Ridge", "Ridge" + E),
      ("Random Forest", "Random Forest" + E), ("XGBoost, basic", "XGBoost" + B), ("XGBoost", "XGBoost" + E)]
table_slide("Failure: trees find 94% of failures in the top 10% of machines", 8, ["Model", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Failures in top 10%"],
            [[lab] + [sv("failure", "stratified_20pct", m, c, 3) for c in ("roc_auc", "pr_auc", "f1", "precision", "recall", "top10_capture")] for lab, m in fl],
            [2.8, 1.7, 1.7, 1.7, 1.7, 1.7, 1.9], 14, "2,000 held-out machines, 68 failures per seed. A random guess has PR-AUC 0.034.", row_h=0.62)
image_slide("Ranking quality and the alert threshold", 8, "s8_03_p2_roc_threshold.png",
            "Random Forest and XGBoost rise almost vertically on the ROC curve. F1 is flat from 0.2 to 0.6, so the result does not depend on a precise threshold.")
image_slide("Finding 80% of failures: 3% of machines, not 58%", 8, "s8_07_failure_top_k.png",
            "The rule of thumb needs about 19 times as many inspections as XGBoost to catch the same share of failures.")
ch = [("Rule: longest since purchase", "Reference: recency only"), ("Logistic regression", "Logistic regression" + E), ("Random Forest", "Random Forest" + E), ("Ridge, basic", "Ridge" + B),
      ("Ridge", "Ridge" + E), ("XGBoost, basic", "XGBoost" + B), ("XGBoost", "XGBoost" + E)]
table_slide("Churn: every model reaches about 0.78 ROC-AUC", 8, ["Model", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Lapsers in top 10%"],
            [[lab] + [sv("churn", SP3, m, c, 3) for c in ("roc_auc", "pr_auc", "f1", "precision", "recall", "top10_capture")] for lab, m in ch],
            [2.8, 1.7, 1.7, 1.7, 1.7, 1.7, 1.9], 14, "About 360 test customers, about 32% lapse. Differences between models are smaller than one standard deviation.", row_h=0.62)
image_slide("Alerts at each model's own threshold", 8, "s8_04_confusion_matrices.png",
            "XGBoost flags 59 of 2,000 machines and 54 are real failures, missing 14 of 68. Ridge catches only 28. For churn the lists are long, because the F1 threshold favours recall when lapsing is common. A manager should work down the ranking, not use the threshold.")
image_slide("Do the probabilities mean what they say?", 8, "s8_05_calibration.png",
            "For failure the few high scores sit close to the diagonal. For churn both models are over-confident at the high end: about 0.76 predicted, 0.58 to 0.60 observed. We show risk as a rank.")
image_slide("Spread over the five seeds", 8, "s8_06_seed_variability.png",
            "Demand barely moves. Failure and especially churn vary because the test sets are small, which is why gaps below one standard deviation count as ties.")
image_slide("Where XGBoost helps and where it does not", 8, "10_xgboost_vs_ridge.png",
            "Better by 32%, 39%, 24% on ordinary demand tests and 82% on failure. Worse by 333% in the heatwave. No gain on churn.")
table_slide("Scoreboard: the rule of thumb against our models", 8, ["Pillar", "Measure", "Rule of thumb", "Statistical baseline", "Ridge", "XGBoost", "Best result"],
            [["Demand, forecast test", "Error in kW (lower)", "1.02", "0.101", "0.100", "0.076", "XGBoost"], ["Demand, heatwave", "Error in kW (lower)", "1.57", "0.077", "0.077", "0.336", "Linear models"],
             ["Failure", "PR-AUC (higher)", "0.123", "0.487", "0.492", "0.893", "Random Forest 0.912, XGBoost close"], ["Failure", "Failures in top 10%", "36%", "74%", "73%", "94%", "Random Forest and XGBoost"],
             ["Churn", "ROC-AUC (higher)", "0.607", "0.775", "0.777", "0.775", "All four models tie"], ["Churn", "Lapsers in top 10%", "16%", "20%", "20%", "20%", "All four models tie"]],
            [2.3, 2.1, 1.5, 1.8, 1.1, 1.2, 3], 14, "Statistical baseline: linear regression for demand, logistic regression for failure and churn.", row_h=0.65)
# ============================================================================ stage 9
image_slide("Architecture: train once, serve in milliseconds", 9, "s9_01_architecture.png",
            "The pipeline writes a 1.4 MB models file and the dashboard only reads it. Training and serving are kept apart.")
image_slide("Demand Planner: a normal day and a heatwave day", 9, "s9_02_planner_output.png",
            "Peak 3.07 kW and 46.8 kWh on a normal July day. Peak 3.91 kW and 75.5 kWh on a heatwave day, where the safeguard uses Ridge from about 11:00 to 22:00.")
image_slide("The heatwave safeguard: Ridge at 40 °C and above", 9, "s9_05_safeguard_logic.png",
            "Planning with XGBoost alone would be low by 0.3 kW or more per hour on the hottest days. The dashboard shows both curves and says how many hours used the safeguard.")
fr = pd.read_csv(RES / "failure_radar_example.csv")
table_slide("Failure Radar: the 12 riskiest machines in a sample of 500", 9, list(fr.columns),
            [[str(v) for v in r] for r in fr.itertuples(index=False)], [1.6, 0.9, 1.4, 1.4, 1.5, 1.2, 1.2, 1.6], 15,
            "The tool lists machines by risk and flags those above the alert threshold (44% for XGBoost).",
            note_line="Illustration: the saved model was trained on all 10,000 machines, so this list is a demonstration, not a test. Honest scores are in stage 8.", row_h=0.42)
rl = pd.read_csv(RES / "retention_list_example.csv")
table_slide("Retention List: the 12 customers most likely to go quiet", 9, list(rl.columns),
            [[str(v) for v in r] for r in rl.itertuples(index=False)], [1.4, 1.2, 1.5, 1.1, 1.2, 1.5, 1.6], 15,
            "The team picks how many customers it can contact, and the tool returns the highest-risk ones.",
            note_line="Illustration: the final model saw these customers. In the honest test, the top 10% of the list reaches 20% of lapsers, twice the random rate.", row_h=0.42)
# ============================================================================ stage 10
image_slide("The monitoring loop", 10, "s10_01_monitoring_loop.png",
            "Input checks run as soon as data arrives and warn early. Error and rate checks need real outcomes and give the final proof.")
table_slide("Alert rules", 10, ["Check", "What is measured", "Warning at", "Alert at", "Action"],
            [["Input drift", "PSI of each model input", ">= 0.10", ">= 0.25", "Check the data source; refit if it persists"],
             ["Demand error", "Monthly MAE against the tested 0.065 kW", ">= 1.5 times", ">= 2.0 times", "Refit on the newest months; check sensors"],
             ["Heatwave safeguard", "Hours above 40 °C that used Ridge", "any", "Error >= 2 x 0.08 kW", "Extend the Ridge range; review features"],
             ["Failure ranking", "Failures found in the top 10% (tested 94%)", "< 85%", "< 70%", "Refit with the new failures; review sensors"],
             ["Churn calibration", "Predicted against actual lapse rate", "Gap > 0.10", "Gap > 0.20", "Refit on the latest cutoff; reset the threshold"]],
            [2.0, 3.8, 1.8, 2.1, 3.6], 15, "The first two rows are implemented and tested as functions; the rest are applied by hand for now.",
            note_line="PSI thresholds are the usual rules of thumb (0.10 moderate, 0.25 large shift), not values fitted to our data.", row_h=0.85, center_from=2)
image_slide("The checks fire when the weather moves into new ground", 10, "s10_02_input_drift.png",
            "Months 10 to 12 against months 1 to 9 alert because of the season. Heatwave hours against cooler hours reach PSI 8.3. For churn, tenure drifts because customers are a year older.")
image_slide("Error over time, and predicted against actual churn", 10, "s10_03_performance_over_time.png",
            "A model trained only on earlier months has large errors in April, May and July. The churn rate prediction misses by 0.26 in the holiday window, although the ranking stays useful.")
image_slide("Maintenance calendar", 10, "s10_05_retraining_timeline.png",
            "Demand is refitted before and after the cooling season, churn each quarter, failure after 100 new logged failures. Yearly review in December.")
# ============================================================================ summary
image_slide("What each tool is for, and what not to use it for", 0, "s12_01_recommendation_map.png",
            "A manager can trust the Demand Planner with its safeguard and the Failure Radar for ranking. The Retention List is a modest aid.")
image_slide("Recommendation: pilot Failure Radar and Demand Planner first", 0, "s12_02_recommendation_table.png",
            "Retrain on company data before relying on any score. Run a four-week side-by-side pilot. Monitoring starts on day one.")
image_slide("Limits and future work: where a graph model might help", 0, "s13_01_future_work_map.png",
            "A graph neural network did not beat simple models in an earlier test, so we removed it. Churn, cross-selling and multi-building demand are the places worth a fair test. It would not fix the heatwave.")
s = base("Conclusion", 0, "Close with the recommendation and thank the audience.")
lines = ["Machine learning helps most where a ranking drives an action: inspecting machines, with 94% of failures in the top 10% of the list.",
         "Demand can be forecast to 0.065 kW, but only a linear model stays safe in a heatwave, so the planner switches at 40 °C.",
         "Retention gains are modest: about twice the hit rate of random calling, and all models tie.",
         "We recommend a pilot of the Failure Radar and the Demand Planner on company data, with monitoring from the first day.",
         "All scores come from public stand-in data and unseen test data; the method transfers, the numbers do not."]
for i, l in enumerate(lines):
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.7), Inches(1.65 + i * 1.0), Inches(0.22), Inches(0.22))
    dot.fill.solid(); dot.fill.fore_color.rgb = NAVY; dot.line.fill.background()
    txt(s, 1.1, 1.45 + i * 1.0, 11.6, 0.8, l, 20)
txt(s, 0.7, 6.6, 12, 0.5, "Thank you. Questions?", 24, True, NAVY)
table_slide("References and data sources", 0, ["Source", "Reference"],
            [["Data 1", "felixpradoh. HVAC Power Consumption (hourly, 2025). Kaggle. kaggle.com/datasets/felixpradoh/hvac-power-consumption"],
             ["Data 2", "Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository, 601."],
             ["Data 3", "Chen, D. (2019). Online Retail II. UCI Machine Learning Repository, 502."],
             ["Ridge", "Hoerl and Kennard (1970). Ridge regression: biased estimation for nonorthogonal problems. Technometrics 12(1)."],
             ["Random Forest", "Breiman (2001). Random forests. Machine Learning 45(1)."],
             ["XGBoost", "Chen and Guestrin (2016). XGBoost: a scalable tree boosting system. KDD."],
             ["scikit-learn", "Pedregosa et al. (2011). Scikit-learn: machine learning in Python. JMLR 12."],
             ["PR curves", "Saito and Rehmsmeier (2015). The precision-recall plot is more informative than the ROC plot on imbalanced data. PLOS ONE 10(3)."],
             ["PSI", "Siddiqi (2006). Credit Risk Scorecards. Wiley."],
             ["GraphSAGE", "Hamilton, Ying and Leskovec (2017). Inductive representation learning on large graphs. NeurIPS 30."]],
            [1.8, 10.7], 13, "", row_h=0.5, center_from=9)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
