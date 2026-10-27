"""
figs_diagrams.py - diagrams for the report and slides (drawn with matplotlib, numbers taken from the saved facts).
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

from hvac_ml import config
from hvac_ml.figs_common import (BLUE, GOLD, GREEN, GREY, LIGHT, NAVY, ORANGE, OWNERS, PURPLE, RED, STAGES, arrow, box, canvas, header_box,
                                 load_facts, owner_of, save, table_figure)


# stage 1
def lifecycle_owners():
    fig, ax = canvas(14, 6.6, "The ten lifecycle stages and who wrote each one", ylim=(-7, 46))
    short = ["Problem\ndefinition", "Data\ncollection", "Data cleaning and\npreprocessing", "Exploratory\ndata analysis", "Feature\nengineering",
             "Model\nselection", "Model\ntraining", "Model evaluation\nand tuning", "Model\ndeployment", "Monitoring and\nmaintenance"]
    out = ["3 decisions,\n3 pillars", "3 public datasets,\n1.7 MB", "invoice table,\nchurn labels", "what drives power,\nfailure, lapse", "gap, power and\norder-gap features",
           "Ridge, XGBoost\n+ baselines", "5 seeds, fixed\nsettings", "unseen months,\nmachines, customers", "dashboard with\nthree tools", "drift and error\nchecks"]
    xs = [1 + i * 19.8 for i in range(5)]
    for i in range(10):
        row, col = divmod(i, 5)
        x, y = xs[col], 26 if row == 0 else 4
        o = OWNERS[owner_of(i + 1)]
        ax.add_patch(plt.Rectangle((x, y), 17.5, 15.5, facecolor="white", edgecolor=o["color"], linewidth=2))
        ax.add_patch(plt.Rectangle((x, y + 11.5), 17.5, 4.0, facecolor=o["color"], edgecolor=o["color"]))
        ax.text(x + 8.75, y + 13.5, f"Stage {i + 1}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax.text(x + 8.75, y + 8.2, short[i], ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222222", linespacing=1.25)
        ax.text(x + 8.75, y + 3.2, out[i], ha="center", va="center", fontsize=8, color="#555555", linespacing=1.25)
        if col < 4:
            arrow(ax, (x + 17.7, y + 7.7), (x + 19.6, y + 7.7))
    ax.annotate("", xy=(xs[0] + 8.75, 25.6), xytext=(xs[4] + 8.75, 25.6), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6, connectionstyle="angle,angleA=0,angleB=90,rad=0"))
    for i, (key, o) in enumerate(OWNERS.items()):
        x = 1 + i * 19.8
        ax.add_patch(plt.Rectangle((x, -5.2), 17.5, 3.6, facecolor=o["color"], edgecolor="none"))
        ax.text(x + 8.75, -3.4, f"{o['full']}\n({o['enroll']}): stages {o['stages'][0]}, {o['stages'][1]}", ha="center", va="center", fontsize=7.6, color="white",
                fontweight="bold", linespacing=1.2)
    save(fig, "s1_01_lifecycle_owners.png", tight=False)


def product_overview():
    f = load_facts()
    fig, ax = canvas(14, 6.6, "HVAC Service Planner: from data to a decision", ylim=(0, 66))
    for x, w, t in ((1, 23, "Data"), (28, 24, "Model"), (56, 22, "Tool in the dashboard"), (82, 17, "Decision it supports")):
        ax.text(x + w / 2, 63.5, t, ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
    lanes = [
        ("Pillar 1: demand", BLUE, 42, f"Kaggle HVAC power\n{f['hvac']['rows']:,} hourly rows\nweather, setpoint, calendar", "XGBoost for normal weather,\nRidge above 40 °C\ntarget: power in kW",
         "Demand Planner\nhourly demand curve,\npeak hour, daily energy", "Staffing, stock and\ngrid load for\ntomorrow"),
        ("Pillar 2: failure", ORANGE, 23, f"UCI AI4I 2020\n{f['ai4i']['rows']:,} machines\n{f['ai4i']['failure_rate'] * 100:.1f}% fail", "XGBoost or Random Forest\ntarget: failure yes or no\n(risk score 0 to 1)",
         "Failure Radar\nrisk score and\nranked inspection list", "Which machines\nto inspect first"),
        ("Pillar 3: retention", GREEN, 4, f"UCI Online Retail II\n{f['retail']['customers']:,} customers\n{f['retail']['invoices']:,} invoices", "Ridge (penalised logistic)\ntarget: buys nothing in\nthe next 90 days",
         "Retention List\nranked call list of\nactive customers", "Which customers\nto call"),
    ]
    for name, color, y, d, m, t, dec in lanes:
        ax.text(0.2, y + 16.3, name, fontsize=9.5, fontweight="bold", color=color, va="bottom")
        for x, w, txt in ((1, 23, d), (28, 24, m), (56, 22, t), (82, 17, dec)):
            box(ax, x, y, w, 15, txt, color=color, size=8.6)
        for x0, x1 in ((24, 28), (52, 56), (78, 82)):
            arrow(ax, (x0 + 0.2, y + 7.5), (x1 - 0.2, y + 7.5), color=color)
    save(fig, "s1_02_product_overview.png", tight=False)


# stage 2
def data_sources():
    f = load_facts()
    fig, ax = canvas(14, 5.0, "The three datasets", ylim=(0, 42))
    cards = [
        ("Pillar 1: HVAC power (Kaggle)", BLUE, 1, f"felixpradoh/hvac-power-consumption\nhvac_power_consumption.csv\n\nsize: {f['hvac']['rows']:,} rows, 8 columns (0.5 MB)\nperiod: one year, hourly (2025)\n\ncolumns: month, hour, outdoor temp,\nindoor temp, humidity, HVAC mode,\npower in kW\n\ntarget: power_kw\nrange {f['hvac']['power']['min']:.2f} to {f['hvac']['power']['max']:.2f} kW,\nmean {f['hvac']['power']['mean']:.2f} kW\n\ncaveat: very regular, probably simulated"),
        ("Pillar 2: machine failures (UCI 601)", ORANGE, 34, f"AI4I 2020 Predictive Maintenance\nai4i2020.csv\n\nsize: {f['ai4i']['rows']:,} machines, 14 columns (0.5 MB)\nsensor readings and a failure label\n\ncolumns: type L/M/H, air temp, process\ntemp, speed, torque, tool wear\n\ntarget: machine failure (yes or no)\n{f['ai4i']['failures']} failures = {f['ai4i']['failure_rate'] * 100:.2f}%\n\ncaveat: synthetic data about machine tools,\nnot HVAC units"),
        ("Pillar 3: customers (UCI 502)", GREEN, 67, f"Online Retail II\nretail_invoices.csv.gz\n\nsize: {f['retail']['invoices']:,} invoices, {f['retail']['customers']:,} customers\n(0.8 MB; the original is 1.07 million lines, 45 MB)\nperiod: Dec 2009 to Dec 2011\n\ncolumns: customer, invoice, date, value,\nitems, products, country, is_return\n\ntarget: built by us (no purchase in the\nnext 90 days)\n\ncaveat: a UK online gift shop, not a\nservice company"),
    ]
    for title, color, x, body in cards:
        header_box(ax, x, 3, 32, 36, title, body, color, size=8.9, title_size=10, header_h=4.2)
    ax.text(50, 0.5, "All three are public. Together they are 1.7 MB in hvac_ml/data/.", ha="center", va="center", fontsize=9.5, color=NAVY)
    save(fig, "s2_01_data_sources.png", tight=False)


# stage 3
def churn_table_timeline():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
    cutoffs = [pd.Timestamp(c) for c in config.TRAIN_CUTOFFS] + [pd.Timestamp(config.TEST_CUTOFF)]
    horizon = pd.Timedelta(days=config.CHURN_HORIZON_DAYS)
    for i, c in enumerate(cutoffs):
        y = len(cutoffs) - 1 - i
        test = i == len(cutoffs) - 1
        color = ORANGE if test else BLUE
        ax.barh(y, (c - start).days, left=mdates.date2num(start), height=0.55, color="#DDE3E8", edgecolor="none")
        ax.barh(y, horizon.days, left=mdates.date2num(c), height=0.55, color=color, edgecolor="none")
        ax.plot([mdates.date2num(c)] * 2, [y - 0.42, y + 0.42], color="black", lw=1.6)
        ax.text(mdates.date2num(c) - 6, y, c.strftime("cutoff %d %b %Y"), ha="right", va="center", fontsize=8.5)
        ax.text(mdates.date2num(c) + horizon.days / 2, y, "target window", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.text(mdates.date2num(start) + 4, len(cutoffs) - 0.45, "grey bar = features: only purchases before the cutoff", va="center", fontsize=8.8, color="#444444")
    ax.set_ylim(-0.6, len(cutoffs) - 0.1)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_yticks([])
    ax.set_xlim(mdates.date2num(start) - 5, mdates.date2num(end) + 12)
    ax.grid(axis="y", visible=False)
    ax.set_title("How one row of the churn table is built: features from before the cutoff, the target from the 90 days after it", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(0.01, -0.17, "Blue rows (training): the five earlier cutoffs, training customers only (80%).    Orange row (test): one later cutoff, unseen customers only (20%).    "
            f"The last training window ends {(cutoffs[-2] + horizon).strftime('%d %b %Y')}, before the test cutoff.", transform=ax.transAxes, fontsize=8.8, color="#333333")
    ax.text(0.01, -0.24, "A customer is included only if they bought in the 90 days before the cutoff. Target = 1 when they buy nothing in the 90 days after it.",
            transform=ax.transAxes, fontsize=8.8, color="#333333")
    save(fig, "s3_02_churn_table_timeline.png")


def split_designs():
    fig, ax = canvas(14, 7.4, "How each pillar is split into training and test data", ylim=(0, 66))
    ax.text(1, 63, "Pillar 1: demand (four tests)", fontsize=11, fontweight="bold", color=BLUE)
    months = "J F M A M J J A S O N D".split()
    rows = [("random 20% of hours", None), ("unseen months", set(config.HELD_OUT_MONTHS)), ("forecast", set(config.FORECAST_MONTHS)), ("heatwave", "heat")]
    for r, (name, test) in enumerate(rows):
        y = 55 - r * 7.5
        ax.text(1, y + 2, name, fontsize=8.6, va="center")
        if test == "heat":
            ax.add_patch(Rectangle((19, y), 21, 4, facecolor=BLUE, edgecolor="white"))
            ax.add_patch(Rectangle((40, y), 11, 4, facecolor=ORANGE, edgecolor="white"))
            ax.text(29.5, y + 2, "train: below 40 °C", color="white", fontsize=8, ha="center", va="center", fontweight="bold")
            ax.text(45.5, y + 2, "test: 40 °C +", color="white", fontsize=8, ha="center", va="center", fontweight="bold")
        else:
            if test is None:                                   # hours drawn at random from the whole year: a fine mix of both colours
                rng = np.random.default_rng(5)
                for k in range(64):
                    ax.add_patch(Rectangle((19 + k * 0.5, y), 0.5, 4, facecolor=ORANGE if rng.random() < 0.2 else BLUE, edgecolor="none"))
                ax.text(19 + 16, y - 1.6, "the whole year, 20% of hours picked at random", fontsize=7.6, color="#555555", ha="center", va="center")
                continue
            for m in range(12):
                c = ORANGE if (m + 1) in test else BLUE
                ax.add_patch(Rectangle((19 + m * 2.7, y), 2.5, 4, facecolor=c, edgecolor="white"))
                ax.text(19 + m * 2.7 + 1.25, y + 2, months[m], color="white", fontsize=7.5, ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((1, 22), 2, 2, facecolor=BLUE)); ax.text(3.6, 23, "training", fontsize=8.5, va="center")
    ax.add_patch(Rectangle((12, 22), 2, 2, facecolor=ORANGE)); ax.text(14.6, 23, "test (never used for training)", fontsize=8.5, va="center")
    ax.text(1, 17.5, "Reference models: the training mean and the\naverage by hour of day, scored on the same test rows.", fontsize=8.4, color="#555555", va="top")
    # pillar 2
    ax.text(57, 63, "Pillar 2: failure", fontsize=11, fontweight="bold", color=ORANGE)
    ax.add_patch(Rectangle((57, 52), 32, 5, facecolor=BLUE)); ax.add_patch(Rectangle((89, 52), 8, 5, facecolor=ORANGE))
    ax.text(73, 54.5, "8,000 machines", color="white", fontsize=8.6, ha="center", va="center", fontweight="bold")
    ax.text(93, 54.5, "2,000", color="white", fontsize=8.6, ha="center", va="center", fontweight="bold")
    ax.text(57, 49.6, "stratified: both parts keep the same 3.4% failure share;\nrepeated with 5 seeds (a different split each time)", fontsize=8.4, color="#555555", va="top")
    # pillar 3
    ax.text(57, 40, "Pillar 3: churn (time and customers)", fontsize=11, fontweight="bold", color=GREEN)
    ax.text(70, 36.2, "earlier cutoffs\n(Jun 2010 to Jun 2011)", ha="center", fontsize=8, va="center")
    ax.text(90, 36.2, "later cutoff\n(Sep 2011)", ha="center", fontsize=8, va="center")
    ax.text(56.5, 29, "training\ncustomers\n(80%)", ha="right", fontsize=8, va="center")
    ax.text(56.5, 19, "unseen\ncustomers\n(20%)", ha="right", fontsize=8, va="center")
    cells = [((58, 25), BLUE, "train"), ((79, 25), "#DDE3E8", "not used"), ((58, 14), "#DDE3E8", "not used"), ((79, 14), ORANGE, "test")]
    for (x, y), c, t in cells:
        ax.add_patch(Rectangle((x, y), 20, 9.5, facecolor=c, edgecolor="white", linewidth=2))
        ax.text(x + 10, y + 4.75, t, ha="center", va="center", fontsize=9, color="white" if c in (BLUE, ORANGE) else "#555555", fontweight="bold")
    ax.text(57, 10.5, "No customer appears in both training and test, and every training window\nends before the test cutoff.", fontsize=8.4, color="#555555", va="top")
    save(fig, "s3_03_split_designs.png", tight=False)


# stage 5
def feature_pipeline():
    fig, ax = canvas(14, 6.6, "From raw columns to model inputs", ylim=(0, 62))
    for x, w, t in ((1, 27, "Raw columns"), (33, 38, "What we compute"), (76, 23, "Model inputs")):
        ax.text(x + w / 2, 59.5, t, ha="center", fontsize=11, fontweight="bold", color=NAVY)
    lanes = [
        (BLUE, 40, "hour, month, outdoor temp,\nindoor temp, humidity,\nHVAC mode",
         "hour and month as sine and cosine\n(23:00 sits next to 00:00)\nmode as three 0/1 columns\nabsolute gap |outdoor - indoor|\ngap x cooling, gap x heating", "10 basic inputs\n13 engineered inputs"),
        (ORANGE, 22, "air temp, process temp,\nspeed, torque, tool wear,\nmachine type",
         "temperature gap (process - air)\nmechanical power = torque x speed\nwear x torque\ntype as three 0/1 columns", "8 basic inputs\n11 engineered inputs"),
        (GREEN, 4, "per invoice: customer, date,\nvalue, items, products,\nreturn flag, country",
         "recency, frequency, total spend, tenure\naverage order value, average gap\norders in the last and the previous 90 days\ntrend, return share, UK flag, logs", "3 basic inputs\n16 engineered inputs"),
    ]
    for color, y, raw, calc, out in lanes:
        box(ax, 1, y, 27, 15, raw, color=color, size=8.8)
        box(ax, 33, y, 38, 15, calc, color=color, size=8.8, fill="#FBFBFB")
        box(ax, 76, y, 23, 15, out, color=color, size=9.5, bold=True)
        arrow(ax, (28.2, y + 7.5), (32.8, y + 7.5), color=color)
        arrow(ax, (71.2, y + 7.5), (75.8, y + 7.5), color=color)
    save(fig, "s5_01_feature_pipeline.png", tight=False)


# stage 6
def model_families():
    df = pd.DataFrame([
        ["Rule of thumb", "no learning", "average by hour; most worn tool first;\nlongest since last purchase", "n/a", "what a company does today"],
        ["Linear / logistic regression", "statistical baseline", "one weight per input, no penalty", "yes (straight line)", "explainable; cannot bend"],
        ["Ridge", "our model 1", "same, with an L2 penalty chosen\nby cross-validation", "yes (straight line)", "stable with many inputs"],
        ["Random Forest", "classic ML baseline", "300 trees on random subsets,\naveraged", "no (flat beyond training range)", "few settings; handles thresholds"],
        ["XGBoost", "our model 2", "400 or 300 small trees, each fixing\nthe errors of the previous ones", "no (flat beyond training range)", "most accurate on tables"],
    ], columns=["Model", "Role in this project", "How it fits", "Beyond the training range?", "Main point"])
    fig = table_figure(df, "The model families we compare", col_widths=[0.17, 0.15, 0.30, 0.20, 0.18], font=10.5, figsize=(14, 4.8), row_height=0.9)
    save(fig, "s6_01_model_families.png")


def extrapolation_toy():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 160))
    y = 0.8 * x + rng.normal(0, 0.5, len(x))
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    xx = np.linspace(0, 16, 200)[:, None]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.scatter(x, y, s=12, color=GREY, label="training data (0 to 10)")
    ax.plot(xx, 0.8 * xx, color="black", ls=":", lw=1.2, label="true relationship")
    ax.plot(xx, LinearRegression().fit(x[:, None], y).predict(xx), color=BLUE, lw=2, label="linear model")
    ax.plot(xx, RandomForestRegressor(200, min_samples_leaf=3, random_state=0).fit(x[:, None], y).predict(xx), color=PURPLE, lw=2, label="Random Forest")
    ax.plot(xx, xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1).fit(x[:, None], y).predict(xx), color=ORANGE, lw=2, label="XGBoost")
    ax.axvspan(10, 16, color="#FFF3E0", alpha=0.8)
    ax.text(13, 1.0, "never seen\nin training", ha="center", fontsize=9, color="#885500")
    ax.set(xlabel="input (for example outdoor temperature)", ylabel="output (for example power)", title="Why trees go flat outside the training range (a toy example)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "s6_02_extrapolation_toy.png")


# stage 7
def training_protocol():
    fig, ax = canvas(14, 4.4, "What happens in one run of the experiment (repeated for seeds 42 to 46)", ylim=(3, 44))
    steps = [("1. Split", "hold out the test data\n(by month, temperature,\nmachine or customer)", BLUE),
             ("2. Features", "build the basic and\nengineered inputs from\nthe training rows", BLUE),
             ("3. Tune penalty", "Ridge alpha or logistic C\nchosen by 5-fold cross-\nvalidation on training\ndata only", GREEN),
             ("4. Fit", "fit each model once on\nall training rows\n(settings fixed in\nconfig.py)", GREEN),
             ("5. Threshold", "F1-best alert threshold\nfrom out-of-fold training\nscores (yes/no tasks)", ORANGE),
             ("6. Test", "one pass over the test\nrows: MAE, PR-AUC,\nROC-AUC, top 10%", RED)]
    w, gap, x0 = 14.6, 1.9, 1.2
    for i, (t, b, c) in enumerate(steps):
        x = x0 + i * (w + gap)
        header_box(ax, x, 20, w, 20, t, b, c, size=7.7, title_size=9.4, header_h=4.4)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.2, 30), (x + w + gap - 0.2, 30))
    ax.text(50, 13, "After the six steps the seed is changed and everything is repeated. We report the mean and the standard deviation over the five seeds.", ha="center", fontsize=9.5, color=NAVY)
    ax.text(50, 7.5, "The test rows are used once, at step 6. Nothing is tuned on them.", ha="center", fontsize=9.5, color="#333333", fontweight="bold")
    save(fig, "s7_01_training_protocol.png", tight=False)


# stage 9
def architecture():
    f = load_facts()
    fig, ax = canvas(14, 7.2, "System architecture", ylim=(0, 68))
    layers = [
        (54, "Data (hvac_ml/data/, 1.7 MB)", GREY, ["hvac_power_consumption.csv", "ai4i2020.csv", "retail_invoices.csv.gz"]),
        (38, "Pipeline (python run_all.py, about 10 minutes)", BLUE, ["data.py\nload and clean", "pillar1/2/3 .py\nfeatures, models, tests", "common.py\nmodels and metrics", "plots.py\nfigures"]),
        (22, "Saved artifacts (hvac_ml/outputs/)", GREEN, [f"models/hvac_models.joblib\n{f['timing']['model_file_mb']} MB", "results/*.csv and\nRESULTS_SUMMARY.md", "figures/*.png", "run_log.txt"]),
        (6, "Application and checks", ORANGE, ["app.py (Streamlit)\nDemand Planner\nFailure Radar, Retention List", "monitoring.py\ndrift and error checks", "tests/ (17 checks)\nrun with pytest"]),
    ]
    for y, title, color, items in layers:
        ax.text(1, y + 11.2, title, fontsize=10, fontweight="bold", color=color)
        n = len(items)
        w = (98 - (n - 1) * 2.5) / n
        for i, it in enumerate(items):
            box(ax, 1 + i * (w + 2.5), y, w, 10, it, color=color, size=8.6)
    for y in (54, 38, 22):
        arrow(ax, (50, y - 0.3), (50, y - 4.4), color=NAVY, lw=2)
    ax.text(51.5, 50.4, "read by", fontsize=8.5, color=NAVY); ax.text(51.5, 34.4, "writes", fontsize=8.5, color=NAVY); ax.text(51.5, 18.4, "loaded by", fontsize=8.5, color=NAVY)
    ax.text(50, 1.8, "users: dispatch manager, maintenance lead, customer-success team", fontsize=9.5, color="#333333", va="center", ha="center")
    save(fig, "s9_01_architecture.png", tight=False)


def safeguard_logic():
    f = load_facts()
    fig, ax = canvas(14, 5.8, "Demand Planner: how the heatwave safeguard picks the model for each hour", ylim=(0, 48))
    box(ax, 1, 18, 17, 12, "Weather forecast\nfor the day\n(24 hourly temperatures)", color=BLUE, size=9)
    box(ax, 23, 18, 15, 12, "Build the model\ninputs for each\nhour", color=BLUE, size=9)
    arrow(ax, (18.2, 24), (22.8, 24)); arrow(ax, (38.2, 24), (42.8, 24))
    ax.add_patch(Polygon([[43, 24], [56, 36], [69, 24], [56, 12]], closed=True, facecolor="#FFF8E1", edgecolor=GOLD, linewidth=2))
    ax.text(56, 24, f"outdoor temperature\n>= {config.HEATWAVE_TEMP_C:.0f} °C, or outside\n{f['hvac']['ambient']['min']:.1f} to {f['hvac']['ambient']['max']:.1f} °C?", ha="center", va="center", fontsize=8.4)
    box(ax, 76, 33, 22, 10, "Ridge model\n(extends the trend)", color=BLUE, size=9.5, bold=True)
    box(ax, 76, 5, 22, 10, "XGBoost model\n(most accurate in\nnormal weather)", color=ORANGE, size=9.5, bold=True)
    arrow(ax, (62, 31), (76, 37), color=BLUE); arrow(ax, (62, 17), (76, 11), color=ORANGE)
    ax.text(68, 36, "yes", fontsize=9.5, color=BLUE, fontweight="bold"); ax.text(68, 13, "no", fontsize=9.5, color=ORANGE, fontweight="bold")
    ax.text(1, 9, "Why: in the heatwave test (train below 40 °C, predict 40 °C and above) Ridge is off by 0.08 kW on average\nand XGBoost by 0.34 kW, because trees stay flat beyond the temperatures they were trained on.", fontsize=9, color="#333333", va="top")
    ax.text(1, 2.6, "The planner shows both curves and says how many hours used the safeguard.", fontsize=9, color="#333333")
    save(fig, "s9_05_safeguard_logic.png", tight=False)


# stage 10
def monitoring_loop():
    fig, ax = canvas(11, 6.4, "The monitoring loop", xlim=(0, 100), ylim=(0, 64))
    nodes = [("1. New data arrives\n(weather, readings,\ninvoices)", 50, 54, BLUE), ("2. Input drift check\nPSI and KS test per input", 82, 41, BLUE), ("3. Score and log\npredictions", 82, 17, GREEN),
             ("4. Outcomes arrive\n(real power, failures,\npurchases)", 50, 7, GREEN), ("5. Error and rate check\nMAE, calibration", 18, 17, ORANGE), ("6. Decide:\nok, warning, alert", 18, 41, RED)]
    for t, x, y, c in nodes:
        box(ax, x - 14, y - 5.5, 28, 11, t, color=c, size=8.6)
    order = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    pts = [(n[1], n[2]) for n in nodes]
    for a, b in order:
        (x0, y0), (x1, y1) = pts[a], pts[b]
        dx, dy = x1 - x0, y1 - y0
        norm = np.hypot(dx, dy)
        arrow(ax, (x0 + dx / norm * 15.5, y0 + dy / norm * 7.5), (x1 - dx / norm * 15.5, y1 - dy / norm * 7.5), color=NAVY, lw=1.8)
    ax.text(50, 29, "warning: look at the cause\nalert: retrain or switch\nto the safe model", ha="center", va="center", fontsize=8.6, color="#333333", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT, edgecolor=GREY))
    save(fig, "s10_01_monitoring_loop.png", tight=False)


def retraining_timeline():
    tasks = [("Input drift check (all pillars)", [1] * 12, BLUE), ("Demand error check (when a month closes)", [1] * 12, GREEN), ("Demand: refit before the cooling season", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ORANGE),
             ("Demand: refit after the heat season", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], ORANGE), ("Churn: new cutoff, refit, rate check (quarterly)", [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], PURPLE),
             ("Failure: refit when 100 new failures are logged", [0] * 12, RED), ("Yearly review: data sources, features, thresholds", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], NAVY)]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    for r, (name, flags, color) in enumerate(tasks):
        y = len(tasks) - 1 - r
        for m, on in enumerate(flags):
            ax.add_patch(Rectangle((m, y), 0.92, 0.8, facecolor=color if on else "#EEF1F4", edgecolor="white", alpha=0.95 if on else 1.0))
        ax.text(-0.3, y + 0.4, name, ha="right", va="center", fontsize=9)
        if name.startswith("Failure"):
            ax.text(6, y + 0.4, "on a trigger (100 new logged failures), not on a fixed date", ha="center", va="center", fontsize=8.6, color=RED, fontweight="bold")
    ax.set_xlim(0, 12); ax.set_ylim(0, len(tasks))
    ax.set_xticks([m + 0.46 for m in range(12)]); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks([]); ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("Maintenance calendar (one year)", fontsize=12, fontweight="bold", color=NAVY)
    fig.subplots_adjust(left=0.36)
    save(fig, "s10_05_retraining_timeline.png", tight=False)


def alert_rules():
    df = pd.DataFrame([
        ["Input drift", "PSI of each model input", f">= {config.PSI_WARN:.2f}", f">= {config.PSI_ALERT:.2f}", "check the data source; refit if it persists"],
        ["Demand error", "monthly MAE vs 0.07 kW tested", f">= {config.ERROR_WARN_RATIO:.1f}x", f">= {config.ERROR_ALERT_RATIO:.1f}x", "refit on the newest months; check sensors"],
        ["Heatwave safeguard", "hours above 40 °C using Ridge", "any", "error >= 2x 0.08 kW", "extend the Ridge range; review features"],
        ["Failure ranking", "failures found in top 10% (tested 94%)", "< 85%", "< 70%", "refit with the new failures; review sensors"],
        ["Churn calibration", "predicted vs actual lapse rate", "gap > 0.10", "gap > 0.20", "refit on the latest cutoff; re-set the threshold"],
    ], columns=["Check", "What is measured", "Warning at", "Alert at", "Action"])
    fig = table_figure(df, "Alert rules (the first two rows are implemented in monitoring.py; the rest are applied by hand)", col_widths=[0.14, 0.27, 0.12, 0.17, 0.30], font=9, figsize=(14, 3.6), row_height=0.7)
    save(fig, "s10_04_alert_rules.png")


ALL = [lifecycle_owners, product_overview, data_sources, churn_table_timeline, split_designs, feature_pipeline, model_families, extrapolation_toy,
       training_protocol, architecture, safeguard_logic, monitoring_loop, retraining_timeline, alert_rules]
