"""
figs_summary.py - summary figures for the closing chapters: recommendation map, recommendation table, future-work map.
Each is a table drawn with wrapped text, so long sentences stay readable.
"""
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from hvac_ml.figs_common import BLUE, GREEN, NAVY, ORANGE, RED, GOLD, save


def wrapped_table(title, headers, rows, widths, wraps, name, colors=None, row_h=None, font=9.2, fig_w=14.0, status_col=None, status_colors=None):
    """Draw a table where every cell is wrapped to a width given in characters (wraps)."""
    lines = [[textwrap.wrap(str(c), w) or [""] for c, w in zip(r, wraps)] for r in rows]
    heights = [max(len(c) for c in r) for r in lines]
    unit = 0.19 * font / 9.2
    row_hs = [h * unit + 0.28 for h in heights]
    head_lines = [textwrap.wrap(h, w) for h, w in zip(headers, wraps)]
    head_h = max(len(h) for h in head_lines) * unit + 0.3
    total_h = head_h + sum(row_hs) + 0.7
    fig, ax = plt.subplots(figsize=(fig_w, total_h))
    ax.set_xlim(0, sum(widths))
    ax.set_ylim(0, total_h)
    ax.axis("off")
    ax.text(sum(widths) / 2, total_h - 0.25, title, ha="center", va="center", fontsize=12.5, fontweight="bold", color=NAVY)
    y = total_h - 0.6
    x = 0
    for w, h in zip(widths, head_lines):
        ax.add_patch(Rectangle((x, y - head_h), w, head_h, facecolor=NAVY, edgecolor="white", lw=1.2))
        ax.text(x + 0.15, y - head_h / 2, "\n".join(h), ha="left", va="center", fontsize=font, color="white", fontweight="bold")
        x += w
    y -= head_h
    for i, (cells, rh) in enumerate(zip(lines, row_hs)):
        x = 0
        for j, (w, cell) in enumerate(zip(widths, cells)):
            face = "#F4F6F8" if i % 2 else "white"
            if status_col is not None and j == status_col:
                face = status_colors[rows[i][j].split()[0]]
            ax.add_patch(Rectangle((x, y - rh), w, rh, facecolor=face, edgecolor="#CCCCCC", lw=0.8))
            bold = j == 0 or (status_col is not None and j == status_col)
            col = "white" if (status_col is not None and j == status_col) else "#222222"
            ax.text(x + 0.15, y - 0.14, "\n".join(cell), ha="left", va="top", fontsize=font, color=col, fontweight="bold" if bold else "normal", linespacing=1.3)
            x += w
        y -= rh
    save(fig, name, tight=False)


def recommendation_map():
    wrapped_table(
        "From result to decision: what each tool is, and what to trust it for",
        ["Tool", "Model in use and the evidence", "Trust it for", "Do not use it for"],
        [["Demand Planner", "XGBoost below 40 °C. Ridge at 40 °C and above, or outside 6.1 to 50.4 °C. Error 0.065 kW against 0.93 kW for the rule of thumb. Heatwave error 0.08 kW.",
          "Hourly demand curve, peak hour and energy for the day, to plan staff, stock and grid load.", "Planning with XGBoost alone in a heatwave. Other buildings or climates without retraining."],
         ["Failure Radar", "XGBoost (Random Forest ties). PR-AUC 0.89 against 0.12 for the rule. The top 10% of the list holds 94% of failures, against 36% for the rule.",
          "Ranking machines for inspection and working down the list as far as the team's capacity allows.", "Treating the 44% alert level as exact. Trusting the scores on HVAC units before retraining on company data."],
         ["Retention List", "Ridge (penalised logistic regression). ROC-AUC 0.78 against 0.61 for the rule. The top 10% holds 20% of lapsers: random 10%, rule 16%.",
          "Choosing whom to phone first when the call budget is small.", "Forecasting the exact number of customers who will lapse (the rate was off by 0.26 in a holiday window). Expecting more than twice the random hit rate."]],
        [1.7, 4.6, 3.4, 3.8], [14, 46, 32, 34], "s12_01_recommendation_map.png", font=9.6)


def recommendation_table():
    wrapped_table(
        "Recommendation to an HVAC service provider",
        ["Priority", "Tool", "What to do", "Evidence from held-out tests", "Condition before relying on it"],
        [["1", "Failure Radar", "Adopt. Rank machines by risk and inspect from the top of the list down, within the team's capacity.", "Top 10% of the list holds 94% of failures (rule: 36%). Inspecting 3% finds 80% (rule: 58%).", "Retrain on the company's own sensor and failure records."],
         ["2", "Demand Planner", "Adopt with the heatwave safeguard. XGBoost in ordinary weather, Ridge at 40 °C and above or outside the training range.", "Error 0.065 kW (rule: 0.93 kW). In the heatwave test 0.08 kW with Ridge against 0.34 kW with XGBoost.", "Retrain on the company's buildings before each cooling season. Never use XGBoost alone."],
         ["3", "Retention List", "Use as a low-cost aid. Keep the model simple (Ridge).", "Top 10% holds 20% of lapsers (random 10%, rule 16%). All models tie at about 0.78 ROC-AUC.", "Add product and contract data before paying for a more complex model."],
         ["4", "Monitoring", "Run from the first day. Check input drift monthly, demand error monthly and churn rate quarterly.", "Alert at PSI 0.25, error twice the tested 0.065 kW, or a rate gap of 0.20.", "Log every prediction and its outcome."]],
        [1.2, 1.8, 4.1, 4.2, 3.4], [9, 15, 38, 40, 32], "s12_02_recommendation_table.png", font=9.6)


def future_work_map():
    wrapped_table(
        "Where a graph neural network might help, and where it would not",
        ["Gap in our results", "Could a graph model help?", "Why it might", "Why it might not"],
        [["Churn: every model ties at about 0.78 ROC-AUC", "Maybe", "Customers who buy the same products may carry signal that one customer's row does not.", "The signal may not exist in purchase data. XGBoost with neighbour-averaged features may do as well."],
         ["Cross-selling a counter-seasonal appliance", "Maybe", "It is link prediction on a customer-product graph, which graph models learn directly.", "Item co-occurrence and matrix factorisation are strong and much simpler."],
         ["Demand across several buildings", "Maybe", "Sites on a shared grid or weather system influence each other, and a graph can say so.", "We have data for one building only, so there is nothing to test it on."],
         ["Heatwave: trees fail above 40 °C", "No", "No reason to expect it.", "Trees cannot predict above the range they saw. The fix is a linear model, which the dashboard already uses."],
         ["Failure: PR-AUC 0.89 to 0.91", "Not needed", "No gap to close.", "The machines in the data are independent rows with no links."]],
        [3.5, 2.0, 4.2, 4.5], [30, 12, 38, 42], "s13_01_future_work_map.png", font=9.6,
        status_col=1, status_colors={"Maybe": ORANGE, "No": RED, "Not": GREEN})


ALL = [recommendation_map, recommendation_table, future_work_map]
