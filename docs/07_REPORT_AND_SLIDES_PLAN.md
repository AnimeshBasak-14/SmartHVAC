# 07. Report and slides plan

Submission rules to satisfy: cover page with group number, title, names and enrollment numbers; **first page with each member's name,
contribution (1 or 2 sentences) and the report pages where it appears (no evaluation without it)**; no code or code screenshots; every
figure explained in words; a clear recommendation in the conclusion based on the analysis; the data source links; a help file; slides.

## Report outline

| # | Section | Content | Figures |
| :--- | :--- | :--- | :--- |
| 0 | Cover page | Group number [to fill], title, five names and enrollment numbers (Kriti Vajpayee's number [to fill]) | |
| 0 | Contributions page | One row per member: name, 1 or 2 sentences, page numbers | |
| 1 | Introduction and problem | HVAC service providers, the three decisions, the product (`01`) | |
| 2 | Data | The three datasets, links, cleaning, how the churn table is built, limits (`02`) | 1, 2, 3 |
| 3 | Method | Models and baselines, features, the four demand tests, the churn split, metrics, the tie rule, leakage checks (`03`) | block diagram (draw by hand) |
| 4 | Results: demand | Table, figures 4 and 5 with explanation | 4, 5 |
| 5 | Results: failure | Table, figures 6 and 7 | 6, 7 |
| 6 | Results: churn | Table, figures 8 and 9 | 8, 9 |
| 7 | The product | Screenshots of the three dashboard tools with a short description (screenshots of the result are allowed, not of code) | dashboard screenshots |
| 8 | Where XGBoost helps and where it fails | Figure 10 and the heatwave story | 10 |
| 9 | Conclusion and recommendation | See below | |
| 10 | Limits and future work | Proxy data; one building; where a graph neural network may help (`05`) | |
| | References | AI4I 2020 (Matzka, 2020); Online Retail II (Chen, 2019); Chen and Guestrin, "XGBoost" (2016); Hoerl and Kennard, "Ridge regression" (1970); Breiman, "Random Forests" (2001) | |

Under every figure write: what is plotted, what we see (with numbers from `04_RESULTS.md`), and what it means for the service provider.
The ready-made explanations in `04_RESULTS.md` can be shortened and pasted.

## Conclusion and recommendation (draft)

1. **Demand Planner.** Use XGBoost for normal weather (error 0.07 to 0.08 kW) and the Ridge model above 40 °C, where XGBoost fails (0.34 kW against 0.08 kW).
2. **Failure Radar.** Use XGBoost or Random Forest with engineered features (PR-AUC about 0.9). Inspecting the top 10% of machines finds about 94% of failures, against 37% for the "most worn tool" rule. Do not use linear models here.
3. **Retention List.** Use the Ridge (penalised logistic) model: all models tie, so take the simplest. It reaches about twice as many lapsing customers as random contact, so treat the list as a way to prioritise calls, not as a prediction of who will leave.
4. **Caveat.** These conclusions rest on public stand-in datasets. The next step is to repeat the same tests on a service company's own data.

## Slides (about 12, every member presents)

| # | Slide | Content | Suggested presenter |
| :--- | :--- | :--- | :--- |
| 1 | Title | Title, group, names | |
| 2 | The problem | Three decisions of an HVAC service manager | |
| 3 | The product | HVAC Service Planner: three tools (dashboard screenshot) | |
| 4 | Data | Three datasets in one table, with links | |
| 5 | How we test | Unseen data, five seeds, baselines and rules of thumb, no leakage | |
| 6 | Demand results | Figure 4 | |
| 7 | The heatwave | Figure 5: where XGBoost fails | |
| 8 | Failure results | Figure 6 | |
| 9 | Churn results | Figure 8 | |
| 10 | Which model where | Figure 10 and the recommendation table | |
| 11 | Live demo | Dashboard: a hot day, the inspection list, the call list | |
| 12 | Limits and future work | Stand-in data; where a graph neural network may help | |

Assign presenters in the group chat, and let every member explain at least one figure and one limit. Rehearse once with a timer.
Have screenshots of the dashboard on the slides in case the live demo fails.

## Checklist

- [ ] Report PDF with cover page, contributions page (names, 1 or 2 sentences, page numbers), explanations under every figure, recommendation, references, no code
- [ ] Slides
- [ ] Code: the project folder without `.venv`
- [ ] Data source links (in `README.md`, `HELP.md` and the report)
- [ ] Help file (`HELP.md`)
- [ ] Fill in: group number, Kriti Vajpayee's enrollment number, check the Kaggle link opens
