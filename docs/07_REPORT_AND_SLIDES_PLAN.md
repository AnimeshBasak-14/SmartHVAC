# 07. Report and slides: what we built and the submission checklist

Submission rules to satisfy: cover page with group number, title, names and enrollment numbers; **first page with each member's name,
contribution and the report pages where it appears**; no code or code screenshots; every figure explained in words; a clear recommendation in the conclusion based on the analysis;
the data source links; a help file; slides.

## The report (`report/HVAC_Service_Planner_Report.docx`, with a PDF copy)

| Part | Where |
| :--- | :--- |
| Cover page: institute, title, course, Group 1, five names and enrollment numbers, guide | page 1 |
| Contribution of the group members, with page numbers | page 2 |
| Abstract, table of contents, list of figures | pages 3 to 9 |
| Chapters 1 to 10: the ten lifecycle stages, one author per stage | `docs/lifecycle/` |
| Chapter 11: results and discussion | `report/src/11_Results_and_Discussion.md` |
| Chapter 12: conclusion and recommendation | `report/src/12_Conclusion_and_Recommendation.md` |
| Chapter 13: limitations and future work | `report/src/13_Limitations_and_Future_Work.md` |
| References and data sources | `report/src/14_References.md` |

Every figure has a caption and a paragraph that explains what is plotted, what we see and what it means. The text of chapters 1 to 10 is the same as the documents in `docs/lifecycle/`,
without the "where to find things in the repository" sections, so the report has no code references.

To rebuild after editing any document: `python report/build_report.py`, then run `report/finish_docx.ps1` in PowerShell (it needs Word; it updates the table of contents and the page numbers and writes the PDF).

## The slides (`report/HVAC_Service_Planner_Presentation.pptx`, with a PDF copy)

61 slides, 16:9. Every slide has a tag with the stage and the presenter. Most slides are one full-slide figure or one table, and the speaker notes hold two or three sentences to say.
Rebuild with `python report/build_slides.py`; `report/export_slides.ps1` writes PNG images and a PDF through PowerPoint.

| Slides | Stage | Presenter |
| :--- | :--- | :--- |
| 1 to 5 | Title, lifecycle map, product, task table, rule of thumb | Kriti Vajpayee |
| 6 to 10 | Data collection | Kriti Vajpayee |
| 11 to 14 | Data cleaning and preprocessing | Dhruv Aggarwal |
| 15 to 20 | Exploratory data analysis | Dhruv Aggarwal |
| 21 to 25 | Feature engineering | Ankit Kumar |
| 26 to 29 | Model selection | Ankit Kumar |
| 30 to 34 | Model training | Animesh Basak |
| 35 to 46 | Model evaluation and tuning | Animesh Basak |
| 47 to 51 | Model deployment | Sourav Kumar Boro |
| 52 to 56 | Monitoring and maintenance | Sourav Kumar Boro |
| 57 to 61 | Recommendation, limits, conclusion, references | Whole group |

The deck is long on purpose. For a 15 to 20 minute talk, pick about 25 slides and keep the rest as backup for questions.

## Conclusion and recommendation (as in the report)

1. **Failure Radar first.** Use XGBoost or Random Forest to rank machines (PR-AUC about 0.9). Inspecting the top 10% finds about 94% of failures, against 36% for the "most worn tool" rule.
2. **Demand Planner with the heatwave safeguard.** XGBoost in ordinary weather (error 0.07 to 0.08 kW), Ridge at 40 °C and above (0.08 kW, against 0.34 kW for XGBoost).
3. **Retention List as a low-cost aid.** Use the Ridge (penalised logistic) model. All models tie, and the list reaches about twice as many lapsing customers as random contact.
4. **Monitoring from day one**, and a four-week pilot on the company's own data before relying on any score.

## Checklist

- [x] Report with cover page, contributions page (names, contribution, page numbers), explanations under every figure, recommendation, references, no code
- [x] Slides
- [ ] Read the contribution text on page 2 of the report and edit it if it does not match what each member did (it is our draft)
- [ ] Check the Kaggle link opens and note its licence
- [ ] Code: the project folder without `.venv`
- [ ] Data source links (in `README.md`, `HELP.md` and the report)
- [x] Help file (`HELP.md`)
- [ ] Delete the older `SmartHVAC_*` files in `report/` if they are no longer needed
