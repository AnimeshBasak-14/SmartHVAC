# 06. Questions we expect, with short answers

Everyone in the group should be able to answer all of these. Numbers are from `hvac_ml/outputs/results/RESULTS_SUMMARY.md`
(mean over 5 seeds).

## The product and the problem

**Q1. What are you building?**
A dashboard for an HVAC service provider with three tools: a Demand Planner (expected power draw by hour), a Failure Radar (which machines to
inspect first) and a Retention List (which active customers are about to go quiet). Each tool is a model plus a ranked list a manager can act on.

**Q2. How do you know it is better than what a company does today?**
We compare every model with a rule of thumb for the same task: the average load for that hour of the day, "inspect the most worn tool first",
and "call the customers who have not bought for the longest time". The models beat these rules in all three pillars (see Q10 to Q12).

**Q3. Is the data real?**
Partly. Online Retail II is real transaction data from a UK shop. AI4I 2020 is synthetic data built to resemble industrial machines. The Kaggle HVAC data is very regular and
probably simulated. All three are stand-ins for HVAC service data, and we say so in the report.

## Method

**Q4. Why Ridge and XGBoost?**
They are two different kinds of model that everyone can explain: a linear model with a penalty that is stable and extrapolates, and a tree ensemble that captures
non-linear patterns. Comparing them shows when complexity helps and when it does not.

**Q5. Why did you add linear regression and Random Forest?**
As baselines. Ordinary linear / logistic regression is the standard statistical baseline and Random Forest is the standard classic machine-learning baseline.
If our models do not beat them, the extra work is not justified.

**Q6. Why not accuracy for failure and churn?**
Only 3.4% of machines fail. A model that always says "no failure" is 96.6% accurate and useless. We use PR-AUC (a random guess scores 0.034), and the share of failures found in the
10% of machines we would inspect first.

**Q7. How do you avoid leakage in the churn task?**
Customers are split first. Features use only invoices before a cutoff date and the target uses only the 90 days after it. Training windows end before the test cutoff.
Four automatic tests check this, including one that deletes all invoices after the cutoff and confirms that no feature changes.

**Q8. Why five seeds?**
A single split can be lucky. We repeat with five seeds and report mean ± standard deviation. If two models are within one standard deviation, we call it a tie and pick
the simpler model.

**Q9. Did you tune the models on the test data?**
No. Model settings are fixed in `config.py`. The Ridge penalty and the alert threshold are chosen on training data only (cross-validation and out-of-fold scores).

## Results

**Q10. Pillar 1: how good is the demand forecast?**
On hours it never saw, XGBoost's average error is 0.07 to 0.08 kW (average load is 1.9 kW), against about 1.0 kW for the "average by hour" rule. The linear model is
0.10 to 0.12 kW.

**Q11. Pillar 2: how good is the failure model?**
XGBoost reaches a PR-AUC of 0.89 and Random Forest 0.91 (they tie); the rule "most worn tool first" scores 0.12 and a random guess 0.034. Inspecting the top 10% of machines
finds about 94% of failures with XGBoost, against 36% with the wear rule. Ridge and logistic regression reach only 0.49, because failures depend on thresholds that a straight line cannot represent.

**Q12. Pillar 3: how good is the churn model?**
ROC-AUC of 0.77 to 0.78 for all models (Ridge, XGBoost, Random Forest and logistic regression), against 0.61 for the recency rule. Contacting the 10% of active customers with the highest risk reaches about 20% of those who lapse, twice as many as random contact.
Churn is hard to predict from purchase history alone, and no model did better than plain logistic regression.

**Q13. Where does XGBoost fail?**
In the heatwave test. Trained below 40 °C, XGBoost's error above 40 °C is 0.34 kW, and its predictions go flat (about 3.1 kW) while the real demand keeps climbing to 4 kW. Trees cannot predict outside the range
they have seen. The linear model on the temperature gap is off by only 0.08 kW. The product therefore switches to the linear model whenever the temperature is above 40 °C.

**Q14. Does feature engineering matter?**
For the linear models, a lot: demand error falls from 0.18-0.24 kW to 0.10-0.12 kW, and in the heatwave from 0.47 kW to 0.08 kW. For XGBoost, a lot for failure (PR-AUC 0.79 to 0.89) and almost nothing for demand.

**Q15. Why is Ridge almost identical to plain linear regression?**
We have thousands of rows and at most 16 inputs, so the penalty has little to correct. That is expected.

## Limits and future work

**Q16. What would change with real company data?**
The absolute numbers would change, and probably get worse. The method (honest splits, simple baselines, tie rule) would stay the same. Real data would also let us define churn from
service-contract renewals instead of shop purchases.

**Q17. Why did you remove the graph neural network?**
An earlier version used a GraphSAGE model on synthetic graph data. On held-out data it did not beat a ridge model with product terms, because in that data each label depended only on the node's own
features. We could not show a benefit, so we kept the simpler models and moved the GNN to future work (`05_FUTURE_WORK.md`).

**Q18. Where might a GNN help?**
Possibly on churn and cross-selling, using a customer-product graph, and on demand across several buildings that share a grid. This is a hypothesis; we have not tested it. A GNN would not fix the
heatwave failure, which comes from how trees extrapolate.

**Q19. What is the biggest weakness of the project?**
The datasets are proxies. The scores say the method works; they do not say how much a real HVAC company would gain.

## Deployment and monitoring

**Q20. How does the dashboard decide between XGBoost and Ridge for demand?**
For each hour it checks the outdoor temperature. At 40 °C or above, or outside the range seen in training (6.1 to 50.4 °C), it uses Ridge. Otherwise it uses XGBoost. It shows both curves and says how many hours used the safeguard.

**Q21. The tie rule picks Random Forest for failure and logistic regression for churn. Why does the dashboard use XGBoost and Ridge?**
Both choices are inside the tie band: XGBoost scores 0.893 against 0.912 for Random Forest on failure, and Ridge scores 0.777 against 0.775 for logistic regression on churn, with standard deviations of 0.04 and 0.03.
We kept XGBoost and Ridge because they are the two models the group set out to compare. The strict rule would swap them, and the stage 6 document says so.

**Q22. What do you monitor after deployment?**
Three things: input drift (Population Stability Index and the Kolmogorov-Smirnov test, warning at 0.10 and alert at 0.25), the monthly demand error against the tested 0.065 kW (warning at 1.5 times, alert at 2 times), and the predicted against actual positive rate for failure and churn.

**Q23. Does a drift alert mean we must retrain?**
No. PSI tells us the inputs moved, not that the model got worse. Months 10 to 12 alert against months 1 to 9 only because of the season. We retrain when the error check fires, and we use the safeguard when temperatures leave the training range.

**Q24. When do you retrain?**
Demand before and after the cooling season, churn each quarter with a new cutoff, failure after 100 new logged failures.

**Q25. Are the lists in the dashboard test results?**
No. The final models were trained on all rows, so the Failure Radar and Retention List examples are illustrations. The honest scores are in stage 8 and in `RESULTS_SUMMARY.md`.

**Q26. What did you do about the odd quirks in the data?**
We kept them and reported them in stage 3: monthly temperature steps in the HVAC file, 368 idle hours at exactly 0.10 kW, nine failures with no cause flag, and 100 churn rows on the 90-day boundary.
