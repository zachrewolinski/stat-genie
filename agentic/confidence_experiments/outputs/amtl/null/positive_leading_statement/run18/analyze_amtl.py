import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_likert(coef: float, pval: float) -> int:
    """
    Map the human-effect coefficient and p-value to a 0–100 Likert score
    answering: \"Do humans have higher AMTL than non-human primates?\".
    """
    if np.isnan(coef) or np.isnan(pval):
        return 50

    # Strong evidence thresholds based mainly on p-value, with sign of coef
    if pval < 0.001:
        return 95 if coef > 0 else 5
    if pval < 0.01:
        return 90 if coef > 0 else 10
    if pval < 0.05:
        return 80 if coef > 0 else 20
    if pval < 0.10:
        return 65 if coef > 0 else 35
    # Little evidence either way: keep close to neutral but respect direction
    return 55 if coef > 0 else 45


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Descriptive statistics: observed AMTL rates by human vs non-human
    group_desc = (
        df.groupby("is_human")
        .agg(
            n_rows=("specimen", "size"),
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .reset_index()
    )
    group_desc["observed_rate"] = (
        group_desc["total_missing"] / group_desc["total_sockets"]
    )

    # Binomial regression: humans vs others, controlling for age, sex, tooth class.
    # Use a two-column binomial response (successes, failures) for numerical stability.
    endog = np.column_stack(
        [df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()]
    )

    exog = df[["is_human", "age", "prob_male", "tooth_class"]].copy()
    exog = pd.get_dummies(exog, columns=["tooth_class"], drop_first=True)
    exog = sm.add_constant(exog)

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()

    coef_human = float(result.params["is_human"])
    pval_human = float(result.pvalues["is_human"])
    or_human = float(np.exp(coef_human))
    ci_low, ci_high = result.conf_int().loc["is_human"].tolist()
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Model-based predicted AMTL rates for humans vs non-humans.
    df_pred = exog.copy()
    df_pred["predicted_rate"] = result.predict(df_pred)
    mean_pred_human = float(df_pred.loc[df["is_human"] == 1, "predicted_rate"].mean())
    mean_pred_nonhuman = float(
        df_pred.loc[df["is_human"] == 0, "predicted_rate"].mean()
    )

    # Observed rates from the group summary
    observed_human_rate = float(
        group_desc.loc[group_desc["is_human"] == 1, "observed_rate"].iloc[0]
    )
    observed_nonhuman_rate = float(
        group_desc.loc[group_desc["is_human"] == 0, "observed_rate"].iloc[0]
    )

    n_human_rows = int(group_desc.loc[group_desc["is_human"] == 1, "n_rows"].iloc[0])
    n_nonhuman_rows = int(group_desc.loc[group_desc["is_human"] == 0, "n_rows"].iloc[0])

    # Likert-scale response summarizing strength of evidence
    likert_score = compute_likert(coef_human, pval_human)

    explanation = f"""
Research question
-----------------
Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL)
than non-human primate genera (Pan, Pongo, Papio), after accounting for age, sex, and tooth class?

Data and outcome
----------------
- Dataset: 1,450 rows of tooth-class–level observations from modern humans and three non-human primate genera.
- Each row records the number of teeth missing (num_amtl) out of the number of observable sockets (sockets),
  along with estimated age at death, an estimate of sex (prob_male), tooth class (anterior/posterior/premolar),
  genus, and population.
- I model AMTL as a binomial outcome: the number of missing teeth out of the number of sockets.

Descriptive comparison
----------------------
- Human rows: {n_human_rows} observations.
- Non-human rows (Pan, Pongo, Papio): {n_nonhuman_rows} observations.
- Observed AMTL rate (num_amtl / sockets):
  * Humans: {observed_human_rate:.3f}
  * Non-humans: {observed_nonhuman_rate:.3f}

Binomial regression model
-------------------------
- Model: GLM with binomial family and logit link.
- Response: num_amtl / sockets, with binomial variability accounted for via weights = sockets.
- Predictors: indicator for humans vs non-humans (is_human), age, prob_male (sex estimate),
  and categorical tooth_class (anterior, posterior, premolar).

Key coefficient: effect of being human
--------------------------------------
- Log-odds coefficient for humans (is_human): {coef_human:.3f}
- Odds ratio (human vs non-human): {or_human:.3f}
- 95% CI for odds ratio: [{or_ci_low:.3f}, {or_ci_high:.3f}]
- p-value for human effect: {pval_human:.3g}

These results indicate that, after adjusting for age, sex, and tooth class, humans have
{'higher' if coef_human > 0 else 'lower'} odds of AMTL than non-human primates, and the
effect is {'statistically significant' if pval_human < 0.05 else 'not statistically significant at the 0.05 level'}.

Model-based predicted AMTL rates
--------------------------------
- Mean predicted AMTL rate for humans (from the fitted model): {mean_pred_human:.3f}
- Mean predicted AMTL rate for non-humans: {mean_pred_nonhuman:.3f}

Interpretation
--------------
- The descriptive comparison shows that humans have an observed AMTL rate of {observed_human_rate:.3f}
  compared with {observed_nonhuman_rate:.3f} for non-human primates.
- The binomial regression, which controls for age, sex, and tooth class, estimates an odds ratio of
  {or_human:.3f} for humans vs non-humans with a p-value of {pval_human:.3g}, indicating that this
  difference is {'unlikely' if pval_human < 0.05 else 'not clearly unlikely'} to be due to random sampling alone.
- The model-based predicted AMTL rates are {mean_pred_human:.3f} for humans and {mean_pred_nonhuman:.3f}
  for non-humans, reinforcing the descriptive pattern.

Conclusion
----------
Taken together, the descriptive statistics and the regression model provide
{'strong' if likert_score >= 90 else 'moderate' if likert_score >= 70 else 'limited' if likert_score >= 55 else 'little'}
evidence that modern humans have {'higher' if coef_human > 0 else 'lower'} frequencies of antemortem tooth loss
than non-human primates after accounting for age, sex, and tooth class. The Likert-scale score summarizes the
strength of this evidence on a 0–100 scale, where higher values correspond to a stronger \"Yes\" answer to the
research question.
""".strip()

    conclusion = {"response": int(likert_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
