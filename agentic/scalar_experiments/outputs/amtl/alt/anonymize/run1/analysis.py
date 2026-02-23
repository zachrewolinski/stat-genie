import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: drop rows with missing key fields and ensure positive trial counts
    df = df.dropna(subset=["feature1", "feature3", "feature4", "feature5", "feature7", "feature8"])
    df = df[df["feature4"] > 0].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth (successes / trials)
    df["prop_missing"] = df["feature3"] / df["feature4"]

    # Descriptive statistics: AMTL rates by genus
    genus_summary = (
        df.groupby("feature8")
        .agg(
            n_rows=("feature3", "size"),
            n_missing=("feature3", "sum"),
            n_sockets=("feature4", "sum"),
        )
        .reset_index()
    )
    genus_summary["amtl_rate"] = genus_summary["n_missing"] / genus_summary["n_sockets"]

    # Aggregate humans vs non-humans
    df["is_nonhuman"] = 1 - df["is_human"]
    human_mask = df["is_human"] == 1
    nonhuman_mask = df["is_human"] == 0

    human_missing = df.loc[human_mask, "feature3"].sum()
    human_sockets = df.loc[human_mask, "feature4"].sum()
    nonhuman_missing = df.loc[nonhuman_mask, "feature3"].sum()
    nonhuman_sockets = df.loc[nonhuman_mask, "feature4"].sum()

    human_rate = float(human_missing / human_sockets) if human_sockets > 0 else np.nan
    nonhuman_rate = float(nonhuman_missing / nonhuman_sockets) if nonhuman_sockets > 0 else np.nan

    # Binomial regression: prop_missing with binomial family and socket counts as frequency weights
    # Adjusting for age at death (feature5), sex estimate (feature7), and tooth class (feature1).
    model = smf.glm(
        "prop_missing ~ is_human + feature5 + feature7 + C(feature1)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()

    # Extract human effect
    coef_human = float(result.params["is_human"])
    pval_human = float(result.pvalues["is_human"])
    ci_lower, ci_upper = result.conf_int().loc["is_human"].tolist()
    ci_lower = float(ci_lower)
    ci_upper = float(ci_upper)
    odds_ratio = float(np.exp(coef_human))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Predicted average AMTL rate if everyone were human vs non-human,
    # averaging over the observed distribution of age, sex, and tooth class.
    df_human = df.copy()
    df_human["is_human"] = 1
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0

    pred_human = result.predict(df_human)
    pred_nonhuman = result.predict(df_nonhuman)

    mean_pred_human = float(np.average(pred_human, weights=df["feature4"]))
    mean_pred_nonhuman = float(np.average(pred_nonhuman, weights=df["feature4"]))
    abs_diff = float(mean_pred_human - mean_pred_nonhuman)
    rel_diff = float(abs_diff / mean_pred_nonhuman) if mean_pred_nonhuman > 0 else np.nan

    # Map evidence to 0–100 response scale where 0 = strong "No", 100 = strong "Yes"
    # "Yes" corresponds to humans having higher AMTL rates than non-humans.
    if pval_human < 0.001:
        yes_score = 95
        no_score = 5
    elif pval_human < 0.01:
        yes_score = 90
        no_score = 10
    elif pval_human < 0.05:
        yes_score = 80
        no_score = 20
    else:
        # Lack of statistically significant evidence for a difference
        response_value = 30
        direction = "no statistically robust difference in AMTL rates between humans and non-human primates"
        answer_text = "no"
    if pval_human < 0.05:
        if coef_human > 0:
            response_value = yes_score
            direction = "modern humans have higher AMTL frequencies than non-human primates"
            answer_text = "yes"
        else:
            response_value = no_score
            direction = "modern humans have lower AMTL frequencies than non-human primates"
            answer_text = "no"

    # Build explanation text
    n_rows = int(df.shape[0])
    n_specimens = int(df["feature2"].nunique())

    genus_lines = []
    for _, row in genus_summary.iterrows():
        genus = row["feature8"]
        rate = row["amtl_rate"]
        n_miss = int(row["n_missing"])
        n_sock = int(row["n_sockets"])
        genus_lines.append(
            f"- {genus}: {n_miss}/{n_sock} teeth missing "
            f"({rate:.3f} proportion missing)."
        )
    genus_block = "\n".join(genus_lines)

    explanation = f"""
Research question
-----------------
Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL) than non-human primate genera (Pan, Papio, Pongo) after accounting for age, sex, and tooth class?

Data and outcome
----------------
The dataset contains {n_rows} rows representing tooth-class by individual combinations from approximately {n_specimens} specimens. For each row, we observe the number of missing teeth of a given class (feature3) and the number of observable sockets that could be scored (feature4), along with estimated age at death (feature5), a continuous sex estimate (feature7), tooth class (anterior/posterior/premolar; feature1), and genus (feature8).

Raw AMTL frequencies by genus
-----------------------------
Overall proportions of missing teeth (missing teeth / observable sockets) by genus are:
{genus_block}

Aggregating across genera, humans have a raw AMTL rate of {human_missing}/{human_sockets} teeth missing ({human_rate:.3f}), whereas non-human primates combined have {nonhuman_missing}/{nonhuman_sockets} teeth missing ({nonhuman_rate:.3f}).

Binomial regression model
-------------------------
To adjust for potential confounders, I fit a binomial regression model with a logit link where the response is the proportion of missing teeth and the binomial denominator is the number of observable sockets in each row. The model is:

  logit(AMTL proportion) ~ is_human + age at death (feature5) + sex estimate (feature7) + tooth class (C(feature1)),

using the number of observable sockets as frequency weights so that rows with more teeth contribute more information.

The coefficient for the human indicator (is_human = 1 for Homo sapiens, 0 for Pan/Papio/Pongo) is {coef_human:.3f}, corresponding to an odds ratio of {odds_ratio:.3f} (95% CI {or_ci_lower:.3f}–{or_ci_upper:.3f}), with p-value {pval_human:.3g}. This p-value quantifies the evidence for a difference in AMTL frequencies between humans and non-human primates after adjusting for age, sex, and tooth class.

Adjusted AMTL rates
-------------------
Using the fitted model, I computed adjusted AMTL rates by predicting the AMTL probability for each observation twice: once assuming it is human (is_human = 1) and once assuming it is non-human (is_human = 0), while keeping age, sex, and tooth class as observed. Averaging these predictions with socket counts as weights yields:

- Predicted AMTL proportion if all observations were humans: {mean_pred_human:.3f}
- Predicted AMTL proportion if all observations were non-human primates: {mean_pred_nonhuman:.3f}
- Absolute difference (human minus non-human): {abs_diff:.3f}
  (relative difference of {rel_diff:.1%} where defined).

Interpretation
--------------
Taken together, the raw genus-specific rates and the binomial regression results indicate that {direction}. The regression coefficient and odds ratio quantify both the direction and strength of this difference, and the p-value {pval_human:.3g} reflects the statistical evidence after adjusting for age, sex, and tooth class.

Conclusion and Likert-scale response
------------------------------------
Because the human indicator effect is {'statistically significant (p < 0.05)' if pval_human < 0.05 else 'not statistically significant (p ≥ 0.05)'} and its direction indicates that {direction}, I answer \"{answer_text}\" to the research question \"Do modern humans have higher AMTL frequencies than non-human primates after accounting for age, sex, and tooth class?\".

On a 0–100 Likert scale where 0 represents a strong \"No\" and 100 represents a strong \"Yes\", the evidence in this dataset corresponds to a response value of {response_value}.
""".strip()

    conclusion = {
        "response": int(response_value),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

