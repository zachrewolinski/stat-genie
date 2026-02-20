import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.weightstats import ttest_ind
except ImportError as exc:  # Fallback is unlikely needed but keeps script robust
    raise SystemExit(f"Required statsmodels package not available: {exc}")


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Affair engagement: any non‑zero value in feature2 is treated as engagement.
    df["affair_any"] = (df["feature2"] > 0).astype(int)
    # Children indicator: 1 = yes, 0 = no
    df["children_ind"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Basic group summaries
    grp_affair = df.groupby("children_ind")["affair_any"].agg(["mean", "sum", "count"])
    grp_freq = df.groupby("children_ind")["feature2"].mean()

    # Logistic regression for probability of any affair, controlling for key covariates
    # feature3: gender (categorical)
    # feature4: age, feature5: years married, feature7–10: other numeric covariates
    formula = (
        "affair_any ~ children_ind + C(feature3) + feature4 + feature5 "
        "+ feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)

    coef_children = float(logit_model.params["children_ind"])
    pval_children = float(logit_model.pvalues["children_ind"])
    odds_ratio = float(np.exp(coef_children))

    # Marginal difference in predicted probability when toggling children status
    df_child = df.copy()
    df_child["children_ind"] = 1
    df_no_child = df.copy()
    df_no_child["children_ind"] = 0
    prob_child = float(logit_model.predict(df_child).mean())
    prob_no_child = float(logit_model.predict(df_no_child).mean())
    prob_diff = prob_child - prob_no_child  # negative => children lowers probability

    # Difference in average frequency of affairs (feature2) with Welch t‑test
    freq_children = df.loc[df["children_ind"] == 1, "feature2"]
    freq_no_children = df.loc[df["children_ind"] == 0, "feature2"]
    t_stat, pval_ttest, _ = ttest_ind(freq_children, freq_no_children, usevar="unequal")
    t_stat = float(t_stat)
    pval_ttest = float(pval_ttest)

    # Determine direction based on multiple pieces of evidence
    mean_affair_with_children = float(grp_affair.loc[1, "mean"])
    mean_affair_without_children = float(grp_affair.loc[0, "mean"])
    mean_freq_with_children = float(grp_freq.loc[1])
    mean_freq_without_children = float(grp_freq.loc[0])

    decreases_prob = (
        coef_children < 0
        and prob_child < prob_no_child
        and mean_affair_with_children <= mean_affair_without_children
    )
    decreases_freq = mean_freq_with_children <= mean_freq_without_children

    # Decide answer: do children decrease engagement?
    if decreases_prob and decreases_freq and pval_children < 0.05 and pval_ttest < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Strength of effect: combine p‑values and effect sizes crudely
    # Start from a base depending on significance, then adjust by odds ratio distance from 1
    if pval_children < 0.001:
        base_strength = 90
    elif pval_children < 0.01:
        base_strength = 80
    elif pval_children < 0.05:
        base_strength = 70
    elif pval_children < 0.1:
        base_strength = 55
    else:
        base_strength = 40

    odds_distance = abs(np.log(odds_ratio))
    if odds_distance > 0.7:
        base_strength += 10
    elif odds_distance < 0.1:
        base_strength -= 10

    base_strength = max(0, min(100, base_strength))

    # Confidence: how sure we are about the Yes/No call.
    # Use both p‑values; if they disagree on significance, reduce confidence.
    if pval_children < 0.001 and pval_ttest < 0.001:
        confidence = 90
    elif pval_children < 0.01 and pval_ttest < 0.01:
        confidence = 85
    elif pval_children < 0.05 and pval_ttest < 0.05:
        confidence = 75
    elif pval_children < 0.1 or pval_ttest < 0.1:
        confidence = 60
    else:
        confidence = 50

    # If different evidence pieces disagree with the final Yes/No direction,
    # soften strength and confidence.
    if response == "Yes":
        if not (decreases_prob and decreases_freq):
            base_strength = min(base_strength, 60)
            confidence = min(confidence, 60)
    else:  # response == "No"
        if decreases_prob or decreases_freq:
            base_strength = min(base_strength, 60)
            confidence = min(confidence, 65)

    strength = int(round(base_strength))
    confidence = int(round(confidence))

    # Build explanation text with key numerical evidence
    expl = []
    expl.append(
        "I analyzed the 1969 Psychology Today marriage survey sample "
        "using the local affairs.csv data (601 married individuals)."
    )
    expl.append(
        f" I defined engagement in extramarital affairs as any non-zero value of feature2 "
        f"(past-year extramarital intercourse), and used feature6 to indicate whether "
        f"there are children in the marriage."
    )
    expl.append(
        f" The proportion with any affair was "
        f"{mean_affair_without_children:.3f} without children and "
        f"{mean_affair_with_children:.3f} with children; "
        f"the mean affair frequency (feature2) was "
        f"{mean_freq_without_children:.3f} without children vs "
        f"{mean_freq_with_children:.3f} with children."
    )
    expl.append(
        f" A logistic regression of any affair on children (controlling for gender, age, "
        f"years married, and other covariates) yielded a children coefficient of "
        f"{coef_children:.3f} (odds ratio {odds_ratio:.3f}, p = {pval_children:.3f}), "
        f"and the Welch t-test comparing affair frequency between groups gave "
        f"t = {t_stat:.3f}, p = {pval_ttest:.3f}."
    )
    if response == "Yes":
        expl.append(
            " Taken together, these results indicate that having children is associated "
            "with a lower level of extramarital affair engagement, and this pattern is "
            "supported by the regression and group comparisons."
        )
    else:
        expl.append(
            " Overall, the direction and statistical evidence do not support the claim "
            "that having children meaningfully reduces engagement in extramarital affairs; "
            "any differences appear small and/or statistically weak given this sample."
        )

    explanation = "".join(expl)

    result = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

