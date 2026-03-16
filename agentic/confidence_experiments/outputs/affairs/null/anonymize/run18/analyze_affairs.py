import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Outcome: any extramarital affairs in past year
    df["affair_any"] = (df["feature2"] > 0).astype(int)

    # Key predictor: children in marriage (yes/no)
    # Code as 1 = children present, 0 = no children
    df["children"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Basic descriptive statistics by children status
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affairs=("feature2", "mean"),
            prop_any=("affair_any", "mean"),
            n=("feature2", "size"),
        )
        .reset_index()
    )

    # Logistic regression for having any affair, controlling for available covariates.
    # Use feature3 (gender) as a categorical factor; other predictors numeric.
    formula = (
        "affair_any ~ children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)

    # Extract effect of children from model
    params = result.params
    b_children = params["children"]
    se_children = result.bse["children"]
    p_children = result.pvalues["children"]
    or_children = float(np.exp(b_children))

    # 95% confidence interval for odds ratio
    z = 1.96
    ci_low = float(np.exp(b_children - z * se_children))
    ci_high = float(np.exp(b_children + z * se_children))

    # Determine direction and strength of evidence that children DECREASE affairs
    # Negative coefficient (OR < 1) supports a "Yes" answer.
    if p_children < 0.001 and or_children < 0.7 and ci_high < 1.0:
        response_score = 95
        qualitative = "strong evidence that having children is associated with fewer extramarital affairs"
        yes_no = "Yes"
    elif p_children < 0.01 and or_children < 0.8 and ci_high < 1.0:
        response_score = 85
        qualitative = "clear evidence that having children is associated with fewer extramarital affairs"
        yes_no = "Yes"
    elif p_children < 0.05 and or_children < 1.0 and ci_high <= 1.05:
        response_score = 70
        qualitative = "moderate evidence that having children is associated with somewhat fewer extramarital affairs"
        yes_no = "Yes"
    elif p_children < 0.05 and or_children < 1.0:
        response_score = 60
        qualitative = (
            "statistically significant but modest evidence that having children is associated with fewer extramarital affairs"
        )
        yes_no = "Yes"
    elif p_children >= 0.05 and 0.95 <= or_children <= 1.05:
        response_score = 50
        qualitative = "no clear evidence that having children meaningfully changes engagement in extramarital affairs"
        yes_no = "No"
    elif p_children < 0.05 and or_children > 1.0 and ci_low > 1.0:
        response_score = 5
        qualitative = "strong evidence that having children is associated with more (not fewer) extramarital affairs"
        yes_no = "No"
    elif p_children < 0.05 and or_children > 1.0:
        response_score = 15
        qualitative = "evidence that having children is associated with slightly more (not fewer) extramarital affairs"
        yes_no = "No"
    else:
        response_score = 40
        qualitative = (
            "weak and statistically inconclusive evidence about whether having children changes engagement in extramarital affairs"
        )
        yes_no = "No"

    response_score = int(max(0, min(100, response_score)))

    # Build explanation string with key numerical results
    # Map children code back to labels for clarity
    label_map = {0: "no children", 1: "children present"}
    group_lines = []
    for _, row in group_stats.iterrows():
        label = label_map.get(int(row["children"]), str(row["children"]))
        line = (
            f"For couples with {label} (n={int(row['n'])}), the mean affairs-code is "
            f"{row['mean_affairs']:.3f}, and the proportion reporting any affair is "
            f"{row['prop_any']:.3f}."
        )
        group_lines.append(line)

    explanation = (
        f"Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"Answer on a Yes/No scale: {yes_no} (response score {response_score} on a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes').\n\n"
        f"Evidence summary:\n"
        f"- Outcome variable: an indicator for having any extramarital sexual intercourse during the past year, derived from the affairs frequency code (feature2 > 0).\n"
        f"- Key predictor: presence of children in the marriage (feature6, coded 1 for 'yes' and 0 for 'no').\n"
        f"- Controls: gender (feature3), age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), and self-rated marriage quality (feature10).\n\n"
        f"Descriptive statistics by children status:\n"
        f"{chr(10).join(group_lines)}\n\n"
        f"Logistic regression results (probability of any affair):\n"
        f"- Children coefficient (log-odds): {b_children:.3f}\n"
        f"- Odds ratio for having children vs no children: {or_children:.3f}\n"
        f"- 95% confidence interval for odds ratio: [{ci_low:.3f}, {ci_high:.3f}]\n"
        f"- p-value for children effect: {p_children:.4g}\n\n"
        f"Interpretation: {qualitative}. This assessment accounts for the direction and statistical significance of the "
        f"children coefficient in a multivariable logistic regression model as well as the descriptive differences in "
        f"affair rates between couples with and without children."
    )

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

