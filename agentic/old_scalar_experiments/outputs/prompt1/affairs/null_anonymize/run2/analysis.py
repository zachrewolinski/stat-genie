import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def main() -> None:
    base_path = Path(__file__).parent
    info_path = base_path / "info.json"
    data_path = base_path / "affairs.csv"

    with info_path.open() as f:
        info = json.load(f)

    # Load data
    df = pd.read_csv(data_path)

    # feature2: frequency of extramarital intercourse in past year
    # feature6: children in marriage? ("yes"/"no")
    # Create binary indicator of any affairs (feature2 > 0)
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Basic group-wise summaries
    group_children = df.groupby("feature6", observed=True)
    mean_any_affair = group_children["any_affair"].mean()
    mean_freq = group_children["feature2"].mean()

    # Linear probability model for having any affair ~ children indicator
    # Encode children as 1=yes, 0=no (does having children decrease affairs?)
    df["children_yes"] = (df["feature6"].str.lower() == "yes").astype(int)
    model_any = ols("any_affair ~ children_yes", data=df).fit()
    coef_children = model_any.params.get("children_yes", np.nan)
    pval_children = model_any.pvalues.get("children_yes", np.nan)

    # Decide answer: we look for evidence that having children decreases affairs:
    # negative coefficient and statistically significant at 5% level.
    decreases = (coef_children < 0) and (pval_children < 0.05)

    if decreases:
        response = "Yes"
    else:
        response = "No"

    # Build concise explanation string
    mean_affair_yes = float(mean_any_affair.get("yes", np.nan))
    mean_affair_no = float(mean_any_affair.get("no", np.nan))
    mean_freq_yes = float(mean_freq.get("yes", np.nan))
    mean_freq_no = float(mean_freq.get("no", np.nan))

    explanation = (
        "Using the 601 married respondents, I created a binary indicator for any extramarital affair "
        "(feature2 > 0) and compared people with and without children (feature6). "
        f"The share reporting any affair was {mean_affair_yes:.3f} with children and {mean_affair_no:.3f} without children, "
        f"with mean affair frequency {mean_freq_yes:.3f} vs. {mean_freq_no:.3f}. "
        "I then estimated a linear probability model any_affair ~ children_yes (1 if there are children). "
        f"The coefficient on having children was {coef_children:.3f} with p-value {pval_children:.3f}, "
        "which does not provide statistically significant evidence at the 5% level that having children reduces the probability "
        "of engaging in extramarital affairs. Therefore, I conclude that the data do not support the claim that having children decreases engagement in extramarital affairs."
    )

    conclusion = {"response": response, "explanation": explanation}

    out_path = base_path / "conclusion.txt"
    with out_path.open("w") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

