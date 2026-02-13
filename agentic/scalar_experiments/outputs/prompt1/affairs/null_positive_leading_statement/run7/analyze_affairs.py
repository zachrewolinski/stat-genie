import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Define binary indicator for any extramarital affairs
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children status
    grp = (
        df.groupby("children")
        .agg(
            n=("any_affair", "size"),
            mean_affairs=("affairs", "mean"),
            std_affairs=("affairs", "std"),
            prop_any_affair=("any_affair", "mean"),
        )
        .reset_index()
    )

    # Logistic regression of any affair on children only
    logit_children = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)

    # Logistic regression including common controls
    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_full = smf.logit(formula, data=df).fit(disp=False)

    # Extract key results for children effect
    children_param = [p for p in logit_full.params.index if "C(children)" in p]
    if children_param:
        param_name = children_param[0]
        coef = float(logit_full.params[param_name])
        se = float(logit_full.bse[param_name])
        pval = float(logit_full.pvalues[param_name])
        odds_ratio = float(np.exp(coef))
    else:
        param_name = "C(children)[T.yes]"
        coef = se = pval = odds_ratio = float("nan")

    # Save a compact JSON summary for human interpretation
    summary = {
        "group_summary": grp.to_dict(orient="records"),
        "logit_children": {
            "params": logit_children.params.to_dict(),
            "pvalues": logit_children.pvalues.to_dict(),
        },
        "logit_full": {
            "children_param_name": param_name,
            "coef": coef,
            "se": se,
            "pval": pval,
            "odds_ratio": odds_ratio,
        },
    }

    with open("analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

