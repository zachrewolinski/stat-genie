import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator of having any extramarital affairs in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children status
    group = df.groupby("children", observed=True)
    summary = {
        "n_by_children": group.size().to_dict(),
        "mean_any_affair_by_children": group["any_affair"].mean().to_dict(),
        "mean_affairs_by_children": group["affairs"].mean().to_dict(),
    }

    # Unadjusted logistic regression: any_affair ~ children
    unadj_model = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    unadj_params = unadj_model.params.to_dict()
    unadj_pvalues = unadj_model.pvalues.to_dict()

    # Compute predicted probabilities from the unadjusted model
    prob_no_children = float(
        unadj_model.predict(pd.DataFrame({"children": ["no"]}))[0]
    )
    prob_children = float(
        unadj_model.predict(pd.DataFrame({"children": ["yes"]}))[0]
    )

    # Adjusted logistic regression including standard covariates from the Fair dataset
    adj_formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )
    adj_model = smf.logit(adj_formula, data=df).fit(disp=False)
    adj_params = adj_model.params.to_dict()
    adj_pvalues = adj_model.pvalues.to_dict()

    # Predicted probabilities for a "typical" profile with and without children
    typical_row = {
        "gender": df["gender"].mode()[0],
        "age": df["age"].median(),
        "yearsmarried": df["yearsmarried"].median(),
        "children": "no",
        "religiousness": df["religiousness"].median(),
        "education": df["education"].median(),
        "occupation": df["occupation"].median(),
        "rating": df["rating"].median(),
    }
    typical_no_children = pd.DataFrame([typical_row])
    typical_with_children = typical_no_children.copy()
    typical_with_children["children"] = "yes"

    prob_no_children_adj = float(adj_model.predict(typical_no_children)[0])
    prob_children_adj = float(adj_model.predict(typical_with_children)[0])

    # Collect key numeric results in a JSON-serializable structure that we can inspect
    results = {
        "summary": summary,
        "unadjusted": {
            "params": unadj_params,
            "pvalues": unadj_pvalues,
            "prob_no_children": prob_no_children,
            "prob_children": prob_children,
        },
        "adjusted": {
            "params": adj_params,
            "pvalues": adj_pvalues,
            "prob_no_children_adj": prob_no_children_adj,
            "prob_children_adj": prob_children_adj,
        },
    }

    # Save detailed numeric results for manual inspection if desired.
    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

