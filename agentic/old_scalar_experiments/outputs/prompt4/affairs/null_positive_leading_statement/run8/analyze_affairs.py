import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Create a binary indicator of having any affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    return df


def summarize_children_effect(df: pd.DataFrame) -> dict:
    # Group-wise summaries
    group = df.groupby("children")
    summary = group["affairs"].agg(["mean", "std", "count"]).to_dict("index")

    # Proportion with any affairs by children status
    prop_any = group["has_affair"].mean().to_dict()

    return {
        "affairs_summary_by_children": summary,
        "prop_any_affair_by_children": prop_any,
    }


def fit_logit_model(df: pd.DataFrame):
    # Logistic regression for having any affair vs. covariates including children
    # Use 'C(children)' to treat children as categorical
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    return model


def main():
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)

    descriptive = summarize_children_effect(df)
    logit_model = fit_logit_model(df)

    # Extract effect of having children (relative to no children)
    params = logit_model.params.to_dict()
    conf_int = logit_model.conf_int().to_dict("index")
    pvalues = logit_model.pvalues.to_dict()

    # The coefficient name for children should be like 'C(children)[T.yes]'
    coef_key = None
    for key in params.keys():
        if key.startswith("C(children)"):
            coef_key = key
            break

    if coef_key is None:
        raise RuntimeError("Could not find children coefficient in model.")

    coef = float(params[coef_key])
    ci_low, ci_high = conf_int[coef_key]
    pval = float(pvalues[coef_key])

    # Collect everything into a JSON-friendly dict
    results = {
        "descriptive": descriptive,
        "logit_children_coef": {
            "coef": coef,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": pval,
        },
        "model_summary": logit_model.summary2().as_text(),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

