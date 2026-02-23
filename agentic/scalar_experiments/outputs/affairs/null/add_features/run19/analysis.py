import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Simple group summaries
    grp = df.groupby("children")
    summary = {
        "group_sizes": grp.size().to_dict(),
        "mean_affairs": grp["affairs"].mean().to_dict(),
        "median_affairs": grp["affairs"].median().to_dict(),
        "prop_any_affair": grp["any_affair"].mean().to_dict(),
    }

    # Logistic regression: probability of any affair
    logit_formula_simple = "any_affair ~ children_yes"
    logit_simple = smf.logit(logit_formula_simple, data=df).fit(disp=False)

    # Add a more fully adjusted model for robustness
    # Keep only columns that are clearly numeric and present in the CSV
    controls = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    available_controls = [c for c in controls if c in df.columns]
    if available_controls:
        formula_full = "any_affair ~ children_yes + " + " + ".join(available_controls)
        logit_full = smf.logit(formula_full, data=df).fit(disp=False)
    else:
        logit_full = None

    # Poisson regression on the affair count (including zeros)
    poisson_formula = "affairs ~ children_yes"
    poisson_model = smf.poisson(poisson_formula, data=df).fit(disp=False)

    if available_controls:
        poisson_full_formula = "affairs ~ children_yes + " + " + ".join(available_controls)
        poisson_full = smf.poisson(poisson_full_formula, data=df).fit(disp=False)
    else:
        poisson_full = None

    # Collect model statistics of interest
    def extract_effect(model, var_name: str):
        coef = model.params[var_name]
        se = model.bse[var_name]
        pval = model.pvalues[var_name]
        or_val = float(np.exp(coef))
        ci_low, ci_high = model.conf_int().loc[var_name].tolist()
        ci_low_or = float(np.exp(ci_low))
        ci_high_or = float(np.exp(ci_high))
        return {
            "coef": float(coef),
            "se": float(se),
            "p_value": float(pval),
            "odds_ratio_or_rr": or_val,
            "ci_95_low": ci_low_or,
            "ci_95_high": ci_high_or,
        }

    effects = {
        "logit_simple_children_yes": extract_effect(logit_simple, "children_yes"),
        "poisson_children_yes": extract_effect(poisson_model, "children_yes"),
    }
    if logit_full is not None:
        effects["logit_full_children_yes"] = extract_effect(logit_full, "children_yes")
    if poisson_full is not None:
        effects["poisson_full_children_yes"] = extract_effect(poisson_full, "children_yes")

    results = {
        "summary": summary,
        "effects": effects,
    }

    # Print results to stdout so they can be inspected from the CLI.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
