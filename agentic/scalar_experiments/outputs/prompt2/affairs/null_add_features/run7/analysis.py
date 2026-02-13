import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning / type checks
    if "children" not in df.columns or "affairs" not in df.columns:
        raise ValueError("Expected 'children' and 'affairs' columns to be present.")

    # Create a binary indicator for having at least one affair in the last year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Drop rows with missing values in the key variables, if any
    df_model = df.dropna(subset=["affair_any", "children"])

    # Encode children as an indicator: 1 = has children, 0 = no children
    df_model["children_yes"] = (df_model["children"].astype(str).str.lower() == "yes").astype(int)

    # Descriptive statistics: proportion with any affair by children status
    group_stats = (
        df_model.groupby("children")["affair_any"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_affair_any", "count": "n"})
    )

    # Logistic regression of any affair on children, adjusting for a few plausible confounders if present
    covariates = []
    for col in ["age", "yearsmarried", "religiousness", "rating"]:
        if col in df_model.columns:
            covariates.append(col)

    formula = "affair_any ~ children_yes"
    if covariates:
        formula += " + " + " + ".join(covariates)

    logit_model = smf.logit(formula=formula, data=df_model).fit(disp=False)
    params = logit_model.params
    conf_int = logit_model.conf_int()

    # Extract effect of children
    effect = params["children_yes"]
    ci_low, ci_high = conf_int.loc["children_yes"]

    # Store results to a JSON file for inspection if needed
    results = {
        "group_stats": group_stats.reset_index().to_dict(orient="records"),
        "logit_coef_children": float(effect),
        "logit_ci_children": [float(ci_low), float(ci_high)],
        "logit_pvalue_children": float(logit_model.pvalues["children_yes"]),
        "model_n": int(logit_model.nobs),
        "model_formula": formula,
    }

    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

