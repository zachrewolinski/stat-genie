import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning: ensure expected columns exist
    required_cols = {"affairs", "children"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    # Descriptive statistics by children status (affair count)
    desc = (
        df.groupby("children")["affairs"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )

    # Binary indicator for any extramarital affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    prop_any = (
        df.groupby("children")["any_affair"]
        .mean()
        .reset_index()
        .rename(columns={"any_affair": "prop_any_affair"})
    )

    # Linear regression of affair frequency on children + controls
    # affairs is treated as numeric frequency; controls mirror Fair's classic spec
    formula = "affairs ~ C(children) + age + yearsmarried + religiousness + education + C(gender) + rating"
    model = smf.ols(formula=formula, data=df).fit()

    children_coef = model.params.get("C(children)[T.yes]", float("nan"))
    children_pvalue = model.pvalues.get("C(children)[T.yes]", float("nan"))

    # Logistic regression for probability of any affair
    logit_formula = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + C(gender) + rating"
    logit_model = smf.logit(formula=logit_formula, data=df).fit(disp=False)
    logit_children_coef = logit_model.params.get("C(children)[T.yes]", float("nan"))
    logit_children_pvalue = logit_model.pvalues.get("C(children)[T.yes]", float("nan"))

    results = {
        "group_stats": desc.to_dict(orient="records"),
        "prop_any_affair_by_children": prop_any.to_dict(orient="records"),
        "children_coef": children_coef,
        "children_pvalue": children_pvalue,
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "logit_children_coef": logit_children_coef,
        "logit_children_pvalue": logit_children_pvalue,
    }

    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
