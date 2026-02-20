import json

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Basic group-level summaries
    group_stats = (
        df.groupby("children")["affair_any"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(
            columns={
                "mean": "prop_any_affair",
                "count": "n",
                "sum": "n_any_affair",
            }
        )
    )

    # Unadjusted logistic regression: children only
    unadj_model = smf.logit("affair_any ~ C(children)", data=df).fit(disp=False)

    # Adjusted logistic regression with key covariates
    adj_formula = (
        "affair_any ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    adj_model = smf.logit(adj_formula, data=df).fit(disp=False)

    # Extract children effect from adjusted model
    children_term = "C(children)[T.yes]"
    coef = adj_model.params.get(children_term, float("nan"))
    se = adj_model.bse.get(children_term, float("nan"))
    pval = adj_model.pvalues.get(children_term, float("nan"))

    # Collect a compact summary that we can inspect from the CLI
    summary = {
        "group_stats": group_stats.to_dict(orient="records"),
        "unadjusted_children_coef": float(
            unadj_model.params.get("C(children)[T.yes]", float("nan"))
        ),
        "unadjusted_children_pvalue": float(
            unadj_model.pvalues.get("C(children)[T.yes]", float("nan"))
        ),
        "adjusted_children_coef": float(coef),
        "adjusted_children_se": float(se),
        "adjusted_children_pvalue": float(pval),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

