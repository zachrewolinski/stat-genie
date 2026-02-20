import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affairs
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic summaries by children status
    by_children_any = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "prop_any_affair", "sum": "n_with_affair", "count": "n_total"})
    )
    by_children_affairs = (
        df.groupby("children")["affairs"]
        .agg(["mean", "median"])
        .rename(columns={"mean": "mean_affairs", "median": "median_affairs"})
    )

    # Logistic regression for any affair, controlling for key covariates
    # Children coded as a factor, baseline = "no"
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + C(occupation) + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    coef = logit_model.params.get("C(children)[T.yes]", float("nan"))
    p_value = logit_model.pvalues.get("C(children)[T.yes]", float("nan"))

    # Collect key outputs to inspect from the CLI
    results = {
        "by_children_any": by_children_any.reset_index().to_dict(orient="records"),
        "by_children_affairs": by_children_affairs.reset_index().to_dict(orient="records"),
        "logit_children_coef": coef,
        "logit_children_p_value": p_value,
    }

    # Print JSON so the agent can read and reason about it
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

