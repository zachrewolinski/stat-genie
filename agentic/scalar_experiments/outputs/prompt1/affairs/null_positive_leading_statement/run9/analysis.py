import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_csv(data_path)

    # Binary indicator: any extramarital intercourse in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group-level summaries by presence of children
    group = (
        df.groupby("children")
        .agg(
            n=("has_affair", "size"),
            n_affair=("has_affair", "sum"),
            prop_affair=("has_affair", "mean"),
            mean_affairs=("affairs", "mean"),
        )
        .reset_index()
    )

    print("Group summaries by children (yes/no):")
    print(group.to_string(index=False))
    print()

    # Logistic regression of any affair on children and key covariates
    df["gender"] = df["gender"].astype("category")
    df["children"] = df["children"].astype("category")

    formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )

    model = smf.logit(formula, data=df).fit(disp=False)

    print(model.summary())
    print()

    params = model.params
    conf_int = model.conf_int()
    pvalues = model.pvalues

    # Effect of having children (yes vs no)
    child_term = "C(children)[T.yes]"
    if child_term in params.index:
        log_odds = params[child_term]
        ci_low, ci_high = conf_int.loc[child_term]
        p_value = pvalues[child_term]

        odds_ratio = float(np.exp(log_odds))
        or_ci_low = float(np.exp(ci_low))
        or_ci_high = float(np.exp(ci_high))

        print("Effect of children from logistic regression (has_affair outcome):")
        print(f"  Log-odds coefficient (children = yes vs no): {log_odds:.4f}")
        print(
            f"  Odds ratio: {odds_ratio:.3f} "
            f"(95% CI: {or_ci_low:.3f}, {or_ci_high:.3f}), p-value = {p_value:.4g}"
        )
    else:
        print("Children term not found in model parameters.")

    # Save key statistics for downstream explanation if desired
    summary = {
        "group_summary": group.to_dict(orient="list"),
        "logistic_children_coef": float(params.get(child_term, np.nan)),
        "logistic_children_pvalue": float(pvalues.get(child_term, np.nan)),
        "logistic_children_or": float(np.exp(params[child_term]))
        if child_term in params
        else np.nan,
        "logistic_children_or_ci": [
            float(np.exp(conf_int.loc[child_term, 0]))
            if child_term in conf_int.index
            else np.nan,
            float(np.exp(conf_int.loc[child_term, 1]))
            if child_term in conf_int.index
            else np.nan,
        ],
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

