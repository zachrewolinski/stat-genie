import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for having any extramarital affair in the past year
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status
    desc = {}
    for children_status in ["yes", "no"]:
        subset = df[df["children"] == children_status]
        desc[children_status] = {
            "n": int(len(subset)),
            "prop_had_affair": float(subset["had_affair"].mean()),
            "mean_affairs": float(subset["affairs"].mean()),
        }

    # Unadjusted logistic regression: had_affair ~ children
    unadj_model = smf.logit("had_affair ~ C(children)", data=df).fit(disp=False)
    unadj_params = unadj_model.params.to_dict()
    unadj_pvalues = unadj_model.pvalues.to_dict()

    # Adjusted logistic regression with key covariates
    adj_formula = (
        "had_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    adj_model = smf.logit(adj_formula, data=df).fit(disp=False)
    adj_params = adj_model.params.to_dict()
    adj_pvalues = adj_model.pvalues.to_dict()

    # Average predicted probability of an affair for each children status,
    # holding other covariates at their observed values.
    avg_pred = {}
    for children_status in ["yes", "no"]:
        df_copy = df.copy()
        df_copy["children"] = children_status
        avg_pred[children_status] = float(adj_model.predict(df_copy).mean())

    results = {
        "desc": desc,
        "unadjusted": {
            "params": unadj_params,
            "pvalues": unadj_pvalues,
        },
        "adjusted": {
            "params": adj_params,
            "pvalues": adj_pvalues,
            "avg_pred": avg_pred,
        },
    }

    # Save a machine-readable summary so we can base our written conclusion on it.
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

