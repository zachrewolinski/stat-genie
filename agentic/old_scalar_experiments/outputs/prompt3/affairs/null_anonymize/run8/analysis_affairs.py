import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Create key variables
    df["has_children"] = (df["feature6"].str.lower() == "yes").astype(int)
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Group-level summaries
    group_affair_rate = df.groupby("has_children")["has_affair"].mean()
    group_affair_freq = df.groupby("has_children")["feature2"].mean()

    # Logistic regression: affair (any vs none) on children indicator
    X = sm.add_constant(df[["has_children"]])
    y = df["has_affair"]
    model = sm.Logit(y, X).fit(disp=False)

    coef = float(model.params["has_children"])
    pvalue = float(model.pvalues["has_children"])

    # Prepare a small analysis summary to inspect from the CLI
    summary = {
        "n": int(len(df)),
        "affair_rate_children": float(group_affair_rate.get(1, float("nan"))),
        "affair_rate_no_children": float(group_affair_rate.get(0, float("nan"))),
        "affair_freq_children": float(group_affair_freq.get(1, float("nan"))),
        "affair_freq_no_children": float(group_affair_freq.get(0, float("nan"))),
        "logit_coef_children": coef,
        "logit_pvalue_children": pvalue,
    }

    # Write analysis summary to a JSON file for inspection
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

