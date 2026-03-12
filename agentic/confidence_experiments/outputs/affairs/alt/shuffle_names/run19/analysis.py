import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent

    info_path = base_path / "info.json"
    data_path = base_path / "affairs.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Based on info.json descriptions:
    # - Column "age" is actually the affair frequency scale.
    # - Column "religiousness" is a yes/no indicator for children in the marriage.
    df = df.copy()
    df["any_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop rows with missing mappings, if any.
    df = df.dropna(subset=["any_affair", "has_children"])

    # Descriptive statistics: affair rate by children status.
    group_stats = (
        df.groupby("has_children")["any_affair"]
        .agg(["mean", "count"])
        .rename(index={0: "no_children", 1: "has_children"})
    )
    print("Affair prevalence by children status:")
    print(group_stats)
    print()

    # Logistic regression: any_affair ~ has_children
    model = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
    print(model.summary())

    # Extract key quantities for reasoning.
    params = model.params
    conf_int = model.conf_int()
    pvalues = model.pvalues

    effect = params["has_children"]
    ci_low, ci_high = conf_int.loc["has_children"]
    pval = pvalues["has_children"]

    print()
    print("Effect of having children on log-odds of any affair:")
    print(f"  coef = {effect:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}], p = {pval:.4g}")


if __name__ == "__main__":
    main()

