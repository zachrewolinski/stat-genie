import pandas as pd
import statsmodels.api as sm
import numpy as np
from pathlib import Path


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # According to info.json, "religiousness" actually encodes
    # "Are there children in the marriage?" with yes/no values.
    # The "age" column encodes frequency of extramarital intercourse.
    if "religiousness" not in df.columns or "age" not in df.columns:
        raise SystemExit("Expected columns 'religiousness' and 'age' not found.")

    # Binary indicator for having children
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Binary outcome: any extramarital intercourse in past year
    df["any_affair"] = (df["age"] > 0).astype(int)

    # Drop rows with missing encodings, if any
    df = df.dropna(subset=["has_children", "any_affair"])

    # Descriptive comparison of proportions
    prop_table = df.groupby("has_children")["any_affair"].mean()

    # Logistic regression of any_affair on has_children
    X = sm.add_constant(df["has_children"])
    y = df["any_affair"]
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    coef_children = result.params["has_children"]
    p_value_children = result.pvalues["has_children"]
    odds_ratio = float(np.exp(coef_children))

    # Prepare a human-readable summary to guide the final conclusion
    summary_path = Path("analysis_summary.txt")
    with summary_path.open("w") as f:
        f.write("Proportion with any affair by children status (has_children=0/1):\n")
        f.write(prop_table.to_string())
        f.write("\n\nLogit(any_affair ~ has_children):\n")
        f.write(f"coef_children = {coef_children:.4f}\n")
        f.write(f"odds_ratio = {odds_ratio:.4f}\n")
        f.write(f"p_value_children = {p_value_children:.4g}\n")


if __name__ == "__main__":
    main()
