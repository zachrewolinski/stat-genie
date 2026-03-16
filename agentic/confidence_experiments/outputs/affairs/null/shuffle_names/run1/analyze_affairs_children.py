import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (not strictly needed for computations here, but keeps mapping transparent)
    info_path = base_path / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    data_path = base_path / "affairs.csv"
    df = pd.read_csv(data_path)

    # According to info.json descriptions, the true semantics are:
    # - "age" column: frequency of extramarital sexual intercourse during past year
    # - "religiousness" column: factor, "Are there children in the marriage?" (yes/no)
    #
    # Construct variables of interest:
    #   affair_any: 1 if any extramarital intercourse (age > 0), else 0
    #   children: 1 if "religiousness" == "yes", else 0
    df = df.copy()
    df["affair_any"] = (df["age"] > 0).astype(int)
    df["children_bin"] = (df["religiousness"].astype(str).str.lower() == "yes").astype(int)

    # Drop rows with missing data in key variables, if any
    df = df.dropna(subset=["affair_any", "children_bin"])

    # Basic group summaries
    grouped = df.groupby("children_bin")["affair_any"]
    prop_with_affair = grouped.mean()
    n_per_group = grouped.size()

    # Logistic regression: affair_any ~ children_bin
    logit_model = smf.logit("affair_any ~ children_bin", data=df).fit(disp=False)
    params = logit_model.params
    conf_int = logit_model.conf_int()
    pvalues = logit_model.pvalues

    children_coef = params["children_bin"]
    children_p = pvalues["children_bin"]
    children_ci_low, children_ci_high = conf_int.loc["children_bin"]
    odds_ratio = float(np.exp(children_coef))
    or_ci_low = float(np.exp(children_ci_low))
    or_ci_high = float(np.exp(children_ci_high))

    # Also compute mean affair frequency (raw "age" score) by children status
    freq_grouped = df.groupby("children_bin")["age"]
    mean_freq = freq_grouped.mean()
    std_freq = freq_grouped.std()

    # Print a concise summary for manual inspection
    print("=== Descriptive stats ===")
    for children_val in [0, 1]:
        label = "no_children" if children_val == 0 else "children"
        print(
            f"{label}: n={int(n_per_group[children_val])}, "
            f"prop_any_affair={prop_with_affair[children_val]:.3f}, "
            f"mean_freq_score={mean_freq[children_val]:.3f}, "
            f"sd_freq_score={std_freq[children_val]:.3f}"
        )

    print("\n=== Logistic regression (affair_any ~ children_bin) ===")
    print(f"coef_children={children_coef:.3f}")
    print(f"p_children={children_p:.4g}")
    print(
        f"OR_children={odds_ratio:.3f} "
        f"(95% CI: {or_ci_low:.3f}, {or_ci_high:.3f})"
    )


if __name__ == "__main__":
    main()

