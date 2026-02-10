import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).parent

def load_metadata():
    info_path = BASE_DIR / "info.json"
    with info_path.open("r") as f:
        return json.load(f)


def load_data():
    csv_path = BASE_DIR / "caschools.csv"
    return pd.read_csv(csv_path)


def run_analysis():
    info = load_metadata()
    df = load_data()

    # Construct student-teacher ratio
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]

    # Outcome: average of reading and math scores
    df["avgscore"] = df[["read", "math"]].mean(axis=1)

    # Simple bivariate regression: avgscore on stratio
    X = sm.add_constant(df["stratio"])
    y = df["avgscore"]
    model = sm.OLS(y, X).fit()

    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]
    r2 = model.rsquared

    result = {
        "coef_stratio": float(coef),
        "pvalue_stratio": float(pval),
        "r_squared": float(r2),
        "n_obs": int(model.nobs),
        "question": info["research_questions"][0],
    }

    return result


def map_to_scalar(coef: float, pval: float, r2: float) -> int:
    """Map evidence about association to Likert scalar.

    Negative coef -> lower ratio associated with higher performance.
    We interpret strength using effect size, significance, and R^2.
    """

    # Direction: negative supports "Yes" (association as posed), positive supports "No".
    if pval > 0.1 or abs(coef) < 0.1:
        base = 0
    else:
        # Scale magnitude of coefficient (per one student per teacher) into [-60, 60]
        mag = min(abs(coef) / 1.0, 1.0)  # cap
        strength = int(round(mag * 60))
        if coef < 0:
            base = strength  # evidence for "Yes"
        else:
            base = -strength  # evidence for "No"

    # Adjust based on model fit (R^2 up to ~0.5 giving +/-20)
    r2_adj = int(round(min(max(r2, 0.0), 0.5) / 0.5 * 20))
    if base > 0:
        base += r2_adj
    elif base < 0:
        base -= r2_adj

    # Ensure within bounds
    if base > 100:
        base = 100
    if base < -100:
        base = -100

    return int(base)


def main():
    result = run_analysis()
    scalar = map_to_scalar(
        coef=result["coef_stratio"],
        pval=result["pvalue_stratio"],
        r2=result["r_squared"],
    )

    # Write scalar only to conclusion.txt
    out_path = BASE_DIR / "conclusion.txt"
    with out_path.open("w") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()
