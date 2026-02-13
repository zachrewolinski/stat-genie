import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_PATH = Path("caschools.csv")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # According to info.json descriptions:
    # - "english": total enrollment
    # - "students": number of teachers (FTE)
    # - "district": average reading score
    # - "expenditure": average math score
    df = df.copy()
    df["student_teacher_ratio"] = df["english"] / df["students"]
    df["avg_score"] = df[["district", "expenditure"]].mean(axis=1)
    return df


def simple_association(df: pd.DataFrame):
    """Univariate association between STR and average score."""
    mask = df["student_teacher_ratio"].replace([np.inf, -np.inf], np.nan).notna()
    sub = df.loc[mask, ["student_teacher_ratio", "avg_score"]].dropna()

    X = sm.add_constant(sub["student_teacher_ratio"])
    y = sub["avg_score"]
    model = sm.OLS(y, X).fit()

    corr = sub["student_teacher_ratio"].corr(sub["avg_score"])
    return {
        "n": int(sub.shape[0]),
        "corr": float(corr),
        "coef": float(model.params["student_teacher_ratio"]),
        "p_value": float(model.pvalues["student_teacher_ratio"]),
        "r_squared": float(model.rsquared),
        "summary": model.summary().as_text(),
    }


def adjusted_association(df: pd.DataFrame):
    """Association controlling for key covariates."""
    covariates = ["income", "school", "computer", "rownames"]
    cols = ["student_teacher_ratio", "avg_score"] + covariates

    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(sub[["student_teacher_ratio"] + covariates])
    y = sub["avg_score"]
    model = sm.OLS(y, X).fit()

    return {
        "n": int(sub.shape[0]),
        "coef": float(model.params["student_teacher_ratio"]),
        "p_value": float(model.pvalues["student_teacher_ratio"]),
        "r_squared": float(model.rsquared),
        "summary": model.summary().as_text(),
    }


def main():
    df = load_data()

    simple = simple_association(df)
    adjusted = adjusted_association(df)

    results = {
        "simple": {k: v for k, v in simple.items() if k != "summary"},
        "adjusted": {k: v for k, v in adjusted.items() if k != "summary"},
    }

    print(json.dumps(results, indent=2))

    # Also print brief directional interpretation for quick inspection.
    direction = "negative" if simple["coef"] < 0 else "positive"
    print(
        f"\nSimple association: coef={simple['coef']:.3f} "
        f"(p={simple['p_value']:.3g}, corr={simple['corr']:.3f}, "
        f"direction={direction})"
    )

    direction_adj = "negative" if adjusted["coef"] < 0 else "positive"
    print(
        f"Adjusted association: coef={adjusted['coef']:.3f} "
        f"(p={adjusted['p_value']:.3g}, direction={direction_adj})"
    )


if __name__ == "__main__":
    main()

