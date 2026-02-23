import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata (not strictly needed for calculations, but kept for transparency)
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    data_path = base_dir / "caschools.csv"
    df = pd.read_csv(data_path)

    # Map shuffled column names to their semantic meanings based on info.json descriptions.
    # Semantics:
    # - Total enrollment (students): column "english"
    # - Number of teachers: column "students"
    # - Reading score: column "district"
    # - Math score: column "expenditure"
    # - Percent CalWorks: column "school"
    # - Percent reduced-price lunch: column "computer"
    # - Percent English learners: column "rownames"
    # - Expenditure per student: column "grades"
    enroll = df["english"].astype(float)
    teachers = df["students"].astype(float)

    # Avoid division by zero
    valid = teachers > 0
    df = df.loc[valid].copy()
    enroll = enroll[valid]
    teachers = teachers[valid]

    df["stratio"] = enroll / teachers

    # Academic performance: average of reading and math scores
    df["read_score"] = df["district"].astype(float)
    df["math_score"] = df["expenditure"].astype(float)
    df["testscr"] = (df["read_score"] + df["math_score"]) / 2.0

    # Key controls commonly used with this dataset
    df["income_k"] = df["income"].astype(float)
    df["calworks_pct"] = df["school"].astype(float)
    df["lunch_pct"] = df["computer"].astype(float)
    df["ell_pct"] = df["rownames"].astype(float)
    df["expn_stu"] = df["grades"].astype(float)

    # Simple correlation between STR and test scores
    corr, corr_p = pearsonr(df["stratio"], df["testscr"])

    # Bivariate regression: testscr ~ stratio
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit()
    b1 = model1.params["stratio"]
    p1 = model1.pvalues["stratio"]

    # Multivariate regression with standard controls
    X2 = df[
        ["stratio", "income_k", "calworks_pct", "lunch_pct", "ell_pct", "expn_stu"]
    ]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(df["testscr"], X2).fit()
    b2 = model2.params["stratio"]
    p2 = model2.pvalues["stratio"]

    # Summarize results in a compact JSON-like structure for manual inspection
    summary = {
        "research_question": research_question,
        "n_obs": int(df.shape[0]),
        "stratio": {
            "mean": float(df["stratio"].mean()),
            "std": float(df["stratio"].std()),
            "min": float(df["stratio"].min()),
            "max": float(df["stratio"].max()),
        },
        "testscr": {
            "mean": float(df["testscr"].mean()),
            "std": float(df["testscr"].std()),
            "min": float(df["testscr"].min()),
            "max": float(df["testscr"].max()),
        },
        "correlation": {
            "pearson_r": float(corr),
            "p_value": float(corr_p),
        },
        "ols_bivariate": {
            "coef_stratio": float(b1),
            "p_value_stratio": float(p1),
            "r_squared": float(model1.rsquared),
        },
        "ols_multivariate": {
            "coef_stratio": float(b2),
            "p_value_stratio": float(p2),
            "r_squared": float(model2.rsquared),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

