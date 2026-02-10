import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    base = Path(__file__).parent

    info = load_metadata(base / "info.json")
    print("Research question:")
    for q in info.get("research_questions", []):
        print(f" - {q}")

    df = pd.read_csv(base / "caschools.csv")
    print(f"\nLoaded data with shape: {df.shape}")
    print("Columns:", list(df.columns))

    # Based on the metadata in info.json, the true semantic variables are:
    # - Total enrollment: column 'english'
    # - Number of teachers: column 'students'
    # - Reading score: column 'district'
    # - Math score: column 'expenditure'
    enroll = df["english"].astype(float)
    teachers = df["students"].astype(float)
    read_score = df["district"].astype(float)
    math_score = df["expenditure"].astype(float)

    # Student–teacher ratio: students per teacher
    ratio = enroll / teachers

    # Overall academic performance: average of reading and math scores
    avg_score = (read_score + math_score) / 2.0

    df_analysis = pd.DataFrame(
        {
            "ratio": ratio,
            "read": read_score,
            "math": math_score,
            "avg_score": avg_score,
        }
    ).dropna()

    print("\nSummary statistics:")
    print(df_analysis.describe())

    corr = df_analysis["ratio"].corr(df_analysis["avg_score"])
    print(f"\nCorrelation between student-teacher ratio and avg score: {corr:.4f}")

    # Simple linear regression: avg_score ~ ratio
    X = sm.add_constant(df_analysis["ratio"])
    y = df_analysis["avg_score"]
    model = sm.OLS(y, X).fit()
    print("\nLinear regression: avg_score ~ ratio")
    print(model.summary())

    slope = model.params["ratio"]
    p_value = model.pvalues["ratio"]
    r_squared = model.rsquared

    print("\nKey effects:")
    print(f"  Slope (effect of 1-student increase in ratio): {slope:.4f}")
    print(f"  p-value for slope: {p_value:.4g}")
    print(f"  R-squared: {r_squared:.4f}")

    # Save a small JSON with key metrics to help choose the Likert scalar.
    metrics = {
        "corr_ratio_avg": float(corr),
        "slope_ratio_avg": float(slope),
        "p_value_slope": float(p_value),
        "r_squared": float(r_squared),
    }
    (base / "analysis_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved metrics to analysis_metrics.json")


if __name__ == "__main__":
    main()

