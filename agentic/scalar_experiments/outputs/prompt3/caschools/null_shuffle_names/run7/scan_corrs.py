from pathlib import Path

import pandas as pd


def main():
    base_dir = Path(__file__).parent
    df = pd.read_csv(base_dir / "caschools.csv")

    # Construct overall test score as in analysis.py
    read_score = df["district"]
    math_score = df["expenditure"]
    testscr = (read_score + math_score) / 2.0
    df["testscr"] = testscr

    numeric_cols = df.select_dtypes(include="number").columns

    corrs = {}
    for col in numeric_cols:
        if col == "testscr":
            continue
        corrs[col] = df[col].corr(df["testscr"])

    # Print correlations sorted by absolute magnitude
    for col, r in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{col}: {r:.4f}")


if __name__ == "__main__":
    main()

