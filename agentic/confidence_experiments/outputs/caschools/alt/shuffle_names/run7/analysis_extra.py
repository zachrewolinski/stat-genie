from pathlib import Path

import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")
    df["str"] = df["english"] / df["students"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    vars_of_interest = ["str", "income", "school", "computer", "rownames", "grades", "testscr"]
    corr = df[vars_of_interest].corr()
    Path("corr_matrix.csv").write_text(corr.to_csv())
    print(corr)


if __name__ == "__main__":
    main()

