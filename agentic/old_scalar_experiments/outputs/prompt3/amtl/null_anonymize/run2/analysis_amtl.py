import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic cleaning and validity checks for counts
    df = df.dropna(
        subset=[
            "feature1",
            "feature3",
            "feature4",
            "feature5",
            "feature7",
            "feature8",
        ]
    )

    df = df[df["feature4"] > 0]  # must have observable sockets
    df = df[df["feature3"] >= 0]
    df = df[df["feature3"] <= df["feature4"]]

    # Cast count variables to integers
    df["feature3"] = df["feature3"].astype(int)
    df["feature4"] = df["feature4"].astype(int)

    return df


def make_tooth_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        missing = int(row["feature3"])
        sockets = int(row["feature4"])
        present = sockets - missing

        base = {
            "genus": row["feature8"],
            "tooth_class": row["feature1"],
            "age": float(row["feature5"]),
            "sex_est": float(row["feature7"]),
        }

        for _ in range(missing):
            rows.append({**base, "amtl": 1})
        for _ in range(present):
            rows.append({**base, "amtl": 0})

    long_df = pd.DataFrame(rows)
    long_df["genus"] = long_df["genus"].astype("category")
    long_df["tooth_class"] = long_df["tooth_class"].astype("category")
    return long_df


def fit_logistic_model(long_df: pd.DataFrame):
    formula = "amtl ~ C(genus) + age + sex_est + C(tooth_class)"
    model = smf.logit(formula=formula, data=long_df)
    result = model.fit(disp=False)
    return result


def main() -> None:
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)
    long_df = make_tooth_level_dataframe(df)

    # Fit logistic regression model
    result = fit_logistic_model(long_df)

    # Print key summaries for interactive inspection
    print(result.summary())

    genus_rates = long_df.groupby("genus")["amtl"].mean().sort_values(ascending=False)
    print("\nMean AMTL probability by genus:")
    print(genus_rates)


if __name__ == "__main__":
    main()

