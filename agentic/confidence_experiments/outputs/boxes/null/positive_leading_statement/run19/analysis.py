from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def fit_and_print(label: str, formula: str, data: pd.DataFrame) -> None:
    print(f"=== {label} ===")
    model = smf.logit(formula, data=data).fit(disp=False)
    print(model.summary())
    print(f"LLR p-value: {model.llr_pvalue:.4f}")
    print()


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Derived outcomes
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    social_df = df[df["social"] == 1].copy()

    # Reliance on social information
    fit_and_print("Social vs asocial ~ age", "social ~ age", df)
    fit_and_print("Social vs asocial ~ culture", "social ~ C(culture)", df)
    fit_and_print(
        "Social vs asocial ~ age + culture", "social ~ age + C(culture)", df
    )

    # Preference for majority cues among social choosers
    fit_and_print(
        "Majority vs minority ~ age", "majority_choice ~ age", social_df
    )
    fit_and_print(
        "Majority vs minority ~ culture", "majority_choice ~ C(culture)", social_df
    )
    fit_and_print(
        "Majority vs minority ~ age + culture",
        "majority_choice ~ age + C(culture)",
        social_df,
    )


if __name__ == "__main__":
    main()

