from pathlib import Path

import pandas as pd


def main() -> None:
    cwd = Path(__file__).resolve().parent
    df = pd.read_csv(cwd / "boxes.csv")
    df["social_choice"] = (df["y"] != 1).astype(int)
    social_df = df[df["social_choice"] == 1].copy()
    social_df = social_df[social_df["y"].isin([2, 3])]
    social_df["majority_choice"] = (social_df["y"] == 2).astype(int)

    print("Social choice rate by culture:")
    print(df.groupby("culture")["social_choice"].mean())
    print("\nMajority-following among social choices by culture:")
    print(social_df.groupby("culture")["majority_choice"].mean())
    print("\nSocial choice rate by age:")
    print(df.groupby("age")["social_choice"].mean())
    print("\nMajority-following among social choices by age:")
    print(social_df.groupby("age")["majority_choice"].mean())


if __name__ == "__main__":
    main()

