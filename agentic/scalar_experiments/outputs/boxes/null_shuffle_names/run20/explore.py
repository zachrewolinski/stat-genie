import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Social reliance: choosing any demonstrated option (majority or minority)
    df["social"] = (df["majority_first"] != 1).astype(int)

    # Majority preference among social choices
    df["majority"] = np.where(
        df["majority_first"] == 2,
        1,
        np.where(df["majority_first"] == 3, 0, np.nan),
    )

    print(f"N = {len(df)}")
    print(f"Overall social reliance (P(social)): {df['social'].mean():.3f}")
    print(
        "Overall majority preference among social choices "
        f"(P(majority | social)): {df['majority'].mean():.3f}"
    )

    by_age_social = df.groupby("age")["social"].mean()
    by_site_social = df.groupby("y")["social"].mean()
    by_age_majority = df.groupby("age")["majority"].mean()
    by_site_majority = df.groupby("y")["majority"].mean()

    print("\nSocial reliance by age:")
    print(by_age_social.to_string())

    print("\nSocial reliance by site (y):")
    print(by_site_social.to_string())

    print("\nMajority preference by age:")
    print(by_age_majority.to_string())

    print("\nMajority preference by site (y):")
    print(by_site_majority.to_string())

    print("\nRanges (max - min):")
    print(f"  Social by age: {by_age_social.max() - by_age_social.min():.3f}")
    print(f"  Social by site: {by_site_social.max() - by_site_social.min():.3f}")
    print(f"  Majority by age: {by_age_majority.max() - by_age_majority.min():.3f}")
    print(f"  Majority by site: {by_site_majority.max() - by_site_majority.min():.3f}")


if __name__ == "__main__":
    main()

