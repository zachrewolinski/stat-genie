import pandas as pd


def main() -> None:
    df = pd.read_csv("boxes.csv")

    print("Basic describe:")
    print(df.describe())

    print("\nValue counts for feature1 (outcome):")
    print(df["feature1"].value_counts().sort_index())

    print("\nValue counts for feature5 (site):")
    print(df["feature5"].value_counts().sort_index())

    print("\nAge summary by outcome (feature1):")
    print(df.groupby("feature1")["feature3"].describe())

    print("\nOutcome proportions by site (rows=site, cols=outcome):")
    site_props = (
        df.pivot_table(
            index="feature5",
            columns="feature1",
            values="feature3",
            aggfunc="count",
        )
        .fillna(0)
    )
    site_props = site_props.div(df.groupby("feature5")["feature1"].count(), axis=0)
    print(site_props)

    print("\nOutcome proportions by age bin (rows=age bin, cols=outcome):")
    age_bins = pd.cut(df["feature3"], bins=[4, 6, 8, 10, 12, 14], right=True, include_lowest=True)
    age_props = (
        df.pivot_table(
            index=age_bins,
            columns="feature1",
            values="feature3",
            aggfunc="count",
        )
        .fillna(0)
    )
    age_props = age_props.div(df.groupby(age_bins)["feature1"].count(), axis=0)
    print(age_props)


if __name__ == "__main__":
    main()

