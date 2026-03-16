import pandas as pd
import statsmodels.formula.api as smf


def load_data() -> pd.DataFrame:
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    df["prop_missing"] = df["missing"] / df["sockets"]
    return df


def explore(df: pd.DataFrame) -> None:
    print("Head:")
    print(df.head())
    print("\nGenus counts:")
    print(df["genus"].value_counts())
    print("\nMean proportion missing by genus:")
    print(df.groupby("genus")["prop_missing"].mean())
    print("\nMean proportion missing by genus and tooth class:")
    print(df.groupby(["genus", "tooth_class"])["prop_missing"].mean())


def fit_logistic(df: pd.DataFrame):
    # Expand counts into per-socket Bernoulli outcomes for a true binomial model
    records = []
    for _, row in df.iterrows():
        missing = int(row["missing"])
        sockets = int(row["sockets"])
        present = sockets - missing
        # 1 = missing tooth (AMTL), 0 = present
        for outcome in ([1] * missing + [0] * present):
            records.append(
                {
                    "amtl": outcome,
                    "genus": row["genus"],
                    "tooth_class": row["tooth_class"],
                    "age": row["age"],
                    "sex_estimate": row["sex_estimate"],
                }
            )

    long_df = pd.DataFrame.from_records(records)
    print(f"\nExpanded dataset has {len(long_df)} socket-level rows.")

    # Model with full genus factor
    model_genus = smf.logit(
        "amtl ~ C(genus) + C(tooth_class) + age + sex_estimate", data=long_df
    ).fit(disp=False)

    print("\nLogistic regression with genus factor (AMTL per socket):")
    print(model_genus.summary())

    # Extract genus coefficients to see how each non-reference genus compares
    genus_params = {
        name: (coef, model_genus.bse[name], model_genus.pvalues[name])
        for name, coef in model_genus.params.items()
        if name.startswith("C(genus)[T.")
    }
    print("\nGenus effects relative to reference level (Homo sapiens):")
    for name, (coef, se, pval) in genus_params.items():
        print(f"{name}: coef={coef:.3f}, se={se:.3f}, p-value={pval:.4g}")

    # Direct human vs non-human contrast
    long_df["is_human"] = (long_df["genus"] == "Homo sapiens").astype(int)
    model_human = smf.logit(
        "amtl ~ is_human + C(tooth_class) + age + sex_estimate", data=long_df
    ).fit(disp=False)

    print("\nLogistic regression: humans vs non-humans:")
    print(model_human.summary())


def main() -> None:
    df = load_data()
    explore(df)
    fit_logistic(df)


if __name__ == "__main__":
    main()
