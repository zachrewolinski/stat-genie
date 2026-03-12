import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the AMTL dataset and construct analysis variables.

    Notes on column semantics (based on inspection of values and metadata):
    - `sockets` column: categorical, values {Anterior, Posterior, Premolar} → tooth class / location.
    - `tooth_class` column: categorical, values {Homo sapiens, Pan, Papio, Pongo} → actual genus.
    - `genus` column: integer 0–12 → count of missing teeth (AMTL) for that row.
    - `age` column: small integer 2–14 → number of observable sockets in that row.
    - `pop` column: continuous 8–71 → estimated age at death (years).
    - `stdev_age` column: values between 0 and 1 in steps like 0, 0.25, 0.5, 0.75, 1 → probability specimen is male.
    """
    df = pd.read_csv(csv_path)

    # Rename into semantically meaningful variables for analysis
    df = df.copy()
    df["tooth_loc"] = df["sockets"]
    df["genus_cat"] = df["tooth_class"]
    df["num_missing"] = df["genus"]
    df["num_sockets"] = df["age"]
    df["age_years"] = df["pop"]
    df["prob_male_num"] = df["stdev_age"]

    # Basic cleaning: keep rows with valid counts
    df = df[df["num_sockets"] > 0].copy()
    # Ensure integer counts for binomial model
    df["num_missing"] = df["num_missing"].round().astype(int)
    df["num_sockets"] = df["num_sockets"].round().astype(int)

    # Clip any pathological rows where missing would exceed sockets (very unlikely)
    df["num_missing"] = df[["num_missing", "num_sockets"]].min(axis=1)

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus_cat"] == "Homo sapiens").astype(int)

    return df


def expand_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """Expand grouped counts into tooth-level binary observations for logistic regression."""
    records = []
    for _, row in df.iterrows():
        n_sockets = int(row["num_sockets"])
        n_missing = int(row["num_missing"])
        if n_sockets <= 0:
            continue

        n_missing = max(0, min(n_missing, n_sockets))
        # Create one record per tooth socket: 1 = missing (AMTL), 0 = present
        outcomes = np.array([1] * n_missing + [0] * (n_sockets - n_missing), dtype=int)

        for y in outcomes:
            records.append(
                {
                    "missing": y,
                    "is_human": row["is_human"],
                    "age_years": row["age_years"],
                    "prob_male_num": row["prob_male_num"],
                    "tooth_loc": row["tooth_loc"],
                }
            )

    return pd.DataFrame.from_records(records)


def fit_logistic_model(tooth_df: pd.DataFrame):
    """Fit a binomial logistic regression for AMTL at the tooth level."""
    formula = "missing ~ is_human + age_years + prob_male_num + C(tooth_loc)"
    model = smf.glm(formula=formula, data=tooth_df, family=sm.families.Binomial())
    result = model.fit()
    return result


def summarize_group_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive AMTL frequencies by genus."""
    df = df.copy()
    df["amtl_prop"] = df["num_missing"] / df["num_sockets"].replace(0, np.nan)
    return (
        df.groupby("genus_cat")
        .agg(
            mean_amtl_prop=("amtl_prop", "mean"),
            sd_amtl_prop=("amtl_prop", "std"),
            n_rows=("amtl_prop", "size"),
        )
        .reset_index()
    )


def main():
    df = load_and_prepare_data("amtl.csv")

    # Sanity checks on mapping of counts
    invalid = (df["num_missing"] > df["num_sockets"]).sum()
    print(f"Rows with num_missing > num_sockets: {invalid}")

    group_means = summarize_group_means(df)
    print("\nAMTL proportion by genus (using grouped data):")
    print(group_means.to_string(index=False))

    tooth_df = expand_tooth_level(df)
    print(f"\nExpanded tooth-level rows: {len(tooth_df)}")

    result = fit_logistic_model(tooth_df)
    print("\nLogistic regression summary (missing ~ is_human + age_years + prob_male_num + tooth_loc):")
    print(result.summary())

    # Extract key effect: human vs non-human
    coef_human = result.params.get("is_human", np.nan)
    pvalue_human = result.pvalues.get("is_human", np.nan)

    print("\nKey effect for research question:")
    print(f"is_human coefficient (log-odds): {coef_human:.4f}")
    print(f"is_human p-value: {pvalue_human:.4g}")


if __name__ == "__main__":
    main()

