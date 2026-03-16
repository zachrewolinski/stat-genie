import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expand_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Expand count data (num_amtl / sockets) into per-socket binary outcomes."""
    records = []
    for _, row in df.iterrows():
        n_sockets = int(row["sockets"])
        n_missing = int(row["num_amtl"])
        # 1 = missing (AMTL), 0 = present
        outcomes = [1] * n_missing + [0] * (n_sockets - n_missing)
        for outcome in outcomes:
            records.append(
                {
                    "amtl": outcome,
                    "age": row["age"],
                    "prob_male": row["prob_male"],
                    "genus": row["genus"],
                    "tooth_class": row["tooth_class"],
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    print("Number of rows:", len(df))
    print("Columns:", list(df.columns))
    print("Genus levels:", df["genus"].unique())
    print("Tooth classes:", df["tooth_class"].unique())

    # Overall AMTL rate by genus
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    genus_summary = (
        df.groupby("genus")
        .agg(
            mean_rate=("amtl_rate", "mean"),
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .assign(overall_rate=lambda x: x["total_missing"] / x["total_sockets"])
    )
    print("\nAMTL summary by genus:")
    print(genus_summary)

    # Expand to long format: one row per tooth socket with binary AMTL outcome
    df_long = expand_to_long(df)
    print("\nLong-format data (per socket):")
    print(df_long.head())
    print("Long-format rows:", len(df_long))

    # Logistic regression on per-socket AMTL outcome
    # Homo sapiens is used as the reference genus.
    formula = (
        "amtl ~ C(genus, Treatment(reference='Homo sapiens')) + "
        "C(tooth_class) + age + prob_male"
    )

    logit_model = smf.logit(formula=formula, data=df_long).fit(disp=False)

    print("\nLogistic regression results (per socket):")
    print(logit_model.summary())

    # Extract coefficients for non-human genera vs Homo sapiens
    print("\nGenus coefficients (relative to Homo sapiens):")
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if term in logit_model.params:
            coef = logit_model.params[term]
            pval = logit_model.pvalues[term]
            odds_ratio = float(np.exp(coef))
            print(
                f"{genus}: coef = {coef:.4f}, odds ratio = {odds_ratio:.3f}, "
                f"p-value = {pval:.4g}"
            )
        else:
            print(f"{genus}: term not found in model.")


if __name__ == "__main__":
    main()
