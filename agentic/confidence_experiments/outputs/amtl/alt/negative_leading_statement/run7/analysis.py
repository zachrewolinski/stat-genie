import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """Expand aggregated AMTL counts into per-tooth binary outcomes."""
    records = []
    for row in df.itertuples(index=False):
        sockets = int(row.sockets)
        num_amtl = int(row.num_amtl)
        # 1 for missing teeth (AMTL), 0 for present teeth
        outcomes = [1] * num_amtl + [0] * (sockets - num_amtl)
        for outcome in outcomes:
            records.append(
                {
                    "tooth_class": row.tooth_class,
                    "specimen": row.specimen,
                    "age": row.age,
                    "stdev_age": row.stdev_age,
                    "prob_male": row.prob_male,
                    "genus": row.genus,
                    "pop": row.pop,
                    "amtl_flag": outcome,
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Keep only rows with at least one observable socket
    df = df[df["sockets"] > 0].copy()

    # Expand to per-tooth data for a logistic regression on AMTL presence/absence
    tooth_df = expand_to_tooth_level(df)

    formula = (
        "amtl_flag ~ "
        "C(genus, Treatment(reference='Homo sapiens')) + "
        "age + prob_male + C(tooth_class)"
    )

    model = smf.logit(formula=formula, data=tooth_df)
    result = model.fit(disp=False)

    print("=== Genus coefficients (relative to Homo sapiens) ===")
    genus_params = result.params.filter(like='C(genus')  # type: ignore[arg-type]
    print(genus_params)

    print("\n=== Genus p-values (relative to Homo sapiens) ===")
    genus_pvals = result.pvalues.filter(like='C(genus')  # type: ignore[arg-type]
    print(genus_pvals)

    print("\n=== Genus odds ratios (relative to Homo sapiens) ===")
    print(genus_params.apply(lambda x: float(np.exp(x))))

    print("\n=== Model fit information ===")
    print(f"Log-likelihood: {result.llf:.3f}")
    print(f"Pseudo R-squared (McFadden): {result.prsquared:.3f}")


if __name__ == "__main__":
    main()
