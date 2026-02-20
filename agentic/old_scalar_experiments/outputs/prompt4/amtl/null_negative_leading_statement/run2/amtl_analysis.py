import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Proportion of antemortem tooth loss per row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    print("=== Basic AMTL summary by genus ===")
    genus_summary = (
        df.groupby("genus")["prop_amtl"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print(genus_summary.to_string(index=False))
    print()

    # Expand to per-tooth data so we can fit a logistic regression on individual sockets.
    records = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        num_missing = int(row["num_amtl"])
        num_present = sockets - num_missing

        base = {
            "age": row["age"],
            "prob_male": row["prob_male"],
            "tooth_class": row["tooth_class"],
            "is_human": row["is_human"],
            "genus": row["genus"],
            "specimen": row["specimen"],
            "pop": row["pop"],
        }

        if num_missing > 0:
            missing_rec = base.copy()
            missing_rec["missing"] = 1
            records.extend([missing_rec] * num_missing)

        if num_present > 0:
            present_rec = base.copy()
            present_rec["missing"] = 0
            records.extend([present_rec] * num_present)

    tooth_df = pd.DataFrame.from_records(records)
    print("Per-tooth dataframe rows:", len(tooth_df))
    print()

    # Logistic regression: probability a given socket shows AMTL.
    # Control for age, estimated sex (prob_male), and tooth class.
    print("=== Logistic regression: missing ~ is_human + age + prob_male + tooth_class ===")
    logit_model = smf.logit(
        "missing ~ is_human + age + prob_male + C(tooth_class)", data=tooth_df
    ).fit(disp=False)

    print(logit_model.summary())
    print()

    coef = logit_model.params["is_human"]
    or_human = float(np.exp(coef))
    pvalue = float(logit_model.pvalues["is_human"])

    print(f"is_human logit coef: {coef:.3f}")
    print(f"is_human odds ratio: {or_human:.3f}")
    print(f"is_human p-value: {pvalue:.4g}")
    print()

    # Also compare adjusted predicted probabilities for a representative individual.
    median_age = float(df["age"].median())
    mean_prob_male = float(df["prob_male"].mean())
    common_tooth_class = df["tooth_class"].mode().iat[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [median_age, median_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [common_tooth_class, common_tooth_class],
        }
    )

    pred_probs = logit_model.predict(pred_df)
    print("Representative predictions at median age and mean prob_male,",
          f"tooth_class={common_tooth_class}:")
    print(f"  Non-human primates (is_human=0): {pred_probs.iloc[0]:.3f}")
    print(f"  Humans          (is_human=1): {pred_probs.iloc[1]:.3f}")


if __name__ == "__main__":
    main()

