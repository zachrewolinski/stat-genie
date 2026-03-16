import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expand_to_teeth(df: pd.DataFrame) -> pd.DataFrame:
    """Expand count data to one row per tooth with 0/1 AMTL outcome."""
    rows = []
    for _, row in df.iterrows():
        n_amtl = int(row["num_amtl"])
        n_sockets = int(row["sockets"])
        n_present = n_sockets - n_amtl

        base = row.drop(labels=["num_amtl", "sockets"]).to_dict()

        for _ in range(n_amtl):
            rec = base.copy()
            rec["is_amtl"] = 1
            rows.append(rec)

        for _ in range(n_present):
            rec = base.copy()
            rec["is_amtl"] = 0
            rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv("amtl.csv")
    teeth = expand_to_teeth(df)

    model = smf.glm(
        formula="is_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=teeth,
        family=sm.families.Binomial(),
    )
    result = model.fit()

    print("=== GLM Binomial Results (per tooth) ===")
    print(result.summary())

    # Compute predicted probabilities for each genus at typical covariate values
    # (mean age, mean prob_male, and most common tooth_class).
    mean_age = teeth["age"].mean()
    mean_prob_male = teeth["prob_male"].mean()
    common_tooth_class = teeth["tooth_class"].mode().iat[0]

    print("\n=== Predicted AMTL probability by genus ===")
    rows = []
    for genus in sorted(teeth["genus"].unique()):
        row = {
            "genus": genus,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": common_tooth_class,
        }
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_probs = result.get_prediction(pred_df).summary_frame(alpha=0.05)

    summary = pd.concat(
        [
            pred_df[["genus"]],
            pred_probs[["mean", "mean_ci_lower", "mean_ci_upper"]],
        ],
        axis=1,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
