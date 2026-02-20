import json

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: drop rows with missing key variables or zero sockets
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus", "specimen"])
    df = df[df["sockets"] > 0]

    # Ensure counts are valid
    df = df[df["num_amtl"] >= 0]
    df = df[df["num_amtl"] <= df["sockets"]]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Expand to one row per tooth (Bernoulli trials) to avoid numerical issues
    records: list[dict] = []
    for _, row in df.iterrows():
        base = {
            "specimen": row["specimen"],
            "age": row["age"],
            "prob_male": row["prob_male"],
            "tooth_class": row["tooth_class"],
            "is_human": row["is_human"],
        }
        # Missing teeth (AMTL = 1)
        for _ in range(int(row["num_amtl"])):
            rec = base.copy()
            rec["amtl"] = 1
            records.append(rec)
        # Present teeth (AMTL = 0)
        present = int(row["sockets"] - row["num_amtl"])
        for _ in range(present):
            rec = base.copy()
            rec["amtl"] = 0
            records.append(rec)

    long_df = pd.DataFrame.from_records(records)

    # Binomial (logistic) regression on per-tooth AMTL status
    formula = "amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=long_df,
        family=sm.families.Binomial(),
    )

    # Cluster-robust SEs by specimen to account for repeated observations per individual
    res = model.fit(cov_type="cluster", cov_kwds={"groups": long_df["specimen"]})

    print("Model summary (cluster-robust by specimen):")
    print(res.summary())

    # Extract effect of humans vs non-humans
    human_coef = res.params.get("is_human", float("nan"))
    human_se = res.bse.get("is_human", float("nan"))
    human_p = res.pvalues.get("is_human", float("nan"))

    print("\nEffect of modern humans (is_human=1) vs non-human primates:")
    print(f"  Coefficient (log-odds): {human_coef:.4f}")
    print(f"  Std. error:            {human_se:.4f}")
    print(f"  p-value:               {human_p:.4g}")

    # Predicted probabilities at representative values
    mean_age = long_df["age"].mean()
    mean_prob_male = long_df["prob_male"].mean()

    new_data = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            # Use Posterior as a common tooth class of interest
            "tooth_class": ["Posterior", "Posterior"],
        }
    )
    pred_means = res.get_prediction(new_data).summary_frame()
    nonhuman_prob = pred_means["mean"].iloc[0]
    human_prob = pred_means["mean"].iloc[1]
    diff = human_prob - nonhuman_prob

    print("\nPredicted AMTL probability (Posterior teeth, average age/sex):")
    print(f"  Non-human primates: {nonhuman_prob:.4f}")
    print(f"  Modern humans:      {human_prob:.4f}")
    print(f"  Absolute difference: {diff:.4f}")

    # Save key results to a small JSON file for inspection (not the final conclusion.txt)
    summary_dict = {
        "human_coef": human_coef,
        "human_p": human_p,
        "nonhuman_prob": float(nonhuman_prob),
        "human_prob": float(human_prob),
        "diff": float(diff),
    }
    with open("analysis_results.json", "w") as f:
        json.dump(summary_dict, f, indent=2)


if __name__ == "__main__":
    main()
