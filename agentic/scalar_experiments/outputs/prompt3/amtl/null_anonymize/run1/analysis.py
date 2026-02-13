import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Rename columns to semantic names based on info.json
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "observable",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: ensure counts are valid
    df = df[(df["observable"] > 0) & (df["missing"] >= 0)]
    df = df[df["missing"] <= df["observable"]]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth for given tooth class
    df["prop_missing"] = df["missing"] / df["observable"]

    # Quick descriptive statistics by genus
    genus_summary = (
        df.groupby("genus")
        .agg(
            mean_prop_missing=("prop_missing", "mean"),
            median_prop_missing=("prop_missing", "median"),
            n_records=("prop_missing", "size"),
            total_missing=("missing", "sum"),
            total_observable=("observable", "sum"),
        )
        .reset_index()
    )

    # Fit binomial regression (GLM with binomial family)
    # Response is the proportion of missing teeth with observable teeth as binomial trials.
    model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_estimate + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observable"],
    )
    result = model.fit()

    # Extract key statistics for the human effect
    coef_human = result.params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    pvalue_human = result.pvalues.get("is_human", np.nan)
    ci_lower, ci_upper = result.conf_int().loc["is_human"].tolist()
    odds_ratio = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    # Predicted probabilities for a typical case (mean age, mean sex, most common tooth class)
    mode_tooth_class = df["tooth_class"].mode().iat[0]
    mean_age = df["age"].mean()
    mean_sex = df["sex_estimate"].mean()

    new_data = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "sex_estimate": [mean_sex, mean_sex],
            "tooth_class": [mode_tooth_class, mode_tooth_class],
        }
    )
    pred = result.get_prediction(new_data).summary_frame()
    # Predicted probability of AMTL for non-humans vs humans
    prob_nonhuman = float(pred.loc[0, "mean"])
    prob_human = float(pred.loc[1, "mean"])
    prob_diff = prob_human - prob_nonhuman

    # Print a concise summary for inspection
    print("Genus-level descriptive summary:")
    print(genus_summary.to_string(index=False))
    print("\nBinomial regression results (key terms):")
    print(result.summary().tables[1])
    print("\nEffect of being human (is_human):")
    print(f"  Coefficient (log-odds): {coef_human:.4f}")
    print(f"  Std. error: {se_human:.4f}")
    print(f"  p-value: {pvalue_human:.4g}")
    print(f"  95% CI (log-odds): [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Odds ratio: {odds_ratio:.3f}")
    print("\nPredicted AMTL probability for typical case:")
    print(f"  Non-human primates: {prob_nonhuman:.4f}")
    print(f"  Humans: {prob_human:.4f}")
    print(f"  Difference (human - non-human): {prob_diff:.4f}")


if __name__ == "__main__":
    main()

