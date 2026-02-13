import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Ensure expected columns are present
    required_cols = [
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "stdev_age",
        "prob_male",
        "genus",
        "pop",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Drop any rows with non-positive socket counts (defensive; metadata says min is 2)
    df = df[df["sockets"] > 0].copy()

    # Proportion of missing teeth for binomial modeling
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Treat genus and tooth_class as ordered categoricals with Homo sapiens as genus reference
    genus_order = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df["genus"] = pd.Categorical(df["genus"], categories=genus_order, ordered=False)
    tooth_classes = ["Anterior", "Posterior", "Premolar"]
    df["tooth_class"] = pd.Categorical(df["tooth_class"], categories=tooth_classes, ordered=False)

    # Basic sanity checks: raw AMTL rates by genus
    genus_summary = (
        df.groupby("genus")
        .apply(lambda g: pd.Series(
            {
                "n_rows": len(g),
                "total_missing": g["num_amtl"].sum(),
                "total_sockets": g["sockets"].sum(),
                "raw_amtl_rate": g["num_amtl"].sum() / g["sockets"].sum(),
                "mean_age": g["age"].mean(),
                "mean_prob_male": g["prob_male"].mean(),
            }
        ))
        .sort_index()
    )

    print("Raw AMTL summary by genus:")
    print(genus_summary.to_string(float_format=lambda x: f"{x:0.4f}"))
    print()

    # Binomial regression: AMTL proportion ~ genus + age + sex + tooth_class
    # Use sockets as frequency weights so rows with more observable teeth have more influence.
    formula = "amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial GLM results (logit link, Homo sapiens as reference for genus):")
    print(result.summary())
    print()

    # Extract and print genus coefficients and confidence intervals
    params = result.params
    conf_int = result.conf_int()
    print("Genus coefficients relative to Homo sapiens (log-odds scale):")
    for genus in genus_order[1:]:
        term = f"C(genus)[T.{genus}]"
        if term in params.index:
            est = params[term]
            lo, hi = conf_int.loc[term]
            pval = result.pvalues[term]
            print(
                f"{genus:10s}: coef = {est: .4f}, 95% CI = [{lo: .4f}, {hi: .4f}], p = {pval: .4g}"
            )
    print()

    # Model-based adjusted AMTL rates by genus:
    # For each genus, set genus to that level but keep age, sex, and tooth_class
    # at their observed values, and average the predicted probabilities.
    print("Model-adjusted AMTL rates by genus (predicted proportion of missing teeth):")
    adjusted_rates = {}
    for genus in genus_order:
        tmp = df.copy()
        tmp["genus"] = genus
        preds = result.predict(tmp)
        # Weight by sockets to reflect number of teeth represented
        adjusted_rate = np.average(preds, weights=tmp["sockets"])
        adjusted_rates[genus] = adjusted_rate
        print(f"{genus:12s}: adjusted AMTL rate = {adjusted_rate:0.4f}")

    print()
    print("Summary:")
    print("Raw AMTL rates by genus:")
    for genus in genus_order:
        if genus in genus_summary.index:
            raw_rate = genus_summary.loc[genus, "raw_amtl_rate"]
            print(f"  {genus:12s}: raw = {raw_rate:0.4f}, adjusted = {adjusted_rates[genus]:0.4f}")


if __name__ == "__main__":
    main()

