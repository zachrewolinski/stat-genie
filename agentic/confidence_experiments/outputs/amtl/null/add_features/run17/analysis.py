import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic cleaning and variable setup
    df = df.copy()
    df["genus"] = df["genus"].astype(str).str.strip()
    df["tooth_class"] = df["tooth_class"].astype(str).str.strip()

    # Focus on the four genera in the research question
    target_genus = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(target_genus)].copy()

    # Drop rows with missing key variables or non-positive socket counts
    key_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    df = df.dropna(subset=key_cols)
    df = df[df["sockets"] > 0]

    # Proportion of teeth lost
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Descriptive summaries
    print("Number of rows after filtering:", len(df))
    print("Genus counts:")
    print(df["genus"].value_counts())
    print("\nMean AMTL proportion by genus:")
    print(df.groupby("genus")["prop_amtl"].mean())
    print("\nMean AMTL proportion by genus and tooth_class:")
    print(df.groupby(["genus", "tooth_class"])["prop_amtl"].mean())

    # Design matrix: human indicator, age, sex proxy, and tooth class
    predictors = ["is_human", "age", "prob_male", "tooth_class"]
    X = pd.get_dummies(df[predictors], drop_first=True)
    X = sm.add_constant(X)

    # Binomial response as (successes, failures)
    successes = df["num_amtl"].to_numpy()
    failures = (df["sockets"] - df["num_amtl"]).to_numpy()
    y = np.column_stack([successes, failures])

    # Fit binomial GLM
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    print("\nGLM convergence status:", getattr(result, "converged", "NA"))
    print("\nCoefficient estimates:")
    print(result.params)
    print("\nStandard errors:")
    print(result.bse)
    print("\nP-values:")
    print(result.pvalues)

    # Extract human effect
    if "is_human" in result.params.index:
        coef_human = result.params["is_human"]
        pval_human = result.pvalues["is_human"]
        or_human = float(np.exp(coef_human))
        ci_low, ci_high = np.exp(result.conf_int().loc["is_human"])

        print("\nEffect of being human (Homo sapiens) vs non-human primates:")
        print(f"  Log-odds coefficient: {coef_human:.4f}")
        print(f"  Odds ratio: {or_human:.3f}")
        print(f"  95% CI for OR: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  p-value: {pval_human:.4g}")

        # Average marginal effect on AMTL probability
        X_human = X.copy()
        X_human["is_human"] = 1
        X_nonhuman = X.copy()
        X_nonhuman["is_human"] = 0

        mean_prob_human = float(result.predict(X_human).mean())
        mean_prob_nonhuman = float(result.predict(X_nonhuman).mean())
        diff = mean_prob_human - mean_prob_nonhuman

        print(
            "\nAverage predicted AMTL probability (controlling for age, sex, tooth class):"
        )
        print(f"  Humans:     {mean_prob_human:.4f}")
        print(f"  Non-humans: {mean_prob_nonhuman:.4f}")
        print(f"  Difference: {diff:.4f} (absolute percentage-point difference)")
    else:
        print("\nNo is_human term found in fitted model; something went wrong with design matrix.")


if __name__ == "__main__":
    main()
