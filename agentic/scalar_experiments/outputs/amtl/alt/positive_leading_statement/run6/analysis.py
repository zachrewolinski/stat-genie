import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import exp


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Create human vs non-human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth for binomial regression
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Basic sanity checks printed to stdout
    print("Unique genera:", df["genus"].unique())
    print("Rows:", len(df))
    print("Overall mean prop_missing by human status:")
    print(df.groupby("is_human")["prop_missing"].mean())

    # Binomial regression with grouped data:
    # response is proportion missing with sockets as frequency weights
    formula = "prop_missing ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    # Use cluster-robust standard errors at the specimen level
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    print("\n=== GLM Binomial Results (cluster-robust by specimen) ===")
    print(result.summary())

    # Compute odds ratio and 95% CI for is_human
    if "is_human" in result.params:
        coef = result.params["is_human"]
        se = result.bse["is_human"]
        or_human = float(exp(coef))
        ci_low = float(exp(coef - 1.96 * se))
        ci_high = float(exp(coef + 1.96 * se))
        pval = float(result.pvalues["is_human"])

        print("\nEffect of being human (is_human):")
        print(f"  log-odds coefficient: {coef:.4f}")
        print(f"  odds ratio: {or_human:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f})")
        print(f"  p-value: {pval:.3g}")

        # Predicted probabilities at mean covariates by human status
        mean_vals = df[["age", "prob_male"]].mean()
        base = {
            "age": mean_vals["age"],
            "prob_male": mean_vals["prob_male"],
            "tooth_class": "Anterior",
        }

        def predict_for(is_human_val: int) -> float:
            row = base.copy()
            row["is_human"] = is_human_val
            # Create a small DataFrame for prediction
            pred_df = pd.DataFrame([row])
            p = result.predict(pred_df)[0]
            return float(p)

        p_nonhuman = predict_for(0)
        p_human = predict_for(1)
        print(
            f"\nPredicted missing-tooth proportion (Anterior, mean age/sex):"
            f"\n  Non-human: {p_nonhuman:.3f}"
            f"\n  Human:     {p_human:.3f}"
            f"\n  Difference: {p_human - p_nonhuman:.3f}"
        )


if __name__ == "__main__":
    main()
