import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    print("Number of rows:", len(df))
    print("Columns:", list(df.columns))

    # Create human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Quick group summaries of AMTL per socket by genus
    df["amtl_per_socket"] = df["num_amtl"] / df["sockets"]
    genus_summary = (
        df.groupby("genus")["amtl_per_socket"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    print("\nAMTL per socket by genus:")
    print(genus_summary)

    # Poisson regression treating num_amtl as a count with exposure = sockets
    df["log_sockets"] = np.log(df["sockets"])
    poisson_formula = "num_amtl ~ is_human + age + prob_male + C(tooth_class)"
    poisson_model = smf.glm(
        formula=poisson_formula,
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_sockets"],
    )
    poisson_results = poisson_model.fit()
    print("\nPoisson regression results (num_amtl, offset log(sockets)):")
    print(poisson_results.summary())

    if "is_human" in poisson_results.params:
        coef_human = poisson_results.params["is_human"]
        se_human = poisson_results.bse["is_human"]
        rate_ratio = float(np.exp(coef_human))
        print(f"\nPoisson is_human coef: {coef_human:.4f} (SE {se_human:.4f})")
        print(f"Rate ratio (human vs non-human): {rate_ratio:.3f}")

        # Marginal predicted AMTL per socket for human vs non-human,
        # averaging over the covariate distribution in the sample.
        df_human = df.copy()
        df_human["is_human"] = 1
        df_nonhuman = df.copy()
        df_nonhuman["is_human"] = 0

        pred_human = poisson_results.predict(df_human, offset=df["log_sockets"])
        pred_nonhuman = poisson_results.predict(df_nonhuman, offset=df["log_sockets"])

        mean_rate_human = float(pred_human.sum() / df["sockets"].sum())
        mean_rate_nonhuman = float(pred_nonhuman.sum() / df["sockets"].sum())
        print(
            f"Predicted AMTL per socket - human: {mean_rate_human:.3f}, "
            f"non-human: {mean_rate_nonhuman:.3f}"
        )

    # Binomial sensitivity analysis on rows with num_amtl <= sockets
    df_binom = df[df["num_amtl"] <= df["sockets"]].copy()
    df_binom = df_binom[df_binom["sockets"] > 0]
    if not df_binom.empty:
        df_binom["prop_amtl"] = df_binom["num_amtl"] / df_binom["sockets"]
        binom_formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
        binom_model = smf.glm(
            formula=binom_formula,
            data=df_binom,
            family=sm.families.Binomial(),
            freq_weights=df_binom["sockets"],
        )
        binom_results = binom_model.fit()
        print(
            "\nBinomial regression results (subset with num_amtl <= sockets, "
            "using sockets as weights):"
        )
        print(binom_results.summary())

        if "is_human" in binom_results.params:
            coef_human_b = binom_results.params["is_human"]
            se_human_b = binom_results.bse["is_human"]
            odds_ratio = float(np.exp(coef_human_b))
            print(
                f"\nBinomial is_human coef: {coef_human_b:.4f} "
                f"(SE {se_human_b:.4f})"
            )
            print(f"Odds ratio (human vs non-human): {odds_ratio:.3f}")


if __name__ == "__main__":
    main()

