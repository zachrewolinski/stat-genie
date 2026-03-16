import json

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Proportion of missing teeth per observation
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Binomial GLM with Homo sapiens as reference genus, adjusting for age, sex proxy, and tooth class
    formula = (
        "amtl_prop ~ C(genus, Treatment(reference='Homo sapiens')) + "
        "age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    # Cluster-robust SEs by specimen to account for repeated tooth classes within individuals
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    # Compute standardized predicted probabilities for each genus by averaging
    # predictions over the empirical distribution of age, sex, and tooth class,
    # weighting by the number of sockets.
    genera = sorted(df["genus"].unique())

    def avg_pred_for_genus(genus_name: str) -> float:
        df_new = df.copy()
        df_new["genus"] = genus_name
        pred = result.predict(df_new)
        # Socket-weighted average probability of a tooth being missing
        avg_prob = float((pred * df_new["sockets"]).sum() / df_new["sockets"].sum())
        return avg_prob

    avg_preds = {g: avg_pred_for_genus(g) for g in genera}

    # Extract genus contrasts vs Homo sapiens from the GLM coefficients
    coef_info = {}
    for g in genera:
        if g == "Homo sapiens":
            continue
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        if term in params.index:
            ci_low, ci_high = conf_int.loc[term]
            coef_info[g] = {
                "coef": float(params[term]),
                "pvalue": float(pvalues[term]),
                "ci_lower": float(ci_low),
                "ci_upper": float(ci_high),
            }

    # Differences in standardized predicted probabilities (Homo minus genus)
    homo_pred = avg_preds.get("Homo sapiens")
    diff_preds = {}
    if homo_pred is not None:
        for g, val in avg_preds.items():
            if g == "Homo sapiens":
                continue
            diff_preds[g] = float(homo_pred - val)

    results = {
        "avg_preds": avg_preds,
        "coef_info_vs_homo": coef_info,
        "diff_preds_homo_minus_genus": diff_preds,
        "model_converged": bool(result.mle_retvals.get("converged", True)
                                 if hasattr(result, "mle_retvals")
                                 else True),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
