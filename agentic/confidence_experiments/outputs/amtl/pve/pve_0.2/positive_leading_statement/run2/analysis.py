import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import patsy
from scipy import stats


def main():
    df = pd.read_csv("amtl.csv")

    # Ensure categorical types for modeling
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # OLS with cluster-robust SEs at specimen level to account for repeated measures
    formula = "num_amtl ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"
    model = smf.ols(formula, data=df)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    params = fit.params
    cov = fit.cov_params()

    # Build design matrices for marginal (covariate-adjusted) means
    design_info = fit.model.data.design_info

    def design_matrix_for_genus(genus_name: str):
        tmp = df.copy()
        tmp["genus"] = genus_name
        mats = patsy.build_design_matrices([design_info], tmp, return_type="dataframe")
        return mats[0]

    # Marginal mean predictions for each genus
    genus_levels = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    adjusted_means = {}
    mean_design = {}

    for g in genus_levels:
        Xg = design_matrix_for_genus(g)
        mean_design[g] = Xg.mean(axis=0)
        adjusted_means[g] = float(np.dot(mean_design[g], params))

    # Contrast helper
    def contrast_stats(g1, g2):
        d = mean_design[g1] - mean_design[g2]
        diff = float(np.dot(d, params))
        var = float(np.dot(d, np.dot(cov, d)))
        se = float(np.sqrt(var)) if var >= 0 else float("nan")
        z = diff / se if se and se > 0 else float("nan")
        p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else float("nan")
        return diff, se, z, p

    # Homo vs each non-human
    contrasts = {}
    for g in ["Pan", "Papio", "Pongo"]:
        diff, se, z, p = contrast_stats("Homo sapiens", g)
        contrasts[g] = {
            "diff_homo_minus_genus": diff,
            "se": se,
            "z": z,
            "p": p,
        }

    # Homo vs average non-human genera
    nonhuman_mean = (mean_design["Pan"] + mean_design["Papio"] + mean_design["Pongo"]) / 3.0
    d_avg = mean_design["Homo sapiens"] - nonhuman_mean
    diff_avg = float(np.dot(d_avg, params))
    var_avg = float(np.dot(d_avg, np.dot(cov, d_avg)))
    se_avg = float(np.sqrt(var_avg)) if var_avg >= 0 else float("nan")
    z_avg = diff_avg / se_avg if se_avg and se_avg > 0 else float("nan")
    p_avg = 2 * (1 - stats.norm.cdf(abs(z_avg))) if np.isfinite(z_avg) else float("nan")

    result = {
        "adjusted_means": adjusted_means,
        "contrasts_homo_vs_each": contrasts,
        "contrast_homo_vs_avg_nonhuman": {
            "diff": diff_avg,
            "se": se_avg,
            "z": z_avg,
            "p": p_avg,
        },
        "model_summary": {
            "n": int(fit.nobs),
            "r2": float(fit.rsquared),
            "r2_adj": float(fit.rsquared_adj),
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Also print a concise view for debugging
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
