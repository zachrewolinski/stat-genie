import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy


def main():
    df = pd.read_csv("amtl.csv")

    # Response as proportion with binomial trials = sockets
    df = df.copy()
    df["proportion"] = df["num_amtl"] / df["sockets"]

    # Ensure categorical ordering so a non-human genus is the reference
    genus_order = ["Pan", "Papio", "Pongo", "Homo sapiens"]
    tooth_order = ["Anterior", "Posterior", "Premolar"]
    df["genus"] = pd.Categorical(df["genus"], categories=genus_order, ordered=True)
    df["tooth_class"] = pd.Categorical(df["tooth_class"], categories=tooth_order, ordered=True)

    formula = (
        "proportion ~ C(genus, Treatment(reference='Pan')) "
        "+ age + prob_male + C(tooth_class, Treatment(reference='Anterior'))"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    # Build design matrices for marginal predictions with genus fixed
    design_info = model.model.data.design_info

    def build_exog(genus_value: str):
        df_tmp = df.copy()
        df_tmp["genus"] = genus_value
        return patsy.build_design_matrices([design_info], df_tmp, return_type="dataframe")[0]

    exog_homo = build_exog("Homo sapiens")
    exog_pan = build_exog("Pan")
    exog_papio = build_exog("Papio")
    exog_pongo = build_exog("Pongo")

    def mean_pred(exog, params):
        linpred = np.asarray(exog) @ params
        mu = model.family.link.inverse(linpred)
        return mu.mean()

    params = model.params.values

    mean_homo = mean_pred(exog_homo, params)
    mean_pan = mean_pred(exog_pan, params)
    mean_papio = mean_pred(exog_papio, params)
    mean_pongo = mean_pred(exog_pongo, params)

    mean_nonhuman = np.mean([mean_pan, mean_papio, mean_pongo])
    diff_point = mean_homo - mean_nonhuman

    # Parametric bootstrap for uncertainty on the mean difference
    rng = np.random.default_rng(42)
    cov = model.cov_params().values
    draws = rng.multivariate_normal(params, cov, size=1000)
    diff_samples = []
    for b in draws:
        mh = mean_pred(exog_homo, b)
        mnh = np.mean([
            mean_pred(exog_pan, b),
            mean_pred(exog_papio, b),
            mean_pred(exog_pongo, b),
        ])
        diff_samples.append(mh - mnh)
    diff_samples = np.asarray(diff_samples)

    ci_low, ci_high = np.percentile(diff_samples, [2.5, 97.5])
    prob_positive = (diff_samples > 0).mean()

    print("Model summary (truncated):")
    print(model.summary().tables[1])
    print()
    print(f"Mean predicted AMTL rate (Homo sapiens): {mean_homo:.4f}")
    print(f"Mean predicted AMTL rate (non-human mean): {mean_nonhuman:.4f}")
    print(f"Difference (Homo - non-human mean): {diff_point:.4f}")
    print(f"95% CI for difference: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Pr(difference > 0) from bootstrap: {prob_positive:.3f}")


if __name__ == "__main__":
    main()
