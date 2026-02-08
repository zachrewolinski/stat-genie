import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy


def fit_model(df):
    # Binomial GLM with successes/failures to avoid proportion edge cases
    present = df["total"] - df["missing"]
    endog = np.column_stack([df["missing"], present])
    exog = patsy.dmatrix("C(genus) + age + sex + C(tooth)", data=df, return_type="dataframe")
    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()
    return result, exog.design_info


def adjusted_mean_prob(result, design_info, base_df, genus_value):
    df = base_df.copy()
    df["genus"] = genus_value
    # Use model to predict per-socket probability
    exog = patsy.dmatrix(design_info, df, return_type="dataframe")
    pred = result.predict(exog)
    return float(np.mean(pred))


def main():
    df = pd.read_csv("amtl.csv")
    df = df.rename(
        columns={
            "feature1": "tooth",
            "feature2": "specimen",
            "feature3": "missing",
            "feature4": "total",
            "feature5": "age",
            "feature6": "age_unc",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning
    df = df.dropna(subset=["missing", "total", "age", "sex", "tooth", "genus"]).copy()
    df = df[(df["total"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["total"])].copy()

    # Fit model on full data
    result, design_info = fit_model(df)

    # Adjusted probabilities by genus using g-computation over observed covariates
    genera = sorted(df["genus"].unique())
    adj_probs = {g: adjusted_mean_prob(result, design_info, df, g) for g in genera}

    # Define non-human genera
    nonhuman = [g for g in genera if g != "Homo sapiens"]
    homo_prob = adj_probs.get("Homo sapiens")

    if homo_prob is None or len(nonhuman) == 0:
        raise RuntimeError("Expected Homo sapiens and non-human genera present in data.")

    nonhuman_prob = float(np.mean([adj_probs[g] for g in nonhuman]))
    diff = homo_prob - nonhuman_prob

    # Bootstrap for strength of evidence that Homo > non-human
    rng = np.random.default_rng(7)
    B = 500
    diffs = []
    for _ in range(B):
        sample_idx = rng.integers(0, len(df), size=len(df))
        boot = df.iloc[sample_idx].copy()
        try:
            boot_result, boot_design_info = fit_model(boot)
            boot_adj = {g: adjusted_mean_prob(boot_result, boot_design_info, boot, g) for g in genera}
            boot_homo = boot_adj.get("Homo sapiens")
            boot_nonhuman = float(np.mean([boot_adj[g] for g in nonhuman]))
            diffs.append(boot_homo - boot_nonhuman)
        except Exception:
            # Skip failed bootstrap fits
            continue

    diffs = np.array(diffs)
    if diffs.size == 0:
        p_gt_zero = 0.5
    else:
        p_gt_zero = float(np.mean(diffs > 0))

    # Map probability to Likert scale [-100, 100]
    likert = int(np.round(200 * (p_gt_zero - 0.5)))
    likert = max(-100, min(100, likert))

    # Write conclusion
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(likert))


if __name__ == "__main__":
    main()
