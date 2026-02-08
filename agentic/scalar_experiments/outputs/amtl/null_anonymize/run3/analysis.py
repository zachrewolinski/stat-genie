import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "total",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )
    df = df.dropna(subset=["missing", "total", "age", "sex", "tooth_class", "genus"])
    df = df[df["total"] > 0]
    df = df[df["missing"] >= 0]
    df = df[df["missing"] <= df["total"]]
    return df


def fit_model(df: pd.DataFrame):
    # Binomial GLM on proportions with trial counts as variance weights
    df = df.copy()
    df["prop"] = df["missing"] / df["total"]
    # Keep proportions in (0,1) for numerical stability
    eps = 1e-6
    df["prop"] = df["prop"].clip(eps, 1 - eps)

    model = smf.glm(
        "prop ~ C(genus) + age + sex + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["total"],
    )
    return model.fit()


def predicted_mean_for_genus(result, df: pd.DataFrame, genus_value: str) -> float:
    tmp = df.copy()
    tmp["genus"] = genus_value
    preds = result.predict(tmp)
    return np.average(preds, weights=tmp["total"])


def analyze(df: pd.DataFrame, n_boot: int = 300, seed: int = 7):
    result = fit_model(df)

    genera = ["Pan", "Pongo", "Papio", "Homo sapiens"]
    for g in genera:
        if g not in df["genus"].unique():
            raise ValueError(f"Missing genus {g} in data")

    homo_mean = predicted_mean_for_genus(result, df, "Homo sapiens")
    nonhuman_means = [predicted_mean_for_genus(result, df, g) for g in ["Pan", "Pongo", "Papio"]]
    nonhuman_mean = float(np.mean(nonhuman_means))
    diff = homo_mean - nonhuman_mean
    ratio = homo_mean / nonhuman_mean if nonhuman_mean > 0 else np.nan

    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(df), size=len(df))
        boot_df = df.iloc[sample_idx].reset_index(drop=True)
        try:
            boot_res = fit_model(boot_df)
            boot_homo = predicted_mean_for_genus(boot_res, boot_df, "Homo sapiens")
            boot_non = np.mean(
                [predicted_mean_for_genus(boot_res, boot_df, g) for g in ["Pan", "Pongo", "Papio"]]
            )
            boot_diffs.append(boot_homo - boot_non)
        except Exception:
            continue

    boot_diffs = np.array(boot_diffs)
    ci_low, ci_high = (np.nan, np.nan)
    p_gt_zero = np.nan
    if len(boot_diffs) > 10:
        ci_low, ci_high = np.quantile(boot_diffs, [0.025, 0.975])
        p_gt_zero = float(np.mean(boot_diffs > 0))

    return {
        "n": len(df),
        "homo_mean": homo_mean,
        "nonhuman_mean": nonhuman_mean,
        "diff": diff,
        "ratio": ratio,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_gt_zero": p_gt_zero,
        "boot_n": len(boot_diffs),
    }


if __name__ == "__main__":
    df = load_data("amtl.csv")
    stats = analyze(df)
    for k, v in stats.items():
        print(f"{k}: {v}")
