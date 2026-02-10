import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["is_majority"] = (df["feature1"] == 2).astype(int)
    df["any_social"] = (df["feature1"] != 1).astype(int)

    # ----- Variation across cultures (sites) -----
    site_majority = df.groupby("feature5")["is_majority"].mean()
    site_social = df.groupby("feature5")["any_social"].mean()

    # Ranges of site-level rates (bounded by [0, 1])
    site_majority_range = float(site_majority.max() - site_majority.min())
    site_social_range = float(site_social.max() - site_social.min())

    # Normalize by the maximum possible range (0.5 is a large, practically strong difference)
    def norm_range(x: float) -> float:
        return max(0.0, min(1.0, x / 0.5))

    site_majority_norm = norm_range(site_majority_range)
    site_social_norm = norm_range(site_social_range)

    # Logistic regression for majority choice with site dummies (controlling for age)
    X_site = pd.get_dummies(df["feature5"].astype("category"), prefix="site", drop_first=True)
    X_site = pd.concat([df[["feature3"]], X_site], axis=1)
    X_site = sm.add_constant(X_site)
    model_site = sm.Logit(df["is_majority"], X_site).fit(disp=False)

    site_pvals = model_site.pvalues[[c for c in model_site.params.index if c.startswith("site_")]]
    frac_site_sig = float((site_pvals < 0.05).mean()) if len(site_pvals) > 0 else 0.0

    # Combine magnitude and significance into a culture-variation score
    culture_score = 0.5 * (site_majority_norm + site_social_norm) + 0.5 * frac_site_sig

    # ----- Variation across developmental stages (age) -----
    # Age-binned majority rates
    df["age_bin"] = pd.cut(
        df["feature3"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
    )
    age_bin_majority = df.groupby("age_bin")["is_majority"].mean()
    age_majority_range = float(age_bin_majority.max() - age_bin_majority.min())

    age_majority_norm = norm_range(age_majority_range)

    # Logistic regression for majority choice with age
    X_age = sm.add_constant(df[["feature3"]])
    model_age = sm.Logit(df["is_majority"], X_age).fit(disp=False)
    age_pval = float(model_age.pvalues.get("feature3", 1.0))

    age_sig = 1.0 if age_pval < 0.05 else 0.0
    age_score = 0.7 * age_majority_norm + 0.3 * age_sig

    # ----- Combine culture and developmental variation -----
    # Weight culture slightly more, as the empirical signal is clearer there.
    combined_score = 0.6 * culture_score + 0.4 * age_score

    # Map combined_score in [0,1] to Likert scalar in [0,100]
    combined_score = max(0.0, min(1.0, combined_score))
    scalar = int(round(combined_score * 100))

    # Write the scalar conclusion to file with no extra text or lines
    with open("conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

