import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def fit_models(df, outcome):
    """Fit nested GLM models to assess age and site (y) effects."""
    df = df.copy()
    formula_null = f"{outcome} ~ 1"
    formula_age = f"{outcome} ~ age"
    formula_age_site = f"{outcome} ~ age + C(y)"

    model_null = smf.glm(formula_null, data=df, family=sm.families.Binomial()).fit()
    model_age = smf.glm(formula_age, data=df, family=sm.families.Binomial()).fit()
    model_age_site = smf.glm(formula_age_site, data=df, family=sm.families.Binomial()).fit()

    def lr_test(full, reduced):
        lr_stat = 2 * (full.llf - reduced.llf)
        df_diff = full.df_model - reduced.df_model
        p_val = stats.chi2.sf(lr_stat, df_diff)
        return lr_stat, df_diff, p_val

    lr_age, df_age, p_age = lr_test(model_age, model_null)
    lr_site, df_site, p_site = lr_test(model_age_site, model_age)

    # Effect sizes: predicted probabilities across age and sites
    age_min, age_max = df["age"].min(), df["age"].max()
    age_grid = pd.DataFrame({"age": [age_min, age_max]})
    p_age_min = model_age.predict(pd.DataFrame({"age": [age_min]}))[0]
    p_age_max = model_age.predict(pd.DataFrame({"age": [age_max]}))[0]

    # Site variation at median age
    age_median = df["age"].median()
    sites = sorted(df["y"].unique())
    site_probs = []
    for s in sites:
        row = pd.DataFrame({"age": [age_median], "y": [s]})
        site_probs.append((s, model_age_site.predict(row)[0]))

    return {
        "models": {
            "null": model_null,
            "age": model_age,
            "age_site": model_age_site,
        },
        "tests": {
            "age": {"lr": lr_age, "df": df_age, "p": p_age},
            "site": {"lr": lr_site, "df": df_site, "p": p_site},
        },
        "effects": {
            "age_range": (age_min, age_max),
            "prob_age_min": float(p_age_min),
            "prob_age_max": float(p_age_max),
            "age_diff": float(p_age_max - p_age_min),
            "site_probs_at_median_age": site_probs,
        },
    }


def main():
    df = pd.read_csv("boxes.csv")

    # Outcomes
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_choice"] = (df["majority_first"] != 1).astype(int)

    print("N =", len(df))
    print("Overall choice distribution (1=unchosen,2=majority,3=minority):")
    print(df["majority_first"].value_counts().sort_index())
    print()

    print("Overall majority-choice rate:", df["majority_choice"].mean())
    print("Overall social-choice rate (majority or minority):", df["social_choice"].mean())
    print()

    print("=== Majority choice ~ age + site ===")
    maj_results = fit_models(df, "majority_choice")
    print("Age LRT: LR={lr:.3f}, df={df}, p={p:.4g}".format(
        lr=maj_results["tests"]["age"]["lr"],
        df=int(maj_results["tests"]["age"]["df"]),
        p=maj_results["tests"]["age"]["p"],
    ))
    print("Site LRT (given age): LR={lr:.3f}, df={df}, p={p:.4g}".format(
        lr=maj_results["tests"]["site"]["lr"],
        df=int(maj_results["tests"]["site"]["df"]),
        p=maj_results["tests"]["site"]["p"],
    ))
    print("Age range:", maj_results["effects"]["age_range"])
    print("Predicted majority prob at min age: {:.3f}".format(maj_results["effects"]["prob_age_min"]))
    print("Predicted majority prob at max age: {:.3f}".format(maj_results["effects"]["prob_age_max"]))
    print("Difference (max - min): {:.3f}".format(maj_results["effects"]["age_diff"]))
    site_probs = maj_results["effects"]["site_probs_at_median_age"]
    min_site = min(site_probs, key=lambda x: x[1])
    max_site = max(site_probs, key=lambda x: x[1])
    print("Site variation at median age (majority choice):")
    for s, p in site_probs:
        print(f"  site {s}: {p:.3f}")
    print("Min vs max site prob difference:", max_site[1] - min_site[1])
    print()

    print("=== Social choice (any demonstrated option) ~ age + site ===")
    soc_results = fit_models(df, "social_choice")
    print("Age LRT: LR={lr:.3f}, df={df}, p={p:.4g}".format(
        lr=soc_results["tests"]["age"]["lr"],
        df=int(soc_results["tests"]["age"]["df"]),
        p=soc_results["tests"]["age"]["p"],
    ))
    print("Site LRT (given age): LR={lr:.3f}, df={df}, p={p:.4g}".format(
        lr=soc_results["tests"]["site"]["lr"],
        df=int(soc_results["tests"]["site"]["df"]),
        p=soc_results["tests"]["site"]["p"],
    ))
    print("Age range:", soc_results["effects"]["age_range"])
    print("Predicted social prob at min age: {:.3f}".format(soc_results["effects"]["prob_age_min"]))
    print("Predicted social prob at max age: {:.3f}".format(soc_results["effects"]["prob_age_max"]))
    print("Difference (max - min): {:.3f}".format(soc_results["effects"]["age_diff"]))
    site_probs_soc = soc_results["effects"]["site_probs_at_median_age"]
    min_site_soc = min(site_probs_soc, key=lambda x: x[1])
    max_site_soc = max(site_probs_soc, key=lambda x: x[1])
    print("Site variation at median age (social choice):")
    for s, p in site_probs_soc:
        print(f"  site {s}: {p:.3f}")
    print("Min vs max site prob difference:", max_site_soc[1] - min_site_soc[1])


if __name__ == "__main__":
    main()

