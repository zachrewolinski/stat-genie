import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def map_p_to_points(p: float) -> int:
    if np.isnan(p):
        return 0
    if p < 1e-6:
        return 25
    if p < 1e-3:
        return 15
    if p < 0.05:
        return 5
    if p > 0.5:
        return -5
    return 0


def main() -> None:
    df = pd.read_csv("boxes.csv")

    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_choice"] = (df["majority_first"] != 1).astype(int)
    df["majority_vs_minority"] = (
        df["majority_first"].replace({2: 1, 3: 0}).where(df["social_choice"] == 1, np.nan)
    )

    score = 0

    try:
        model_social = smf.glm(
            "social_choice ~ age + C(y)", data=df, family=sm.families.Binomial()
        ).fit()
        p_age_social = float(model_social.pvalues.get("age", np.nan))
        p_site_social = float(
            model_social.pvalues.filter(like="C(y)").min()
            if model_social.pvalues.filter(like="C(y)").size > 0
            else np.nan
        )
        score += map_p_to_points(p_age_social)
        score += map_p_to_points(p_site_social)
    except Exception:
        pass

    try:
        df_mm = df.dropna(subset=["majority_vs_minority"])
        if not df_mm.empty and df_mm["majority_vs_minority"].nunique() > 1:
            model_majority = smf.glm(
                "majority_vs_minority ~ age + C(y)",
                data=df_mm,
                family=sm.families.Binomial(),
            ).fit()
            p_age_majority = float(model_majority.pvalues.get("age", np.nan))
            p_site_majority = float(
                model_majority.pvalues.filter(like="C(y)").min()
                if model_majority.pvalues.filter(like="C(y)").size > 0
                else np.nan
            )
            score += map_p_to_points(p_age_majority)
            score += map_p_to_points(p_site_majority)
    except Exception:
        pass

    score = int(max(-100, min(100, score)))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()

