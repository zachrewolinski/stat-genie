import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path


def map_p_to_strength(p: float) -> float:
    """
    Map a p-value to a [0, 1] evidence strength for an effect.
    Higher values mean stronger evidence that variation exists.
    """
    if p < 1e-6:
        return 1.0
    if p < 1e-4:
        return 0.9
    if p < 1e-3:
        return 0.8
    if p < 1e-2:
        return 0.6
    if p < 5e-2:
        return 0.4
    if p < 1e-1:
        return 0.2
    return 0.0


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Reliance on social information: choosing any demonstrated option
    df["social_use"] = df["y"].isin([2, 3]).astype(int)

    # Preference for majority among those who use social information
    df_social = df[df["social_use"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Logistic regression: social information use ~ age + culture (+ control for majority_first)
    model_social = smf.logit(
        "social_use ~ age + C(culture) + majority_first",
        data=df,
    ).fit(disp=False)

    # Logistic regression: majority vs minority choice ~ age + culture (+ control)
    model_pref = smf.logit(
        "majority_choice ~ age + C(culture) + majority_first",
        data=df_social,
    ).fit(disp=False)

    # Collect p-values for age and culture effects in both models
    pvals_social = model_social.pvalues
    pvals_pref = model_pref.pvalues

    age_p_social = float(pvals_social.get("age", 1.0))
    age_p_pref = float(pvals_pref.get("age", 1.0))

    culture_terms_social = [k for k in pvals_social.index if k.startswith("C(culture)")]
    culture_terms_pref = [k for k in pvals_pref.index if k.startswith("C(culture)")]

    min_culture_p_social = float(
        pvals_social[culture_terms_social].min()
    ) if culture_terms_social else 1.0
    min_culture_p_pref = float(
        pvals_pref[culture_terms_pref].min()
    ) if culture_terms_pref else 1.0

    # Overall strongest evidence that reliance/preference vary with age or culture
    min_p = min(age_p_social, age_p_pref, min_culture_p_social, min_culture_p_pref)
    strength = map_p_to_strength(min_p)

    # Convert to Likert-style scalar: -100 (strong "No") to 100 (strong "Yes")
    # Here, strength reflects evidence that the answer is "Yes" — that
    # reliance on social information and/or majority preference do vary.
    if strength == 0.0:
        scalar = 0
    else:
        scalar = int(round(strength * 100))

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(scalar))


if __name__ == "__main__":
    main()

