import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Encode whether each child chose the majority option (social majority cue)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Fit logistic regression for majority choices with age and culture (plus basic controls)
    formula_full = "majority_choice ~ age + C(culture) + C(gender) + majority_first"
    formula_noculture = "majority_choice ~ age + C(gender) + majority_first"
    formula_noage = "majority_choice ~ C(culture) + C(gender) + majority_first"

    # Fit models; use try/except to be robust to convergence issues
    try:
        model_full = smf.logit(formula_full, data=df).fit(disp=False)
        model_noculture = smf.logit(formula_noculture, data=df).fit(disp=False)
        model_noage = smf.logit(formula_noage, data=df).fit(disp=False)
    except Exception:
        # Fallback to simpler specification if convergence problems occur
        formula_full = "majority_choice ~ age + C(culture)"
        formula_noculture = "majority_choice ~ age"
        formula_noage = "majority_choice ~ C(culture)"
        model_full = smf.logit(formula_full, data=df).fit(disp=False)
        model_noculture = smf.logit(formula_noculture, data=df).fit(disp=False)
        model_noage = smf.logit(formula_noage, data=df).fit(disp=False)

    # Overall reliance on majority option (social information)
    majority_prop = df["majority_choice"].mean()

    # Likelihood-ratio tests for culture and age effects
    lr_culture = 2.0 * (model_full.llf - model_noculture.llf)
    df_culture = model_full.df_model - model_noculture.df_model
    p_culture = stats.chi2.sf(lr_culture, df_culture) if df_culture > 0 else 1.0

    lr_age = 2.0 * (model_full.llf - model_noage.llf)
    df_age = model_full.df_model - model_noage.df_model
    p_age = stats.chi2.sf(lr_age, df_age) if df_age > 0 else 1.0

    # Map evidence to a Likert-style scalar in [-100, 100]
    support = 0.0

    # Evidence that majority cues are used (reliance on social information)
    if majority_prop > 1.0 / 3.0:
        support += 0.1
    if majority_prop > 0.5:
        support += 0.1

    # Evidence that majority preference varies with age
    if p_age < 0.05:
        support += 0.3
    if p_age < 0.01:
        support += 0.2

    # Evidence that majority preference varies across cultures
    if p_culture < 0.05:
        support += 0.3
    if p_culture < 0.01:
        support += 0.2

    # Cap support between 0 and 1
    support = float(np.clip(support, 0.0, 1.0))

    # Convert support to scalar in [-100, 100]; here we only encode positive support
    scalar = int(round(100 * support))

    # Write final scalar conclusion to file (single integer, no extra text)
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

