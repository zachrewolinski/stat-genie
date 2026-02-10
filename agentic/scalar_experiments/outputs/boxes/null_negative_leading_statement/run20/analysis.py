from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Create a binary outcome capturing preference for the majority option
    df["follow_majority"] = (df["y"] == 2).astype(int)

    # Treat culture as a categorical predictor, age as continuous (developmental stage)
    formula = "follow_majority ~ age + C(culture)"
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Likelihood-ratio test for joint effect of age and culture
    null_model = smf.logit("follow_majority ~ 1", data=df).fit(disp=False)
    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = model.df_model - null_model.df_model
    lr_pvalue = chi2.sf(lr_stat, df_diff)

    print("Logit model summary:")
    print(model.summary())
    print("\nLikelihood-ratio test for age and culture:")
    print(f"LR stat = {lr_stat:.3f}, df = {df_diff}, p = {lr_pvalue:.4g}")

    # Map evidence strength to a Likert scale in [-100, 100]
    # We interpret small p-values as strong evidence that preferences vary
    if lr_pvalue < 1e-4:
        scalar = 90
    elif lr_pvalue < 1e-3:
        scalar = 75
    elif lr_pvalue < 1e-2:
        scalar = 60
    elif lr_pvalue < 5e-2:
        scalar = 40
    elif lr_pvalue < 0.1:
        scalar = 20
    else:
        # No strong evidence of variation; lean toward "No"
        if lr_pvalue > 0.5:
            scalar = -70
        else:
            scalar = -40

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(int(scalar)), encoding="utf-8")


if __name__ == "__main__":
    main()
