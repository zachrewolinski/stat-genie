import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("amtl.csv")
    # rename for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "total",
            "feature5": "age",
            "feature6": "age_uncert",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Drop rows with missing or invalid totals
    df = df.dropna(subset=["missing", "total", "age", "sex", "tooth_class", "genus"]).copy()
    df = df[df["total"] > 0].copy()

    # Create indicator for Homo sapiens
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion missing
    df["prop_missing"] = df["missing"] / df["total"]

    # Build GLM with binomial family, using totals as freq weights
    formula = "prop_missing ~ is_human + age + sex + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total"],
    )
    res = model.fit()

    coef = res.params.get("is_human", np.nan)
    pval = res.pvalues.get("is_human", np.nan)

    # Determine response based on significance and direction
    response = "Yes" if (coef > 0 and pval < 0.05) else "No"

    # Map p-value and effect direction to 0-100 scale
    # Stronger evidence -> closer to 100 or 0
    if np.isnan(pval) or np.isnan(coef):
        scale = 50
    else:
        if coef > 0:
            # Positive effect
            if pval < 1e-6:
                scale = 95
            elif pval < 1e-4:
                scale = 90
            elif pval < 1e-3:
                scale = 85
            elif pval < 1e-2:
                scale = 75
            elif pval < 0.05:
                scale = 65
            else:
                scale = 45
        else:
            # Negative or null effect
            if pval < 1e-6:
                scale = 5
            elif pval < 1e-4:
                scale = 10
            elif pval < 1e-3:
                scale = 15
            elif pval < 1e-2:
                scale = 25
            elif pval < 0.05:
                scale = 35
            else:
                scale = 45

    # Keep scale aligned with response
    if response == "Yes" and scale < 50:
        scale = 60
    if response == "No" and scale > 50:
        scale = 40

    out = {"response": response, "scale": int(scale)}
    with open("conclusion.txt", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
