import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary outcome: any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children status
    group = df.groupby("children", observed=True)
    mean_any = group["any_affair"].mean()
    mean_affairs = group["affairs"].mean()
    counts = group["any_affair"].size()

    # Logistic regression controlling for key covariates
    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df)
    result = logit_model.fit(disp=False)

    coef_children = result.params.get("C(children)[T.yes]", np.nan)
    p_children = result.pvalues.get("C(children)[T.yes]", np.nan)
    odds_ratio = float(np.exp(coef_children)) if np.isfinite(coef_children) else np.nan

    # Decide answer based on direction and significance of the children effect
    # Baseline is children == "no"; coefficient is the change for "yes".
    if np.isnan(coef_children) or np.isnan(p_children):
        response = "No"
        confidence = 40
        reasoning = (
            "The logistic regression model could not estimate a clear effect for the "
            "children variable, so there is insufficient evidence that having children "
            "decreases engagement in extramarital affairs."
        )
    else:
        if coef_children < 0 and p_children < 0.05:
            response = "Yes"
            # Strong, statistically significant negative association
            confidence = 85
        elif coef_children < 0 and p_children < 0.10:
            response = "Yes"
            confidence = 70
        else:
            # Effect is not negative or not statistically distinguishable from zero
            response = "No"
            if p_children >= 0.10:
                confidence = 80
            else:
                confidence = 60

        reasoning = (
            "I modeled the probability of having any extramarital affair in the past "
            "year using logistic regression with a binary outcome (any affair vs. none). "
            "The main predictor was whether there are children in the marriage, and I "
            "controlled for gender, age, years married, religiousness, education, "
            "occupation, and self-rated marital happiness. "
        )
        reasoning += (
            f"In the raw data, the share of individuals reporting at least one affair "
            f"is {mean_any.get('yes', float('nan')):.3f} among those with children and "
            f"{mean_any.get('no', float('nan')):.3f} among those without children "
            f"(average number of affairs: {mean_affairs.get('yes', float('nan')):.3f} "
            f"with children vs. {mean_affairs.get('no', float('nan')):.3f} without). "
            f"Group sizes are {int(counts.get('yes', 0))} with children and "
            f"{int(counts.get('no', 0))} without children. "
        )
        reasoning += (
            f"In the adjusted logistic model, the coefficient for having children "
            f"(relative to not having children) is {coef_children:.3f}, corresponding "
            f"to an odds ratio of approximately {odds_ratio:.3f}, with a p-value of "
            f"{p_children:.3f}. "
        )
        if response == "Yes":
            reasoning += (
                "This negative coefficient indicates that, after adjusting for other "
                "factors, marriages with children have a lower probability of reporting "
                "extramarital affairs, and the p-value suggests this decrease is "
                "unlikely to be due to random chance alone."
            )
        else:
            reasoning += (
                "This coefficient is not negative and statistically convincing at "
                "conventional significance levels, so the data do not provide clear "
                "evidence that having children decreases engagement in extramarital "
                "affairs. If anything, the modeled effect is small relative to sampling "
                "uncertainty once other predictors are taken into account."
            )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": reasoning,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

