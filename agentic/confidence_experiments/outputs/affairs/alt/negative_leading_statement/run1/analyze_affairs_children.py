import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for having any extramarital affairs
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Basic group statistics by children status
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("affair_any", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    # Two-sample t-test on affair counts between groups
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]
    t_stat, t_pvalue = stats.ttest_ind(
        affairs_yes, affairs_no, equal_var=False, nan_policy="omit"
    )

    # Logistic regression for probability of any affair, controlling for covariates
    formula = (
        "affair_any ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )

    try:
        logit_model = smf.logit(formula, data=df).fit(disp=False)
        params = logit_model.params
        pvalues = logit_model.pvalues
        # Find the coefficient corresponding to having children (yes vs no)
        child_param_name = None
        for name in params.index:
            if name.startswith("C(children)"):
                child_param_name = name
                break

        child_coef = float(params[child_param_name]) if child_param_name else 0.0
        child_pvalue = float(pvalues[child_param_name]) if child_param_name else 1.0
        child_or = float(np.exp(child_coef))
    except Exception:
        # Fall back to using the t-test only if the regression fails
        logit_model = None
        child_coef = 0.0
        child_pvalue = 1.0
        child_or = 1.0
        child_param_name = None

    # Determine effect direction and statistical support
    if child_coef < 0:
        direction = "decrease"
    elif child_coef > 0:
        direction = "increase"
    else:
        direction = "no_clear_direction"

    # Map evidence to a 0–100 Likert-style scalar where
    # 0 = strong 'No' (children do not decrease affairs)
    # 100 = strong 'Yes' (children do decrease affairs)
    response_value = map_to_scale(direction, child_coef, child_pvalue, t_pvalue)

    explanation = build_explanation(
        group_stats=group_stats,
        t_stat=t_stat,
        t_pvalue=t_pvalue,
        has_logit=(logit_model is not None),
        child_param_name=child_param_name,
        child_coef=child_coef,
        child_pvalue=child_pvalue,
        child_or=child_or,
        response_value=response_value,
    )

    conclusion = {"response": int(round(response_value)), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


def map_to_scale(
    direction: str, coef: float, pvalue_logit: float, pvalue_ttest: float
) -> float:
    """
    Convert statistical evidence into a 0–100 scalar answering:
    'Does having children decrease engagement in extramarital affairs?'

    0   -> strong 'No'
    50  -> indeterminate / no convincing evidence either way
    100 -> strong 'Yes'
    """
    # Use the most informative p-value available
    if pvalue_logit is not None and not np.isnan(pvalue_logit):
        p = pvalue_logit
    else:
        p = pvalue_ttest

    # No convincing evidence of any effect
    if p >= 0.10 or direction == "no_clear_direction":
        # Lean slightly toward 'No' because we fail to detect
        # a systematic decrease in affairs.
        return 30.0

    # Weak / suggestive evidence (0.05–0.10)
    if 0.05 <= p < 0.10:
        if direction == "decrease":
            return 60.0
        elif direction == "increase":
            return 40.0
        else:
            return 50.0

    # Statistically significant evidence (p < 0.05)
    # Use the magnitude of the logistic coefficient to modulate strength.
    abs_coef = abs(coef)
    # Compress very large coefficients; typical log-odds in this context
    # are often between 0 and 1 in magnitude.
    scaled = max(0.0, min(1.0, abs_coef / 1.0))

    if direction == "decrease":
        # Clear evidence that children decrease the likelihood of affairs
        # -> 'Yes' with strength between 70 and 95.
        return 70.0 + 25.0 * scaled
    elif direction == "increase":
        # Clear evidence that children *increase* affairs
        # -> strong 'No' between 5 and 30.
        return 30.0 - 25.0 * scaled
    else:
        # Significant but numerically tiny and direction unclear; close to neutral.
        return 50.0


def build_explanation(
    group_stats: pd.DataFrame,
    t_stat: float,
    t_pvalue: float,
    has_logit: bool,
    child_param_name: str | None,
    child_coef: float,
    child_pvalue: float,
    child_or: float,
    response_value: float,
) -> str:
    # Extract group-level summaries
    stats_dict = {
        row["children"]: {
            "mean_affairs": float(row["mean_affairs"]),
            "prop_any_affair": float(row["prop_any_affair"]),
            "n": int(row["n"]),
        }
        for _, row in group_stats.iterrows()
    }

    lines = []
    lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    lines.append(
        f"Dataset: 601 married individuals from a 1969–1970 survey on extramarital affairs."
    )
    # Group summaries
    if "yes" in stats_dict and "no" in stats_dict:
        yes_stats = stats_dict["yes"]
        no_stats = stats_dict["no"]
        lines.append(
            "Descriptive statistics by children status (affair count coded 0,1,2,3,7,12,12,12):"
        )
        lines.append(
            f"- With children (n={yes_stats['n']}): mean affairs ≈ {yes_stats['mean_affairs']:.2f}, "
            f"share with any affair ≈ {yes_stats['prop_any_affair']:.2%}."
        )
        lines.append(
            f"- Without children (n={no_stats['n']}): mean affairs ≈ {no_stats['mean_affairs']:.2f}, "
            f"share with any affair ≈ {no_stats['prop_any_affair']:.2%}."
        )

    # T-test description
    lines.append(
        "A two-sample t-test comparing the mean number of affairs "
        f"between those with and without children yields t ≈ {t_stat:.2f} "
        f"with p ≈ {t_pvalue:.3f} (Welch correction)."
    )

    # Logistic regression description
    if has_logit and child_param_name is not None:
        direction_text = (
            "lower"
            if child_coef < 0
            else "higher"
            if child_coef > 0
            else "no clear change in"
        )
        lines.append(
            "To account for demographic and marital factors, a logistic regression "
            "models the probability of having any affair as a function of children, "
            "age, years married, religiousness, education, occupation, marital rating, "
            "and gender."
        )
        lines.append(
            f"The coefficient for having children ({child_param_name}) is ≈ {child_coef:.3f}, "
            f"odds ratio ≈ {child_or:.2f}, with p ≈ {child_pvalue:.3f}, "
            f"indicating {direction_text} odds of reporting an affair for parents "
            "relative to non-parents after adjustment."
        )
    elif has_logit:
        lines.append(
            "A logistic regression including children and other covariates was estimated, "
            "but the children effect could not be cleanly extracted; conclusions therefore "
            "rely primarily on group comparisons and the t-test."
        )
    else:
        lines.append(
            "The logistic regression model did not converge robustly, so conclusions are "
            "based on descriptive statistics and the t-test."
        )

    # Interpretation of the Likert-style response
    qualitative = (
        "a strong 'Yes' (children clearly decrease engagement in extramarital affairs)"
        if response_value >= 75
        else "a moderate 'Yes' (some evidence that children decrease affairs)"
        if 55 <= response_value < 75
        else "indeterminate or weak evidence either way"
        if 45 <= response_value < 55
        else "a moderate 'No' (little to no evidence that children decrease affairs)"
        if 25 <= response_value < 45
        else "a strong 'No' (children do not appear to decrease affairs and may be associated with more affairs)"
    )

    lines.append(
        f"Mapping this evidence onto a 0–100 scale where 0 is a strong 'No' "
        f"and 100 is a strong 'Yes', the quantitative response is "
        f"{response_value:.0f}, which corresponds to {qualitative}."
    )

    return " ".join(lines)


if __name__ == "__main__":
    main()

