import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    return df


def compute_group_rates(df: pd.DataFrame) -> dict:
    group_rates = df.groupby("children")["has_affair"].mean()
    return {str(k): float(v) for k, v in group_rates.items()}


def logistic_analysis(df: pd.DataFrame):
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + rating + C(gender)"
    )
    try:
        model = smf.logit(formula=formula, data=df).fit(disp=False)
    except Exception:
        return None

    term = "C(children)[T.yes]"
    if term not in model.params.index:
        return None

    coef = float(model.params[term])
    p_value = float(model.pvalues[term])
    conf_int = model.conf_int().loc[term]
    ci_low = float(conf_int[0])
    ci_high = float(conf_int[1])
    odds_ratio = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    return {
        "coef": coef,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
    }


def contingency_analysis(df: pd.DataFrame):
    table = pd.crosstab(df["children"], df["has_affair"])
    if table.shape != (2, 2):
        return None

    chi2, p_value, _, _ = stats.chi2_contingency(table)
    # children categories expected: "no", "yes"
    rates = df.groupby("children")["has_affair"].mean()
    rate_no = float(rates.get("no", np.nan))
    rate_yes = float(rates.get("yes", np.nan))
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "rate_no": rate_no,
        "rate_yes": rate_yes,
    }


def build_conclusion(df: pd.DataFrame) -> dict:
    group_rates = compute_group_rates(df)
    rate_no = group_rates.get("no")
    rate_yes = group_rates.get("yes")

    logit_res = logistic_analysis(df)
    cont_res = contingency_analysis(df)

    decision_method = None
    response = "No"

    if logit_res is not None:
        decision_method = "logistic_regression"
        coef = logit_res["coef"]
        p_value = logit_res["p_value"]
        odds_ratio = logit_res["odds_ratio"]
        or_ci_low = logit_res["or_ci_low"]
        or_ci_high = logit_res["or_ci_high"]

        if p_value < 0.05 and odds_ratio < 1.0:
            response = "Yes"
    elif cont_res is not None:
        decision_method = "chi_square"
        p_value = cont_res["p_value"]
        rate_no = cont_res["rate_no"]
        rate_yes = cont_res["rate_yes"]
        if p_value < 0.05 and rate_yes < rate_no:
            response = "Yes"

    n_total = int(df.shape[0])
    n_children_yes = int((df["children"] == "yes").sum())
    n_children_no = int((df["children"] == "no").sum())

    parts = []
    parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs "
        "in this sample of married individuals?"
    )
    parts.append(
        f"I defined an affair as reporting any nonzero value in the 'affairs' count variable "
        f"and created a binary outcome 'has_affair'."
    )
    parts.append(
        f"The dataset contains {n_total} observations: {n_children_yes} with children "
        f"and {n_children_no} without children."
    )
    if rate_no is not None and rate_yes is not None:
        parts.append(
            f"Empirically, about {rate_no*100:.1f}% of respondents without children and "
            f"{rate_yes*100:.1f}% of respondents with children reported at least one affair "
            f"in the past year."
        )

    if logit_res is not None:
        parts.append(
            "I fit a logistic regression of having any affair on an indicator for having "
            "children, controlling for age, years married, gender, religiousness, education, "
            "and self-rated marriage quality."
        )
        parts.append(
            f"In this model, the odds ratio for having children versus not having children "
            f"was {logit_res['odds_ratio']:.2f} (95% CI "
            f"[{logit_res['or_ci_low']:.2f}, {logit_res['or_ci_high']:.2f}], "
            f"p = {logit_res['p_value']:.3f})."
        )
        if response == "Yes":
            parts.append(
                "Because the odds ratio is below 1 and statistically significant at the 5% level, "
                "the data provide evidence that having children is associated with a lower "
                "likelihood of engaging in extramarital affairs, after controlling for these "
                "covariates."
            )
        else:
            parts.append(
                "Because the effect of having children is not a statistically significant "
                "reduction in the odds of an affair at the 5% level, the data do not provide "
                "clear evidence that having children decreases engagement in extramarital affairs."
            )
    elif cont_res is not None:
        parts.append(
            "I also analysed the 2x2 contingency table of children (yes/no) by whether an affair "
            "occurred using a chi-square test."
        )
        parts.append(
            f"The chi-square test yielded p = {cont_res['p_value']:.3f}."
        )
        if response == "Yes":
            parts.append(
                "Children were associated with a significantly lower proportion of respondents "
                "reporting an affair (at the 5% level), suggesting that having children is "
                "linked to reduced engagement in extramarital affairs."
            )
        else:
            parts.append(
                "Although the observed proportions may differ somewhat, the chi-square test does "
                "not show a statistically significant reduction in affairs among those with "
                "children at the 5% level, so the data do not support the claim that having "
                "children decreases engagement in extramarital affairs."
            )
    else:
        parts.append(
            "I attempted both logistic regression and chi-square analysis, but model fitting "
            "failed (e.g., due to numerical issues). In this case I rely only on the observed "
            "group proportions, which do not provide strong enough evidence to conclude that "
            "having children reduces the likelihood of extramarital affairs."
        )

    explanation = " ".join(parts)

    return {
        "response": response,
        "explanation": explanation,
        "decision_method": decision_method,
    }


def main():
    data_path = Path("affairs.csv")
    df = load_data(data_path)

    result = build_conclusion(df)
    output = {
        "response": result["response"],
        "explanation": result["explanation"],
    }

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

