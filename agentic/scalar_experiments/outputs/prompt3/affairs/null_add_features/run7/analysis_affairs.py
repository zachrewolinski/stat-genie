import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def summarize_affairs_by_children(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    grouped = df.groupby("children", observed=True)

    summary = grouped["affairs"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "mean_affairs", "std": "std_affairs", "count": "n"},
    )
    prop_any = grouped["affair_any"].mean().rename("prop_any")

    result = summary.join(prop_any)
    return result.to_dict(orient="index")


def fit_logistic_model(df: pd.DataFrame):
    df = df.copy()
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Use core variables from the classic affairs dataset
    formula = (
        "affair_any ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )

    model = smf.logit(formula=formula, data=df).fit(disp=False)
    return model


def interpret_results(descriptive: dict, model) -> tuple[str, int, int, str]:
    # Descriptive comparison
    children_levels = sorted(descriptive.keys())

    desc_text_parts = []
    for level in children_levels:
        stats = descriptive[level]
        desc_text_parts.append(
            f"For children = '{level}', mean affairs = {stats['mean_affairs']:.3f}, "
            f"proportion with any affair = {stats['prop_any']:.3f} (n = {stats['n']})."
        )

    # Logistic regression coefficient for children
    params = model.params
    conf_int = model.conf_int()

    # Children is treated as a factor; assume 'no' is baseline, so we look for 'children[T.yes]'
    children_param_name = None
    for name in params.index:
        if "children" in name and "[T." in name or "children[T." in name or "children[T.yes]" == name:
            # Pick the most specific standard name if present
            if name == "C(children)[T.yes]" or name == "children[T.yes]":
                children_param_name = name
                break
            if children_param_name is None:
                children_param_name = name

    response = "No"
    strength = 50
    confidence = 60

    if children_param_name is None:
        explanation = (
            "Could not identify the children coefficient in the logistic regression model. "
            "Only descriptive statistics were used."
        )
    else:
        coef = params[children_param_name]
        ci_low, ci_high = conf_int.loc[children_param_name]
        p_value = model.pvalues[children_param_name]

        # Odds ratio
        odds_ratio = float(np.exp(coef))

        # Determine direction: negative coefficient/OR < 1 suggests children decrease affairs
        if ci_high < 0:
            # Credible strong negative effect
            response = "Yes"
            strength = 85
            confidence = 85
        elif ci_low > 0:
            # Strong positive effect
            response = "No"
            strength = 85
            confidence = 85
        else:
            # Confidence interval crosses zero; effect uncertain
            if coef < 0:
                response = "Yes"
                strength = 60
            else:
                response = "No"
                strength = 60

            # Confidence reflects statistical uncertainty
            if p_value < 0.1:
                confidence = 70
            elif p_value < 0.2:
                confidence = 60
            else:
                confidence = 50

        explanation = (
            "Research question: Does having children decrease engagement in extramarital affairs?\n"
            + "Descriptive statistics by children status:\n"
            + "\n".join(desc_text_parts)
            + "\n\n"
            + f"Logistic regression of any affair on children and controls shows coefficient "
            f"for children ({children_param_name}) = {coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], "
            f"p-value = {p_value:.3f}, odds ratio ≈ {odds_ratio:.3f}.\n"
            + "A negative coefficient (odds ratio < 1) suggests that, holding gender, age, years married, "
            "religiousness, education, occupation, and marital satisfaction constant, having children is "
            "associated with a lower likelihood of reporting any extramarital affair; a positive coefficient "
            "suggests the opposite. The final Yes/No answer reflects the sign and uncertainty of this estimate "
            "combined with the descriptive differences in average affair rates between couples with and without "
            "children."
        )

    return response, strength, confidence, explanation


def write_conclusion(path: Path, response: str, strength: int, confidence: int, explanation: str) -> None:
    obj = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False))


def main() -> None:
    data_path = Path("affairs.csv")
    df = load_data(data_path)

    descriptive = summarize_affairs_by_children(df)
    model = fit_logistic_model(df)

    response, strength, confidence, explanation = interpret_results(descriptive, model)

    conclusion_path = Path("conclusion.txt")
    write_conclusion(conclusion_path, response, strength, confidence, explanation)


if __name__ == "__main__":
    main()

