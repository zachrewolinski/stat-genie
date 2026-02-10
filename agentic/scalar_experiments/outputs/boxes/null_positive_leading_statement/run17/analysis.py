import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)
    return df


def fit_logistic_models(df: pd.DataFrame):
    model_age = smf.logit("majority_choice ~ age", data=df).fit(disp=False)
    model_culture = smf.logit("majority_choice ~ C(culture)", data=df).fit(disp=False)
    model_full = smf.logit("majority_choice ~ age + C(culture)", data=df).fit(disp=False)
    return model_age, model_culture, model_full


def summarize_evidence(model_age, model_culture, model_full) -> dict:
    summary = {}

    p_age = model_age.pvalues.get("age", np.nan)
    summary["p_age"] = float(p_age)
    summary["coef_age"] = float(model_age.params.get("age", np.nan))

    pvals_culture = model_culture.pvalues
    pvals_culture = pvals_culture[[k for k in pvals_culture.index if k.startswith("C(culture)")]]
    min_p_culture = float(pvals_culture.min()) if len(pvals_culture) > 0 else np.nan
    summary["min_p_culture"] = min_p_culture

    ll_age = model_age.llf
    ll_culture = model_culture.llf
    ll_full = model_full.llf
    summary["ll_age"] = float(ll_age)
    summary["ll_culture"] = float(ll_culture)
    summary["ll_full"] = float(ll_full)

    return summary


def map_evidence_to_scalar(summary: dict) -> int:
    score = 0

    p_age = summary.get("p_age", np.nan)
    coef_age = summary.get("coef_age", 0.0)
    if not np.isnan(p_age) and p_age < 0.05:
        score += 40
        if coef_age > 0:
            score += 10
    elif not np.isnan(p_age) and p_age < 0.1:
        score += 20

    min_p_culture = summary.get("min_p_culture", np.nan)
    if not np.isnan(min_p_culture) and min_p_culture < 0.05:
        score += 40
    elif not np.isnan(min_p_culture) and min_p_culture < 0.1:
        score += 20

    ll_age = summary.get("ll_age", np.nan)
    ll_culture = summary.get("ll_culture", np.nan)
    ll_full = summary.get("ll_full", np.nan)

    if not any(np.isnan(v) for v in [ll_age, ll_culture, ll_full]):
        improvement_full_vs_age = ll_full - ll_age
        improvement_full_vs_cult = ll_full - ll_culture
        if improvement_full_vs_age > 5 and improvement_full_vs_cult > 5:
            score += 10
        elif improvement_full_vs_age > 2 or improvement_full_vs_cult > 2:
            score += 5

    score = max(-100, min(100, score))
    return int(round(score))


def write_conclusion(value: int, path: str = "conclusion.txt") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(int(value)))


def main():
    df = load_data("boxes.csv")
    df = prepare_data(df)
    model_age, model_culture, model_full = fit_logistic_models(df)
    summary = summarize_evidence(model_age, model_culture, model_full)
    scalar = map_evidence_to_scalar(summary)
    write_conclusion(scalar)


if __name__ == "__main__":
    main()

