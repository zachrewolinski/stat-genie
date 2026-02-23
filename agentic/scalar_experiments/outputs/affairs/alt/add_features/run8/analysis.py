import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    required_cols = {"affairs", "children"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create binary indicator for any affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Drop rows with missing key fields, if any
    df = df.dropna(subset=["affairs", "children", "has_affair"])
    return df


def summarize_affairs_by_children(df: pd.DataFrame) -> dict:
    group = df.groupby("children", observed=True)
    summary = group["affairs"].agg(["mean", "median", "std", "count"]).to_dict(orient="index")

    # Proportion with any affair
    prop_any = group["has_affair"].mean().to_dict()
    for k, v in prop_any.items():
        summary.setdefault(k, {})
        summary[k]["prop_any_affair"] = float(v)

    return summary


def chi_square_affair_children(df: pd.DataFrame) -> dict:
    contingency = pd.crosstab(df["children"], df["has_affair"])
    if contingency.shape != (2, 2):
        # Fallback: return empty if structure is unexpected
        return {
            "chi2": None,
            "p_value": None,
            "dof": None,
            "table": contingency.to_dict(),
        }
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "table": contingency.to_dict(),
    }


def logistic_models(df: pd.DataFrame) -> dict:
    results = {}

    # Simple unadjusted model: has_affair ~ C(children)
    try:
        unadj = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
        params = unadj.params.to_dict()
        conf = unadj.conf_int()
        conf.columns = ["lower", "upper"]
        conf_dict = conf.to_dict(orient="index")
        odds_ratios = np.exp(unadj.params).to_dict()
        results["unadjusted"] = {
            "params": {k: float(v) for k, v in params.items()},
            "conf_int": {k: {"lower": float(v["lower"]), "upper": float(v["upper"])} for k, v in conf_dict.items()},
            "pvalues": {k: float(v) for k, v in unadj.pvalues.to_dict().items()},
            "odds_ratios": {k: float(v) for k, v in odds_ratios.items()},
        }
    except Exception as e:  # pragma: no cover - defensive
        results["unadjusted_error"] = str(e)

    # Adjusted model including key covariates when available
    covariates = []
    for col in ["age", "yearsmarried", "religiousness", "education", "rating"]:
        if col in df.columns:
            covariates.append(col)
    if "occupation" in df.columns:
        covariates.append("C(occupation)")

    if covariates:
        formula = "has_affair ~ C(children) + " + " + ".join(covariates)
    else:
        formula = "has_affair ~ C(children)"

    try:
        adj = smf.logit(formula, data=df).fit(disp=False)
        params = adj.params.to_dict()
        conf = adj.conf_int()
        conf.columns = ["lower", "upper"]
        conf_dict = conf.to_dict(orient="index")
        odds_ratios = np.exp(adj.params).to_dict()
        results["adjusted"] = {
            "formula": formula,
            "params": {k: float(v) for k, v in params.items()},
            "conf_int": {k: {"lower": float(v["lower"]), "upper": float(v["upper"])} for k, v in conf_dict.items()},
            "pvalues": {k: float(v) for k, v in adj.pvalues.to_dict().items()},
            "odds_ratios": {k: float(v) for k, v in odds_ratios.items()},
        }
    except Exception as e:  # pragma: no cover - defensive
        results["adjusted_error"] = str(e)

    return results


def infer_response_scale(analysis: dict) -> dict:
    """
    Map statistical evidence to a 0-100 Likert-style response.

    0   -> strong evidence that having children increases affairs
    50  -> no clear evidence either way
    100 -> strong evidence that having children decreases affairs
    """
    # Default neutral
    response = 50
    direction = "neutral"
    strength = "no clear"

    adj = analysis.get("logistic", {}).get("adjusted") or analysis.get("logistic", {}).get("unadjusted")
    chi2 = analysis.get("chi_square", {})

    # Identify the children coefficient (assuming 'children[T.yes]' if baseline is 'no')
    beta = None
    pval = None
    if adj and "params" in adj:
        for name, val in adj["params"].items():
            if "children" in name:
                beta = val
                pval = adj["pvalues"].get(name)
                break

    if beta is not None and pval is not None:
        # Strong evidence thresholds based on p-value and effect size
        if pval < 0.001:
            strength = "very strong"
            base = 30
        elif pval < 0.01:
            strength = "strong"
            base = 25
        elif pval < 0.05:
            strength = "moderate"
            base = 18
        elif pval < 0.10:
            strength = "weak"
            base = 10
        else:
            strength = "no clear"
            base = 0

        # Scale by standardized effect (roughly)
        effect_scale = min(2.0, max(0.0, abs(beta)))  # cap extreme values
        delta = base * (0.5 + 0.5 * effect_scale / 2.0)  # between base*0.5 and base

        if beta < 0:
            # Having children associated with fewer affairs
            response = int(round(50 + delta))
            direction = "decrease"
        else:
            # Having children associated with more affairs
            response = int(round(50 - delta))
            direction = "increase"

    # Use chi-square as secondary support when available
    chi_p = chi2.get("p_value")
    if chi_p is not None and chi_p < 0.05 and strength == "no clear":
        strength = "moderate (chi-square only)"

    response = max(0, min(100, response))

    return {
        "response": response,
        "direction": direction,
        "strength": strength,
        "beta_children": beta,
        "pval_children": pval,
    }


def main() -> None:
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)

    summaries = summarize_affairs_by_children(df)
    chi = chi_square_affair_children(df)
    logit = logistic_models(df)

    analysis = {
        "summaries": summaries,
        "chi_square": chi,
        "logistic": logit,
    }

    scale_info = infer_response_scale(analysis)

    # Save full analysis to a sidecar JSON for inspection/debugging
    with open("analysis_results.json", "w") as f:
        json.dump(
            {
                "analysis": analysis,
                "scale": scale_info,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

