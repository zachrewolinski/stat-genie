def extract_final_answer(model_output):
    """
    Extracts the effect of the 'Female' indicator from a fitted binary outcome model result
    (e.g., statsmodels Logit/GLM results or a robust wrapper).

    Returns a dict with keys:
      - "object": dict with numeric outputs (coefficient, se, p-value, CI, odds ratio, sample size)
      - "description": short plain-language interpretation about whether female applicants
                       are more/less likely to be approved and whether the effect is significant.
    """
    import math
    import numpy as np
    import pandas as pd

    if model_output is None:
        return {
            "object": None,
            "description": "No model output (model_output is None). The model did not run or returned None."
        }

    # Helper to safely get attribute and convert to pandas Series for indexing
    def to_series(x):
        if x is None:
            return None
        try:
            return pd.Series(x)
        except Exception:
            try:
                return pd.Series(dict(x))
            except Exception:
                return None

    try:
        params = to_series(getattr(model_output, "params", None))
        if params is None or "Female" not in params.index:
            return {
                "object": None,
                "description": "Model output does not contain a parameter named 'Female'."
            }

        coef = float(params.loc["Female"])

        bse_s = to_series(getattr(model_output, "bse", None))
        se_female = float(bse_s.loc["Female"]) if (bse_s is not None and "Female" in bse_s.index) else None

        p_s = to_series(getattr(model_output, "pvalues", None))
        p_female = float(p_s.loc["Female"]) if (p_s is not None and "Female" in p_s.index) else None

        # Confidence interval: statsmodels .conf_int() often returns a DataFrame/array
        ci_lower = ci_upper = None
        if hasattr(model_output, "conf_int"):
            try:
                ci = model_output.conf_int()
                ci_df = pd.DataFrame(ci)
                # conf_int may return unnamed index; try to locate 'Female'
                if "Female" in ci_df.index:
                    ci_lower = float(ci_df.loc["Female"].iloc[0])
                    ci_upper = float(ci_df.loc["Female"].iloc[1])
                else:
                    # fall back to matching by parameter order
                    try:
                        idx = list(params.index).index("Female")
                        ci_lower = float(ci_df.iloc[idx, 0])
                        ci_upper = float(ci_df.iloc[idx, 1])
                    except Exception:
                        ci_lower = ci_upper = None
            except Exception:
                ci_lower = ci_upper = None

        # Odds ratio and its CI (if coef/CI available)
        try:
            odds_ratio = math.exp(coef)
        except Exception:
            odds_ratio = None
        or_ci_lower = or_ci_upper = None
        if (ci_lower is not None) and (ci_upper is not None):
            try:
                or_ci_lower = math.exp(ci_lower)
                or_ci_upper = math.exp(ci_upper)
            except Exception:
                or_ci_lower = or_ci_upper = None

        # nobs if available
        nobs = getattr(model_output, "nobs", None)
        try:
            nobs = int(nobs) if nobs is not None else None
        except Exception:
            nobs = None

        result_obj = {
            "coefficient_log_odds": coef,
            "std_error": se_female,
            "p_value": p_female,
            "95ci_log_odds": (ci_lower, ci_upper) if (ci_lower is not None and ci_upper is not None) else None,
            "odds_ratio": odds_ratio,
            "95ci_odds_ratio": (or_ci_lower, or_ci_upper) if (or_ci_lower is not None and or_ci_upper is not None) else None,
            "nobs": nobs,
        }

        # Plain-language interpretation
        if p_female is None:
            significance_text = "p-value not available, so statistical significance cannot be assessed."
        else:
            significance_text = ("statistically significant (p < 0.05)" if p_female < 0.05
                                 else f"not statistically significant (p = {p_female:.3f})")

        if coef > 0:
            direction_text = "Female applicants have higher log-odds of mortgage acceptance (i.e., higher approval probability)"
        elif coef < 0:
            direction_text = "Female applicants have lower log-odds of mortgage acceptance (i.e., lower approval probability)"
        else:
            direction_text = "No estimated difference in log-odds of mortgage acceptance for female applicants"

        # Add confidence interval info conditionally
        if result_obj["95ci_odds_ratio"] is not None:
            ci_text = (f"Estimated OR = {result_obj['odds_ratio']:.3f} with 95% CI = "
                       f"({result_obj['95ci_odds_ratio'][0]:.3f}, {result_obj['95ci_odds_ratio'][1]:.3f}).")
        elif result_obj["odds_ratio"] is not None:
            ci_text = f"Estimated OR = {result_obj['odds_ratio']:.3f}. 95% CI not available."
        else:
            ci_text = "Odds ratio could not be computed."

        description = (
            f"{direction_text}; {significance_text}. {ci_text} "
            f"Coefficient (log-odds) = {coef:.4f}."
        )

        return {"object": result_obj, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting statistics: {e}"
        }