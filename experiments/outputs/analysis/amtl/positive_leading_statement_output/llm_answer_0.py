def extract_final_answer(model_output):
    """
    Extracts the is_human effect from a fitted statsmodels GLMResultsWrapper (or similar)
    and returns a dictionary with numeric results and a short interpretation.

    Returns:
      {
        "object": {
           "coef": float (log-odds),
           "se": float (standard error used),
           "odds_ratio": float,
           "ci_lower": float,
           "ci_upper": float,
           "p_value": float,
           "significant": bool (p < 0.05 and se available)
        },
        "description": str (text interpretation in plain language)
      }
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    res = model_output

    # Try to get parameters
    try:
        params = res.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not access model parameters from model_output: {e}"
        }

    # Ensure params is a pandas Series with index
    if not hasattr(params, "index"):
        try:
            params = pd.Series(params)
        except Exception:
            return {
                "object": None,
                "description": "Model parameters could not be converted to a pandas Series."
            }

    if "is_human" not in list(params.index):
        return {
            "object": None,
            "description": "The model does not contain a parameter named 'is_human'."
        }

    # Prefer clustered covariance if attached as cov_params_default, otherwise try cov_params()
    cov = None
    if hasattr(res, "cov_params_default") and getattr(res, "cov_params_default") is not None:
        cov = getattr(res, "cov_params_default")
    else:
        try:
            cov = res.cov_params()
        except Exception:
            cov = None

    # Compute standard errors aligned to params index
    se_series = None
    try:
        if cov is None:
            # fallback to baked-in bse if available
            if hasattr(res, "bse") and res.bse is not None:
                se_series = res.bse
            else:
                # cannot obtain SEs
                se_series = pd.Series({name: np.nan for name in params.index})
        else:
            # cov may be a DataFrame or ndarray
            if isinstance(cov, pd.DataFrame):
                # align rows/cols to params.index if possible
                try:
                    cov_sub = cov.loc[params.index, params.index]
                    se_series = pd.Series(np.sqrt(np.diag(cov_sub)), index=params.index)
                except Exception:
                    # fallback to diagonal assumption in order
                    se_series = pd.Series(np.sqrt(np.diag(cov)), index=params.index)
            else:
                # assume numpy array in same order as params
                se_series = pd.Series(np.sqrt(np.diag(cov)), index=params.index)
    except Exception:
        # ultimate fallback
        if hasattr(res, "bse"):
            se_series = res.bse
        else:
            se_series = pd.Series({name: np.nan for name in params.index})

    # Extract values for is_human
    coef = float(params["is_human"])
    # get se robustly
    try:
        se_val = float(se_series["is_human"])
    except Exception:
        # If se_series is ndarray-like or index mismatch, try position-based
        try:
            pos = list(params.index).index("is_human")
            se_val = float(np.sqrt(np.diag(cov))[pos]) if cov is not None else np.nan
        except Exception:
            se_val = np.nan

    # Compute OR, CI, p-value if possible
    or_point = np.exp(coef)
    if np.isfinite(se_val) and se_val > 0:
        z = coef / se_val
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        ci_lower = np.exp(coef - 1.96 * se_val)
        ci_upper = np.exp(coef + 1.96 * se_val)
        significant = (p_val < 0.05)
    else:
        p_val = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        significant = False

    # Build description / interpretation
    if np.isfinite(p_val):
        if significant and or_point > 1:
            interp = (
                "Result: The is_human coefficient is positive and statistically significant "
                f"(coef={coef:.4f}, OR={or_point:.3f}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}], p={p_val:.3g}). "
                "Interpretation: Modern humans (Homo sapiens) have a higher frequency of antemortem tooth loss "
                "compared to the non-human primate genera considered, after adjusting for age, sex, tooth class, "
                "and clustering by specimen."
            )
        elif significant and or_point <= 1:
            interp = (
                "Result: The is_human coefficient is statistically significant but the odds ratio is ≤ 1 "
                f"(coef={coef:.4f}, OR={or_point:.3f}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}], p={p_val:.3g}). "
                "Interpretation: Modern humans do not have higher AMTL; the model indicates lower or equal odds."
            )
        else:
            interp = (
                "Result: The is_human coefficient is not statistically significant "
                f"(coef={coef:.4f}, OR={or_point:.3f}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}], p={p_val:.3g}). "
                "Interpretation: There is no strong evidence that modern humans differ from the non-human primates in AMTL "
                "frequency after adjusting for covariates."
            )
    else:
        interp = (
            f"Result: is_human coef = {coef:.4f}, but standard error / p-value could not be computed reliably (SE={se_val}). "
            "Interpretation: Unable to determine statistical significance for the is_human effect from the provided model object."
        )

    result_object = {
        "coef": coef,
        "se": se_val,
        "odds_ratio": or_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "significant": bool(significant)
    }

    return {
        "object": result_object,
        "description": interp
    }