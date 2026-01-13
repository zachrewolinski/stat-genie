def extract_final_answer(model_output):
    """
    Extracts relevant statistics for 'NameFem' and 'FemaleName' from a fitted statsmodels GLM (robust) result.
    Returns a dict with keys:
      - "object": dict with extracted numeric results (or None if missing)
      - "description": human-readable explanation of what the numbers mean
    
    The returned numeric results for each variable (if present) include:
      - coef: estimated coefficient (log change in expected count)
      - se: robust standard error
      - pvalue: robust p-value
      - ci95: 95% confidence interval for the coefficient (lower, upper)
      - IRR: incidence rate ratio = exp(coef)
      - IRR_CI95: 95% CI for the IRR = exp(ci95)
    """
    import numpy as np
    import pandas as pd

    # Handle the case where model fitting failed and returned None
    if model_output is None:
        return {
            "object": None,
            "description": "model_output is None (the model was not fitted or the fit failed). No statistics to extract."
        }

    # Try to extract standard results attributes
    try:
        params = getattr(model_output, "params", None)
        bse = getattr(model_output, "bse", None)
        pvalues = getattr(model_output, "pvalues", None)

        # If any of the core pieces are missing, abort with explanation
        if params is None or bse is None or pvalues is None:
            return {
                "object": None,
                "description": "Model object does not expose expected attributes (params, bse, pvalues)."
            }

        # Ensure params/pvalues/bse are pandas Series for label-based indexing
        if not isinstance(params, (pd.Series, pd.DataFrame)):
            try:
                params = pd.Series(params, index=getattr(model_output, "param_names", None))
            except Exception:
                params = pd.Series(params)

        if not isinstance(bse, (pd.Series, pd.DataFrame)):
            try:
                bse = pd.Series(bse, index=params.index)
            except Exception:
                bse = pd.Series(bse)

        if not isinstance(pvalues, (pd.Series, pd.DataFrame)):
            try:
                pvalues = pd.Series(pvalues, index=params.index)
            except Exception:
                pvalues = pd.Series(pvalues)

        # Confidence intervals: prefer model_output.conf_int() if available
        try:
            ci = model_output.conf_int()
            # conf_int may return ndarray or DataFrame; convert to DataFrame with columns [0,1]
            if not isinstance(ci, pd.DataFrame):
                ci = pd.DataFrame(ci, index=params.index, columns=[0, 1])
        except Exception:
            # Fall back to normal approximation: coef +/- 1.96*se
            z = 1.96
            ci = pd.DataFrame({
                0: params - z * bse,
                1: params + z * bse
            }, index=params.index)

        # Helper to extract variable info if present
        def extract_var(name):
            if name in params.index:
                coef = float(params.loc[name])
                se = float(bse.loc[name]) if name in bse.index else None
                p = float(pvalues.loc[name]) if name in pvalues.index else None
                lower = float(ci.loc[name, 0]) if name in ci.index else None
                upper = float(ci.loc[name, 1]) if name in ci.index else None
                irr = float(np.exp(coef)) if coef is not None else None
                irr_ci = [float(np.exp(lower)), float(np.exp(upper))] if (lower is not None and upper is not None) else None
                return {
                    "coef": coef,
                    "se": se,
                    "pvalue": p,
                    "ci95": [lower, upper],
                    "IRR": irr,
                    "IRR_CI95": irr_ci
                }
            else:
                return None

        namefem_res = extract_var("NameFem")
        femname_res = extract_var("FemaleName")

        # Additional model-level info
        # Try to get number of observations
        nobs = None
        try:
            if hasattr(model_output, "nobs"):
                nobs = int(model_output.nobs)
            else:
                # try from model/endog
                nobs = int(getattr(model_output, "model").endog.shape[0])
        except Exception:
            nobs = None

        result_object = {
            "variables": {
                "NameFem": namefem_res,
                "FemaleName": femname_res
            },
            "model_info": {
                "nobs": nobs
            }
        }

        # Build a concise description explaining interpretation
        desc_lines = []
        desc_lines.append(
            "Extracted coefficients are from a Negative Binomial GLM predicting hurricane death counts."
        )
        desc_lines.append(
            "Coefficient = log change in expected death count per unit increase (for continuous NameFem) "
            "or for being female vs male (for FemaleName)."
        )
        desc_lines.append(
            "IRR = exp(coef) is the multiplicative change in expected deaths (e.g., IRR < 1 means fewer expected deaths)."
        )
        desc_lines.append("Returned fields per variable: coef, se (robust), pvalue (robust), ci95, IRR, IRR_CI95.")
        # If variable results missing, note that
        if namefem_res is None:
            desc_lines.append("Note: 'NameFem' not found in model parameters.")
        if femname_res is None:
            desc_lines.append("Note: 'FemaleName' not found in model parameters.")

        description = " ".join(desc_lines)

        return {"object": result_object, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting statistics: {e}"
        }