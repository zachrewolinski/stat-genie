def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted count model (negative binomial / fallback Poisson)
    to evaluate the effect of hurricane name femininity on fatalities.

    Returns:
      {
        "object": {
          "nobs": int,
          "variables": {
            "masfem_z": {
              "coef": float,          # raw coefficient (log scale)
              "se": float,
              "pvalue": float,
              "ci_lower": float,      # 95% CI on coef (log scale)
              "ci_upper": float,
              "irr": float,           # incidence rate ratio = exp(coef)
              "irr_ci_lower": float,
              "irr_ci_upper": float
            },
            "female_name": { ... }    # same structure if present in model
          },
          "model_family": str,
          "model_notes": str         # any fallback notes if applicable
        },
        "description": str           # plain-language interpretation in context
      }
    """
    import math
    import numpy as np
    import pandas as pd

    results = model_output if isinstance(model_output, dict) else {}
    res = results.get('nb_model') or results.get('model')  # try common keys

    if res is None:
        return {
            "object": None,
            "description": "No fitted model found in model_output (expected key 'nb_model')."
        }

    # Helper to safely get conf_int and handle different return types
    def _get_conf_int(res):
        try:
            ci = res.conf_int()  # often a DataFrame with index = param names
            # If it's returned as ndarray, convert to DataFrame with param names
            if isinstance(ci, (list, tuple, np.ndarray)):
                ci = pd.DataFrame(ci, index=res.params.index, columns=[0, 1])
            else:
                # ensure columns are numeric 0,1 for consistent indexing
                ci = pd.DataFrame(ci)
                ci.columns = [0, 1]
                ci.index = res.params.index
            return ci
        except Exception:
            # Last resort: approximate CI from coef +/- 1.96*se
            se = getattr(res, 'bse', None)
            if se is None:
                raise
            lower = res.params - 1.96 * se
            upper = res.params + 1.96 * se
            ci = pd.DataFrame(np.column_stack([lower, upper]), index=res.params.index, columns=[0, 1])
            return ci

    params = res.params
    pvalues = getattr(res, 'pvalues', pd.Series([np.nan]*len(params), index=params.index))
    bse = getattr(res, 'bse', pd.Series([np.nan]*len(params), index=params.index))
    ci = _get_conf_int(res)

    vars_of_interest = ['masfem_z', 'female_name']
    extracted = {}

    for v in vars_of_interest:
        if v in params.index:
            coef = float(params[v])
            se = float(bse[v]) if v in bse.index else float(np.nan)
            p = float(pvalues[v]) if v in pvalues.index else float(np.nan)
            try:
                ci_low = float(ci.loc[v].iloc[0])
                ci_high = float(ci.loc[v].iloc[1])
            except Exception:
                # fallback using +/-1.96*se
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se

            irr = float(math.exp(coef))
            irr_low = float(math.exp(ci_low))
            irr_high = float(math.exp(ci_high))

            extracted[v] = {
                "coef": coef,
                "se": se,
                "pvalue": p,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "irr": irr,
                "irr_ci_lower": irr_low,
                "irr_ci_upper": irr_high
            }

    # meta info
    try:
        nobs = int(getattr(res, 'nobs', int(len(res.model.endog))))
    except Exception:
        try:
            nobs = int(len(res.model.endog))
        except Exception:
            nobs = None

    family = None
    try:
        family = str(res.model.family.__class__.__name__)
    except Exception:
        family = "unknown"

    model_notes = ""
    # detect if Poisson fallback (HC3) might have been used by checking family name
    if 'Poisson' in family and 'NegativeBinomial' not in family:
        model_notes = "Model appears to be Poisson (possibly used as fallback with robust covariances)."
    else:
        model_notes = f"Model family: {family} (coefficients are on the model link scale — typically log)."

    # Compose plain-language interpretation focusing on masfem_z (primary hypothesis)
    if 'masfem_z' in extracted:
        e = extracted['masfem_z']
        # significance statement
        sig = "statistically significant" if (not np.isnan(e['pvalue']) and e['pvalue'] < 0.05) else "not statistically significant"
        direction = "higher" if e['coef'] > 0 else "lower" if e['coef'] < 0 else "no change"
        interp = (
            f"masfem_z (standardized femininity score): coef = {e['coef']:.3f} (SE={e['se']:.3f}), "
            f"IRR = {e['irr']:.3f} (95% CI {e['irr_ci_lower']:.3f}–{e['irr_ci_upper']:.3f}), "
            f"p = {e['pvalue']:.3g}. This implies that a one SD increase in name femininity is associated with "
            f"{direction} expected fatalities by a factor of {e['irr']:.3f} on average. The effect is {sig}."
        )
    else:
        interp = "The predictor 'masfem_z' is not present in the fitted model."

    # Also optionally mention binary female_name if present
    if 'female_name' in extracted:
        f = extracted['female_name']
        sigf = "statistically significant" if (not np.isnan(f['pvalue']) and f['pvalue'] < 0.05) else "not statistically significant"
        interp += (
            " For the binary female_name predictor: "
            f"coef = {f['coef']:.3f}, IRR = {f['irr']:.3f} (95% CI {f['irr_ci_lower']:.3f}–{f['irr_ci_upper']:.3f}), "
            f"p = {f['pvalue']:.3g}; effect is {sigf}."
        )

    description = (
        f"Extracted negative binomial/Poisson model statistics for the femininity predictors. "
        f"n = {nobs}. {model_notes} {interp}"
    )

    output_obj = {
        "nobs": nobs,
        "variables": extracted,
        "model_family": family,
        "model_notes": model_notes
    }

    return {"object": output_obj, "description": description}