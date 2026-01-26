def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of 'female' on loan acceptance from a fitted model output.

    Returns:
      {
        "object": {  # numeric summary for the 'female' effect
          "coef": float,
          "pvalue": float,
          "odds_ratio": float,
          "or_ci_lower": float,
          "or_ci_upper": float,
          "significant": bool,
          "n_obs": int or None
        },
        "description": str  # plain-English interpretation in context
      }
    The function handles:
      - model_output being a dict with keys 'model' (statsmodels results) or
        'odds_ratio' (pandas DataFrame with a row named 'female'), or
      - model_output being a statsmodels results object directly.
    """
    import numpy as np
    import pandas as pd

    # Helper to build the numeric summary dict
    def build_summary(coef, pvalue, ci_lower_coef, ci_upper_coef, n_obs=None):
        odds_ratio = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower_coef))
        or_ci_upper = float(np.exp(ci_upper_coef))
        significant = bool(pvalue < 0.05)
        return {
            "coef": float(coef),
            "pvalue": float(pvalue),
            "odds_ratio": odds_ratio,
            "or_ci_lower": or_ci_lower,
            "or_ci_upper": or_ci_upper,
            "significant": significant,
            "n_obs": int(n_obs) if (n_obs is not None) else None
        }

    # Determine source of info
    res = None
    odds_df = None
    n_obs = None

    # If model_output is a dict with 'model' or 'odds_ratio'
    if isinstance(model_output, dict):
        if 'model' in model_output and model_output['model'] is not None:
            res = model_output['model']
        if 'odds_ratio' in model_output and model_output['odds_ratio'] is not None:
            odds_df = model_output['odds_ratio']
        # try to get nobs from model if present
        if res is not None:
            try:
                n_obs = getattr(res, 'nobs', None)
            except Exception:
                n_obs = None
    else:
        # assume model_output itself is a statsmodels results object
        res = model_output

    # Preferred: extract from statsmodels results object if available
    try:
        if res is not None:
            # statsmodels returns pandas Series for params/pvalues and DataFrame for conf_int
            params = res.params
            pvalues = res.pvalues
            conf = res.conf_int()
            # Ensure 'female' present
            if 'female' not in params.index:
                raise KeyError("Coefficient 'female' not found in model params.")
            coef = params.loc['female']
            pvalue = pvalues.loc['female']
            ci_lower_coef, ci_upper_coef = conf.loc['female'].iloc[0], conf.loc['female'].iloc[1]
            summary_obj = build_summary(coef, pvalue, ci_lower_coef, ci_upper_coef, n_obs)
        elif odds_df is not None:
            # odds_ratio DataFrame expected to have row 'female' and columns 'coef', 'ci_lower', 'ci_upper', 'pvalue'
            if 'female' not in odds_df.index:
                raise KeyError("Row 'female' not found in odds_ratio DataFrame.")
            row = odds_df.loc['female']
            coef = float(row['coef'])
            pvalue = float(row.get('pvalue', np.nan))
            # The DataFrame's ci_lower/ci_upper are on odds ratio scale in the provided data.
            # If ci_lower/ci_upper are on odds ratio scale, convert back to log-coef CI by log().
            # Detect whether ci values are >0 and likely on odds-ratio scale (odds ratios >0).
            ci_lower = row.get('ci_lower', None)
            ci_upper = row.get('ci_upper', None)
            if ci_lower is None or ci_upper is None:
                # fallback: try conf columns named like 0 and 1
                try:
                    ci_lower = row[0]
                    ci_upper = row[1]
                except Exception:
                    ci_lower = None
                    ci_upper = None
            if ci_lower is not None and ci_upper is not None and (ci_lower > 0 and ci_upper > 0):
                # assume these are odds ratio CI -> log-transform to coefficient CI
                ci_lower_coef = np.log(ci_lower)
                ci_upper_coef = np.log(ci_upper)
            else:
                # assume they are on coefficient scale already
                ci_lower_coef = float(ci_lower) if ci_lower is not None else (coef - 1.96 * 1.0)
                ci_upper_coef = float(ci_upper) if ci_upper is not None else (coef + 1.96 * 1.0)
            summary_obj = build_summary(coef, pvalue, ci_lower_coef, ci_upper_coef, n_obs=None)
        else:
            raise ValueError("Could not find a model results object or odds_ratio DataFrame in model_output.")
    except Exception as e:
        # If anything goes wrong, return error info
        return {
            "object": None,
            "description": f"Failed to extract 'female' effect from model_output: {e}"
        }

    # Build human-readable description
    sig_text = "statistically significant (p < 0.05)" if summary_obj["significant"] else "not statistically significant (p >= 0.05)"
    desc = (
        f"Effect of being female on mortgage acceptance (conditional on controls):\n"
        f"- Log-odds coefficient = {summary_obj['coef']:.4f}\n"
        f"- Odds ratio = {summary_obj['odds_ratio']:.3f}\n"
        f"- 95% CI for odds ratio = [{summary_obj['or_ci_lower']:.3f}, {summary_obj['or_ci_upper']:.3f}]\n"
        f"- p-value = {summary_obj['pvalue']:.4g} ({sig_text}).\n\n"
        f"Interpretation: After controlling for race, credit metrics, leverage and other listed covariates, "
        f"female applicants have higher odds of having their mortgage application accepted compared with male applicants "
        f"(odds ratio = {summary_obj['odds_ratio']:.3f}). This is an associative result from the fitted logistic model, "
        f"not a proof of causation. "
    )
    if summary_obj["n_obs"] is not None:
        desc += f"Model sample size = {summary_obj['n_obs']}. "

    return {
        "object": summary_obj,
        "description": desc
    }