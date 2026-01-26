def extract_final_answer(model_output):
    """
    Extracts the effect of 'female' from a fitted logistic regression model output.
    Returns a dictionary with keys:
      - "object": a dict containing coef, odds ratio, 95% CI for the OR, p-value,
                  significance flag, sample size, and a short numeric summary.
      - "description": a short plain-language interpretation of the result.
    The function is robust to two expected formats of model_output:
      - A dict containing 'odds_ratios' (pandas DataFrame) and 'model'
      - A statsmodels results object directly
    """
    import numpy as np
    import pandas as pd

    # Initialize defaults
    coef = None
    pvalue = None
    or_val = None
    ci_lower = None
    ci_upper = None
    nobs = None

    # Try to extract from the convenient odds_ratios DataFrame if present
    try:
        odds_df = model_output.get('odds_ratios') if isinstance(model_output, dict) else None
        if isinstance(odds_df, pd.DataFrame) and 'female' in odds_df.index:
            row = odds_df.loc['female']
            or_val = float(row['OR'])
            ci_lower = float(row['CI_lower'])
            ci_upper = float(row['CI_upper'])
            pvalue = float(row['pvalue'])
            # If model object also present, try to get coefficient and nobs
            model = model_output.get('model') if isinstance(model_output, dict) else None
            if model is not None:
                try:
                    coef = float(model.params['female'])
                except Exception:
                    coef = None
                try:
                    nobs = int(getattr(model, 'nobs', None) or getattr(model, 'model', None) and len(model.model.endog))
                except Exception:
                    nobs = None
        else:
            raise ValueError("No odds_ratios DataFrame with index 'female' found.")
    except Exception:
        # Fallback: extract directly from statsmodels results object
        try:
            model = model_output.get('model') if isinstance(model_output, dict) else model_output
            params = model.params
            pvalues = model.pvalues
            conf = model.conf_int()  # DataFrame with columns [0,1]
            coef = float(params['female'])
            pvalue = float(pvalues['female'])
            or_val = float(np.exp(coef))
            ci_lower = float(np.exp(conf.loc['female'][0]))
            ci_upper = float(np.exp(conf.loc['female'][1]))
            nobs = int(getattr(model, 'nobs', None) or (hasattr(model, 'model') and len(model.model.endog)))
        except Exception as e:
            # If extraction fails, return an explanatory message
            return {
                "object": None,
                "description": f"Could not extract 'female' coefficient/odds ratio from model_output. Error: {e}"
            }

    # Determine statistical significance at alpha = 0.05 (two-sided)
    significant = (pvalue is not None) and (pvalue < 0.05)

    # Build the returned object
    result_obj = {
        "coef_female": coef,                      # log-odds coefficient (if available)
        "odds_ratio_female": or_val,              # OR = exp(coef)
        "OR_95CI_lower": ci_lower,
        "OR_95CI_upper": ci_upper,
        "p_value": pvalue,
        "significant_at_0.05": bool(significant),
        "n_observations": nobs
    }

    # Plain-language description
    if or_val is not None and pvalue is not None:
        direction = "higher" if or_val > 1 else "lower" if or_val < 1 else "no change"
        desc = (
            f"Controlling for the listed covariates, female applicants have estimated odds of mortgage "
            f"acceptance {or_val:.2f} times that of male applicants (95% CI: {ci_lower:.2f}–{ci_upper:.2f}), "
            f"p = {pvalue:.3f}. This indicates {direction} odds of approval for females compared to males; "
            f"{'the effect is statistically significant at the 0.05 level.' if significant else 'the effect is not statistically significant at the 0.05 level.'}"
        )
    else:
        desc = "Insufficient information to produce an interpretable estimate for 'female'."

    return {
        "object": result_obj,
        "description": desc
    }