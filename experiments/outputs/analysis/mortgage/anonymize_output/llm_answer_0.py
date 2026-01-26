def extract_final_answer(model_output):
    """
    Extract statistics describing the effect of the 'Female' indicator on mortgage approval
    from a fitted statsmodels Logit result or from a dictionary containing the result.

    Returns a dictionary with keys:
      - "object": a dict of extracted numeric results (coefficient, se, p-value, OR, OR 95% CI)
      - "description": a short interpretation of what those numbers mean for the task
    """
    import numpy as np
    import pandas as pd

    # Resolve wrapped inputs: accept either the statsmodels result or a dict with 'model'
    if isinstance(model_output, dict):
        if 'model' in model_output:
            res = model_output['model']
        else:
            # maybe the user passed in the odds_ratios DataFrame only
            if 'odds_ratios' in model_output:
                or_df = model_output['odds_ratios']
                if 'Female' in or_df.index:
                    or_val = float(or_df.loc['Female', 'OR'])
                    ci_l = float(or_df.loc['Female', 'CI_lower'])
                    ci_u = float(or_df.loc['Female', 'CI_upper'])
                    # Can't extract coefficient/p-value from odds table alone
                    return {
                        "object": {
                            "odds_ratio": or_val,
                            "or_ci_lower": ci_l,
                            "or_ci_upper": ci_u
                        },
                        "description": (
                            "Odds ratio for being female (extracted from provided odds_ratio table). "
                            f"Females have OR={or_val:.3f} (95% CI {ci_l:.3f}–{ci_u:.3f}) for mortgage approval "
                            "relative to males, controlling for listed covariates. No p-value/coefficient available "
                            "in the odds-ratio-only input."
                        )
                    }
            raise ValueError("model_output dict did not contain a 'model' or usable 'odds_ratios'.")
    else:
        # assume it's a statsmodels results object
        res = model_output

    # Ensure it's a statsmodels Results object with required attributes
    required_attrs = ['params', 'bse', 'pvalues', 'conf_int']
    if not all(hasattr(res, attr) for attr in required_attrs):
        raise ValueError("Provided model object does not look like a statsmodels fitted result.")

    # Ensure 'Female' term exists
    if 'Female' not in res.params.index:
        raise ValueError("'Female' not found among model parameters.")

    # Extract log-odds coefficient, standard error, p-value and confidence interval (log-odds scale)
    coef = float(res.params['Female'])
    se = float(res.bse['Female'])
    p_value = float(res.pvalues['Female'])
    ci_log = res.conf_int().loc['Female']  # gives [lower, upper] on log-odds scale
    ci_log_lower = float(ci_log[0])
    ci_log_upper = float(ci_log[1])

    # Convert to odds ratio scale
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_log_lower))
    or_ci_upper = float(np.exp(ci_log_upper))

    # Build a succinct conclusion about effect and significance
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    if or_ci_lower > 1:
        direction = "higher odds of approval for females (vs. males)"
    elif or_ci_upper < 1:
        direction = "lower odds of approval for females (vs. males)"
    else:
        direction = "no clear directional effect (CI includes 1)"

    description = (
        f"Female coefficient (log-odds) = {coef:.4f}, SE = {se:.4f}, p = {p_value:.4g}. "
        f"Odds ratio = {odds_ratio:.3f} (95% CI {or_ci_lower:.3f}–{or_ci_upper:.3f}). "
        f"This indicates {direction}; the effect is {significance} at alpha=0.05."
    )

    return {
        "object": {
            "coef_log_odds": coef,
            "std_error": se,
            "p_value": p_value,
            "odds_ratio": odds_ratio,
            "or_ci_lower": or_ci_lower,
            "or_ci_upper": or_ci_upper
        },
        "description": description
    }