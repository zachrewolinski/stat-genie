def extract_final_answer(model_output):
    """
    Extract statistics for the genus_Homo_sapiens effect from a fitted statsmodels GLMResultsWrapper
    (binomial logit). Returns a dictionary with 'object' containing the numeric results and
    'description' explaining the values and their interpretation.
    
    Expected keys returned in 'object':
      - coef: log-odds coefficient for genus_Homo_sapiens
      - se: standard error of coef
      - z: z-statistic (coef / se)
      - p_two_sided: two-sided p-value
      - p_one_sided_positive: one-sided p-value for the hypothesis coef > 0
      - ci_95_logodds: 95% CI for the log-odds (lower, upper)
      - odds_ratio: exp(coef)
      - ci_95_or: 95% CI for the odds ratio (exp(lower), exp(upper))
      - pct_change_odds: (odds_ratio - 1) * 100
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Parameter name we care about
    param = 'genus_Homo_sapiens'

    # Basic checks
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params).")

    params = res.params
    if param not in params.index:
        raise ValueError(f"Parameter '{param}' not found in model results. Available params: {list(params.index)}")

    coef = float(params[param])

    # Standard error
    # statsmodels usually exposes bse or standard_errors
    if hasattr(res, 'bse'):
        se = float(res.bse[param])
    elif hasattr(res, 'std_errors'):
        se = float(res.std_errors[param])
    else:
        # Fallback: try to compute from covariance matrix
        if hasattr(res, 'cov_params'):
            cov = res.cov_params()
            se = float(np.sqrt(np.abs(cov.loc[param, param])))
        else:
            raise ValueError("Could not find standard errors in model_output.")

    # z-statistic and two-sided p-value
    z = coef / se if se != 0 else np.nan

    if hasattr(res, 'pvalues'):
        p_two = float(res.pvalues[param])
    else:
        # approximate from z using normal dist
        try:
            from math import erf, sqrt
            # two-sided p from z: 2*(1 - Phi(|z|)); Phi can be computed via erf
            Phi = 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))
            p_two = 2.0 * (1.0 - Phi)
        except Exception:
            p_two = np.nan

    # One-sided p-value for positive effect (coef > 0)
    if np.isnan(z):
        p_one_sided_positive = np.nan
    else:
        # If p_two is available and z>0, p_one = p_two / 2. If z<0 then p_one > 0.5.
        if not np.isnan(p_two):
            if z >= 0:
                p_one_sided_positive = p_two / 2.0
            else:
                p_one_sided_positive = 1.0 - (p_two / 2.0)
        else:
            # compute directly from normal CDF
            try:
                from scipy.stats import norm
                p_one_sided_positive = 1.0 - float(norm.cdf(z))
            except Exception:
                # As a last resort approximate with two-sided fallback
                p_one_sided_positive = np.nan

    # Confidence interval on log-odds
    if hasattr(res, 'conf_int'):
        try:
            ci_df = res.conf_int()
            # conf_int returns DataFrame or ndarray with same param order
            if isinstance(ci_df, (pd.DataFrame, pd.Series)):
                if param in ci_df.index:
                    ci_lower, ci_upper = float(ci_df.loc[param, 0]), float(ci_df.loc[param, 1])
                else:
                    # If conf_int returned as array with same order as params
                    ci_arr = ci_df.values if hasattr(ci_df, 'values') else np.asarray(ci_df)
                    idx = list(params.index).index(param)
                    ci_lower, ci_upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            else:
                # numpy array case
                ci_arr = np.asarray(ci_df)
                idx = list(params.index).index(param)
                ci_lower, ci_upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
        except Exception:
            ci_lower, ci_upper = (np.nan, np.nan)
    else:
        ci_lower, ci_upper = (np.nan, np.nan)

    # Odds ratio and CI
    try:
        odds_ratio = float(np.exp(coef))
        ci_or = (float(np.exp(ci_lower)), float(np.exp(ci_upper))) if not (np.isnan(ci_lower) or np.isnan(ci_upper)) else (np.nan, np.nan)
        pct_change_odds = (odds_ratio - 1.0) * 100.0
    except Exception:
        odds_ratio = np.nan
        ci_or = (np.nan, np.nan)
        pct_change_odds = np.nan

    result_object = {
        'param': param,
        'coef_log_odds': coef,
        'se': se,
        'z': z,
        'p_two_sided': p_two,
        'p_one_sided_positive': p_one_sided_positive,
        'ci_95_logodds': (ci_lower, ci_upper),
        'odds_ratio': odds_ratio,
        'ci_95_or': ci_or,
        'pct_change_odds': pct_change_odds
    }

    description_lines = [
        f"Extracted results for parameter '{param}':",
        "- coef_log_odds: estimated change in log-odds of AMTL for Homo sapiens compared to Pan (reference),",
        "- se: standard error of the coefficient,",
        "- z: z-statistic (coef / se) used for testing H0: coef = 0,",
        "- p_two_sided: two-sided p-value for H0: coef = 0; small values (e.g., < 0.05) indicate a statistically significant difference,",
        "- p_one_sided_positive: one-sided p-value for H1: coef > 0 (i.e., Homo sapiens have higher AMTL),",
        "- ci_95_logodds: 95% confidence interval on the coefficient (log-odds scale),",
        "- odds_ratio and ci_95_or: exponentiated coefficient and CI (multiplicative change in odds); pct_change_odds gives percent change in odds."
    ]
    description = " ".join(description_lines)

    return {
        "object": result_object,
        "description": description
    }