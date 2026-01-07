def extract_final_answer(model_output):
    """
    Extracts the estimated effect of being female on mortgage approval from a fitted
    statsmodels Logit (BinaryResultsWrapper) that includes 'Female' and optionally
    an interaction 'Female_Black'.

    Returns a dictionary with keys:
      - "object": dict with numeric results for:
           * 'female_non_black' -> effect (coef, se, z, p, 95% CI) for non-Black applicants
           * 'female_black'     -> effect (coef, se, z, p, 95% CI) for Black applicants
             (None if interaction term not present)
           Each effect also includes 'odds_ratio' and 'or_ci' (95% CI for odds ratio).
      - "description": short interpretation of what the numbers mean.
    """
    import math
    import numpy as np

    # Helpers for normal cdf and two-sided p-value without requiring scipy
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _two_sided_p_from_z(z):
        return 2.0 * (1.0 - _norm_cdf(abs(z)))

    # Ensure we have params and covariance
    params = getattr(model_output, "params", None)
    cov = getattr(model_output, "cov_params", None)
    if params is None or cov is None:
        raise ValueError("model_output does not appear to be a statsmodels results object with .params and .cov_params().")

    # If cov is a callable method, call it to get the matrix
    if callable(cov):
        cov = cov()

    # Convert params and cov to pandas-like access if they are (safe indexing)
    try:
        # pandas Series / DataFrame style
        params_get = lambda name: float(params[name])
        cov_get = lambda i, j: float(cov.loc[i, j])
    except Exception:
        # numpy-array fallback with integer indexing (less likely)
        def params_get(name):
            # try to find index
            idx = list(model_output.params.index).index(name)
            return float(model_output.params.iloc[idx])
        def cov_get(i, j):
            idx_i = list(model_output.params.index).index(i)
            idx_j = list(model_output.params.index).index(j)
            return float(cov.iloc[idx_i, idx_j])

    # Required main effect
    if 'Female' not in list(model_output.params.index):
        raise KeyError("Model does not contain a parameter named 'Female'.")

    # Extract main female effect
    coef_f = params_get('Female')
    var_f = cov_get('Female', 'Female')
    se_f = math.sqrt(max(var_f, 0.0))
    z_f = coef_f / se_f if se_f > 0 else float('nan')
    p_f = _two_sided_p_from_z(z_f) if se_f > 0 else float('nan')
    ci_low_f = coef_f - 1.96 * se_f
    ci_high_f = coef_f + 1.96 * se_f
    or_f = math.exp(coef_f)
    or_ci = (math.exp(ci_low_f), math.exp(ci_high_f))

    female_non_black = {
        'coef': float(coef_f),
        'se': float(se_f),
        'z': float(z_f),
        'p_value': float(p_f),
        'ci_95': (float(ci_low_f), float(ci_high_f)),
        'odds_ratio': float(or_f),
        'or_95_ci': (float(or_ci[0]), float(or_ci[1]))
    }

    # Check for interaction Female_Black
    female_black = None
    if 'Female_Black' in list(model_output.params.index):
        coef_fb = params_get('Female_Black')
        # combined effect = Female + Female_Black
        coef_comb = coef_f + coef_fb
        # variance of sum: Var(F) + Var(FB) + 2*Cov(F,FB)
        try:
            cov_ffb = cov_get('Female', 'Female_Black')
        except Exception:
            cov_ffb = cov_get('Female_Black', 'Female')  # try reverse
        var_fb = cov_get('Female_Black', 'Female_Black')
        var_comb = var_f + var_fb + 2.0 * cov_ffb
        se_comb = math.sqrt(max(var_comb, 0.0))
        z_comb = coef_comb / se_comb if se_comb > 0 else float('nan')
        p_comb = _two_sided_p_from_z(z_comb) if se_comb > 0 else float('nan')
        ci_low_comb = coef_comb - 1.96 * se_comb
        ci_high_comb = coef_comb + 1.96 * se_comb
        or_comb = math.exp(coef_comb)
        or_ci_comb = (math.exp(ci_low_comb), math.exp(ci_high_comb))

        female_black = {
            'coef': float(coef_comb),
            'se': float(se_comb),
            'z': float(z_comb),
            'p_value': float(p_comb),
            'ci_95': (float(ci_low_comb), float(ci_high_comb)),
            'odds_ratio': float(or_comb),
            'or_95_ci': (float(or_ci_comb[0]), float(or_ci_comb[1]))
        }

    result_object = {
        'female_non_black': female_non_black,
        'female_black': female_black
    }

    description = (
        "Extracted estimates describe how being female affects the log-odds of mortgage approval.\n"
        "- 'female_non_black' is the main effect (Female) representing the effect for non-Black applicants (Black=0).\n"
        "- 'female_black' (if present) is the combined effect Female + Female_Black representing the effect for Black applicants.\n"
        "For each effect you get: coefficient (log-odds), standard error, z-statistic, two-sided p-value, 95% CI on the coefficient, "
        "and the corresponding odds ratio with 95% CI. Odds ratio > 1 means higher odds of approval for females; < 1 means lower odds."
    )

    return {"object": result_object, "description": description}