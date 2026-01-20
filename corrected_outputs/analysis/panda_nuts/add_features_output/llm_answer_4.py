def extract_final_answer(model_output):
    """
    Extract key statistics for the predictors of interest from a fitted statsmodels
    MixedLMResults (or wrapper) object.

    Returns a dict with:
      - "object": dictionary containing coefficient, se, z, p-value, 95% CI, and exp(coef)
                  for each predictor (age, sex_male, HelpReceived), plus random and residual variances and AIC/BIC.
      - "description": short explanation of what the numbers mean for nut-cracking efficiency.

    Notes on interpretation included in the description:
      - The model predicts LogEfficiency = log1p(nuts_opened / seconds).
      - Coefficients are changes on the log1p scale; exp(coef) approximates the multiplicative
        change in (nuts/sec + 1) for a one-unit increase in the predictor (or for the binary
        predictor switching from 0 to 1).
      - Positive coef => higher efficiency; negative => lower efficiency.
    """
    from math import exp
    from scipy import stats

    # Try to locate parameter arrays in the object robustly
    # statsmodels MixedLMResultsWrapper provides .params, .bse, .pvalues, .conf_int()
    try:
        params = model_output.params
    except Exception:
        # try alternative attribute
        params = getattr(model_output, 'fe_params', None)
    try:
        bse = model_output.bse
    except Exception:
        bse = getattr(model_output, 'bse', None)
    try:
        pvalues = model_output.pvalues
    except Exception:
        pvalues = getattr(model_output, 'pvalues', None)

    # conf_int may return DataFrame or ndarray; handle gracefully
    try:
        conf = model_output.conf_int()
    except Exception:
        conf = None

    predictors = ['age', 'sex_male', 'HelpReceived']
    effects = {}

    # Helper to check membership in params (works for pandas Series or dict-like)
    def has_param(name):
        try:
            return name in params.index
        except Exception:
            try:
                return name in params
            except Exception:
                return False

    for var in predictors:
        if params is None or not has_param(var):
            effects[var] = None
            continue

        coef = float(params[var])
        se = None
        if bse is not None:
            try:
                se = float(bse[var])
            except Exception:
                se = None

        z = None
        if se is not None and se != 0:
            z = coef / se

        p = None
        if pvalues is not None:
            try:
                p = float(pvalues[var])
            except Exception:
                p = None
        elif z is not None:
            # normal approximation
            p = 2.0 * stats.norm.sf(abs(z))

        # confidence interval
        ci_low = ci_high = None
        if conf is not None:
            try:
                # conf may be DataFrame with columns [0,1] or named columns
                if hasattr(conf, 'loc'):
                    row = conf.loc[var]
                    # support both numeric column labels or strings
                    ci_low = float(row.iloc[0])
                    ci_high = float(row.iloc[1])
                else:
                    # array-like: try to find row by matching order - fallback
                    ci_low = float(conf[var, 0])
                    ci_high = float(conf[var, 1])
            except Exception:
                ci_low = ci_high = None
        if (ci_low is None or ci_high is None) and se is not None:
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se

        effects[var] = {
            'coef': coef,
            'se': se,
            'z_or_t': z,
            'pvalue': p,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'exp_coef': exp(coef)  # multiplicative change in (nuts/sec + 1)
        }

    # Random intercept variance (first element of cov_re) and residual variance (scale)
    re_var = None
    try:
        cov_re = model_output.cov_re
        # cov_re might be DataFrame or ndarray
        if hasattr(cov_re, 'iloc'):
            re_var = float(cov_re.iloc[0, 0])
        else:
            re_var = float(cov_re[0][0])
    except Exception:
        # some MixedLMResults expose random effects var differently
        try:
            re_var = float(model_output.cov_re.iloc[0, 0])
        except Exception:
            re_var = None

    resid_var = None
    try:
        resid_var = float(model_output.scale)
    except Exception:
        resid_var = None

    aic = getattr(model_output, 'aic', None)
    bic = getattr(model_output, 'bic', None)

    output_object = {
        'effects': effects,
        'random_intercept_variance': re_var,
        'residual_variance': resid_var,
        'aic': aic,
        'bic': bic
    }

    description = (
        "Extracted fixed-effect estimates for predictors of interest from the fitted mixed-effects model.\n"
        "- Outcome: LogEfficiency = log1p(nuts_opened / seconds).\n"
        "- For each predictor (age, sex_male where male=1 vs female=0, HelpReceived where 1=yes):\n"
        "    coef: estimated change in the log1p-efficiency per unit change (or for 1 vs 0 for binaries).\n"
        "    se: standard error; z_or_t: test statistic (coef / se) when available.\n"
        "    pvalue: two-sided p-value for the null that coef == 0 (computed from model if available, otherwise from normal approx).\n"
        "    ci_lower / ci_upper: 95% confidence interval for the coef (uses model conf_int if available, otherwise +/-1.96*se).\n"
        "    exp_coef: exp(coef), which approximates the multiplicative change in (nuts/sec + 1) associated with the predictor.\n"
        "- random_intercept_variance: estimated variance of the chimpanzee random intercept (captures between-individual heterogeneity).\n"
        "- residual_variance: estimated residual (within-session) variance.\n\n"
        "Interpretation guidance:\n"
        "- A positive coef means higher nut-cracking efficiency (on the log1p scale) as the predictor increases; negative means lower efficiency.\n"
        "- For binary predictors (sex_male, HelpReceived), exp(coef) > 1 indicates that the group with value=1 has higher (nuts/sec + 1) on average.\n"
        "- Use the p-value (e.g., p < 0.05) and whether the 95% CI excludes zero to judge statistical significance.\n"
        "- Report the returned 'effects' entries for exact numeric estimates and use exp_coef to communicate multiplicative effects on the original (nuts/sec + 1) scale."
    )

    return {"object": output_object, "description": description}