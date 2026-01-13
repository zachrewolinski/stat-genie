def extract_final_answer(model_output):
    """
    Extracts the coefficient, SE, test statistic, p-value, 95% CI, and odds ratio
    for the primary predictor 'is_human' from a fitted statsmodels GLM results
    object (possibly with cluster-robust covariance applied).

    Returns a dictionary with keys:
      - "object": dict with numeric results for 'is_human'
      - "description": human-readable interpretation of the result in context

    Notes:
      - Assumes a binomial GLM with default logit link was used, so coefficients
        are log-odds. Odds ratios are exp(coef).
      - If p-value is not present on the object, it will be computed from the
        coefficient and standard error using a normal approximation.
    """
    import math

    res = model_output

    # Helper to safely extract a value for a given parameter name from various possible containers
    def _get_from_container(container, name):
        # container may be a pandas Series/DataFrame row, numpy array, or dict-like
        try:
            return container[name]
        except Exception:
            try:
                # If container is an ndarray and model has index ordering, try to get positional value.
                # We will try to find the position of parameter in res.params.index if available.
                if hasattr(res, "params") and hasattr(res.params, "index"):
                    idx = list(res.params.index).index(name)
                    return container[idx]
            except Exception:
                pass
        raise KeyError(f"Parameter '{name}' not found in the provided container.")

    param_name = "is_human"

    # Ensure params exist
    if not hasattr(res, "params"):
        raise ValueError("model_output does not have 'params' attribute; not a recognized statsmodels results object.")

    # Helper to format numeric values safely
    def _fmt(val, fmt="{:.4g}", default="NA"):
        try:
            if val is None:
                return default
            # handle numpy types too
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                return fmt.format(val)
        except Exception:
            pass
        return default

    # Extract coefficient
    try:
        coef_raw = _get_from_container(res.params, param_name)
        coef = None if coef_raw is None else float(coef_raw)
    except KeyError:
        raise KeyError(f"Parameter '{param_name}' not found in model parameters. Available parameters: {list(res.params.index)}")
    except Exception:
        coef = None

    # Standard error
    bse = None
    if hasattr(res, "bse"):
        try:
            bse_raw = _get_from_container(res.bse, param_name)
            bse = None if bse_raw is None else float(bse_raw)
        except Exception:
            bse = None

    # p-value
    pval = None
    if hasattr(res, "pvalues"):
        try:
            pval_raw = _get_from_container(res.pvalues, param_name)
            pval = None if pval_raw is None else float(pval_raw)
        except Exception:
            pval = None

    # test statistic (z or t)
    stat = None
    # Some result wrappers have 'tvalues' or 'zvalues' or 'tvalue' etc.
    for attr in ("tvalues", "zvalues", "tvalue", "zvalue"):
        if hasattr(res, attr):
            try:
                stat_raw = _get_from_container(getattr(res, attr), param_name)
                stat = None if stat_raw is None else float(stat_raw)
                break
            except Exception:
                stat = None
    # If no test stat but we have coef and bse, compute stat
    if stat is None and (bse is not None and bse != 0) and (coef is not None):
        try:
            stat = coef / bse
        except Exception:
            stat = None

    # If p-value still missing, compute from normal approx using stat
    if pval is None and stat is not None:
        # two-sided p-value from normal distribution: p = erfc(|z|/sqrt(2))
        try:
            pval = float(math.erfc(abs(stat) / math.sqrt(2)))
        except Exception:
            pval = None

    # Confidence interval
    ci_lower = ci_upper = None
    if hasattr(res, "conf_int"):
        try:
            ci = res.conf_int()
            # conf_int() may return DataFrame or ndarray
            try:
                # If DataFrame-like
                if hasattr(ci, "loc"):
                    row = ci.loc[param_name]
                    # row may be a Series or array-like
                    try:
                        ci_lower = float(row.iloc[0])
                        ci_upper = float(row.iloc[1])
                    except Exception:
                        # fallback if row itself is a tuple/list
                        ci_lower = float(row[0])
                        ci_upper = float(row[1])
                else:
                    # ndarray: need position of param
                    if hasattr(res.params, "index"):
                        idx = list(res.params.index).index(param_name)
                        ci_lower = float(ci[idx, 0])
                        ci_upper = float(ci[idx, 1])
                    else:
                        # fallback: first row
                        ci_lower = float(ci[0, 0])
                        ci_upper = float(ci[0, 1])
            except Exception:
                # Another fallback: try indexing by name if possible
                try:
                    val = ci[param_name]
                    ci_lower = float(val[0])
                    ci_upper = float(val[1])
                except Exception:
                    pass
        except Exception:
            pass

    # Odds ratio and CI for odds ratio if log-odds model (GLM binomial logit)
    try:
        odds_ratio = None if coef is None else math.exp(coef)
    except Exception:
        odds_ratio = None
    or_ci_lower = or_ci_upper = None
    if ci_lower is not None and ci_upper is not None:
        try:
            or_ci_lower = math.exp(ci_lower)
            or_ci_upper = math.exp(ci_upper)
        except Exception:
            pass

    # Significance flag at alpha = 0.05 (two-sided)
    signif = None
    if pval is not None:
        try:
            signif = (pval < 0.05)
        except Exception:
            signif = None

    result_object = {
        "term": param_name,
        "coef_log_odds": coef,
        "std_error": bse,
        "statistic_z_or_t": stat,
        "p_value": pval,
        "ci_95_log_odds": [ci_lower, ci_upper],
        "odds_ratio": odds_ratio,
        "ci_95_odds_ratio": [or_ci_lower, or_ci_upper],
        "significant_p_lt_0_05": signif,
        "notes": "Model assumed binomial GLM (logit link). SEs/p-values may be cluster-robust if model output was produced with clustered cov."
    }

    # Construct a brief description/interpretation
    coef_str = _fmt(coef, "{:.4g}", default="NA")
    odds_ratio_str = _fmt(odds_ratio, "{:.3g}", default="NA")
    pval_str = _fmt(pval, "{:.3g}", default="NA")
    ci_str = f"[{_fmt(ci_lower, '{:.4g}', default='NA')}, {_fmt(ci_upper, '{:.4g}', default='NA')}]"
    or_ci_str = f"[{_fmt(or_ci_lower, '{:.3g}', default='NA')}, {_fmt(or_ci_upper, '{:.3g}', default='NA')}]"

    if coef is not None:
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        if pval is not None:
            if signif:
                interpretation = (
                    f"The estimated effect of 'is_human' is {coef_str} (log-odds), "
                    f"corresponding to an odds ratio of {odds_ratio_str}. The two-sided p-value = {pval_str}, "
                    f"which is < 0.05, indicating evidence that modern humans have {direction} AMTL rates than "
                    "non-human primates after controlling for age, prob_male, tooth_class, and population, "
                    "and accounting for clustering by specimen when robust SEs were applied."
                )
            else:
                interpretation = (
                    f"The estimated effect of 'is_human' is {coef_str} (log-odds), "
                    f"corresponding to an odds ratio of {odds_ratio_str}. The two-sided p-value = {pval_str}, "
                    f"which is not < 0.05, so there is not strong evidence to conclude that modern humans differ "
                    "in AMTL rates from non-human primates after adjustment for the listed covariates."
                )
        else:
            interpretation = (
                f"The estimated effect of 'is_human' is {coef_str} (log-odds), corresponding to an odds ratio of "
                f"{odds_ratio_str}. No p-value was available on the model output; inferential testing could not be "
                "completed here (you may compute p-values from coef/std_error if desired)."
            )
    else:
        interpretation = (
            "Could not extract a numeric coefficient for 'is_human'. "
            f"Extracted object: coef={coef_str}, se={_fmt(bse)}, stat={_fmt(stat)}, pvalue={pval_str}, "
            f"ci_95_log_odds={ci_str}, odds_ratio={odds_ratio_str}, ci_95_odds_ratio={or_ci_str}."
        )

    return {
        "object": result_object,
        "description": interpretation
    }