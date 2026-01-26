def extract_final_answer(model_output):
    """
    Extract coefficients, robust SEs, p-values, 95% CIs, odds ratios, and a plain-language
    interpretation for the genus contrasts from a fitted statsmodels GLMResults (or its
    robustcov_results wrapper) where the model used
      C(genus, Treatment(reference="Homo sapiens"))
    so that coefficients correspond to (non-human genus) minus (Homo sapiens).

    Returns a dict with keys:
      - "object": a dict with per-genus statistics and interpretations
      - "description": brief explanation of the returned object and how to read it
    """
    import re
    import numpy as np
    from math import exp
    from scipy.stats import norm

    res = model_output

    # Get parameter names and values
    params = getattr(res, "params", None)
    if params is None:
        raise ValueError("model_output has no 'params' attribute")

    # Try to get standard errors and p-values; if not available, compute from cov_params()
    try:
        bse = res.bse
    except Exception:
        cov = res.cov_params()
        bse = np.sqrt(np.diag(cov))

    try:
        pvalues = res.pvalues
    except Exception:
        # two-sided p-values from normal approximation
        z = params / bse
        pvalues = 2 * (1 - norm.cdf(np.abs(z)))

    # Try to get conf_int (may already use robust cov if res was robustified)
    try:
        ci_df = res.conf_int()  # returns DataFrame-like with two columns [0]=lower, [1]=upper
    except Exception:
        zval = norm.ppf(0.975)
        lower = params - zval * bse
        upper = params + zval * bse
        ci_df = np.vstack([lower, upper]).T

    # Identify genus contrast parameter names (those representing levels vs Homo sapiens)
    # Typical name from patsy: 'C(genus, Treatment(reference="Homo sapiens"))[T.Pan]'
    genus_param_pattern = re.compile(r'\[T\.(.+?)\]')  # capture level after T.
    comparisons = {}

    for name in params.index:
        m = genus_param_pattern.search(name)
        if m and ('genus' in name or 'C(genus' in name):
            level = m.group(1)
            coef = float(params[name])
            se = float(bse[name]) if hasattr(bse, "__getitem__") else float(bse[params.index.get_loc(name)])
            pval = float(pvalues[name]) if hasattr(pvalues, "__getitem__") else float(pvalues[params.index.get_loc(name)])
            try:
                ci_low = float(ci_df.loc[name][0])
                ci_high = float(ci_df.loc[name][1])
            except Exception:
                # ci_df may be ndarray
                idx = list(params.index).index(name)
                ci_low, ci_high = float(ci_df[idx, 0]), float(ci_df[idx, 1])

            # For binomial-logit model: coef is log-odds difference (non-human minus Homo).
            or_point = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))

            # Interpretation:
            # - If coef < 0 and significant -> non-human has lower log-odds than Homo -> Homo has higher AMTL
            # - If coef > 0 and significant -> non-human has higher AMTL
            alpha = 0.05
            if pval < alpha:
                if coef < 0:
                    verdict = ("Statistically significant: {} has LOWER AMTL than Homo sapiens (p={:.3g}). "
                               "Thus Homo sapiens have higher AMTL than {}.").format(level, pval, level)
                else:
                    verdict = ("Statistically significant: {} has HIGHER AMTL than Homo sapiens (p={:.3g}). "
                               "Thus {} has higher AMTL than Homo sapiens.").format(level, pval, level)
            else:
                verdict = ("No statistically significant difference in AMTL between {} and Homo sapiens "
                           "(p={:.3g}).").format(level, pval)

            comparisons[level] = {
                "param_name": name,
                "coef_log_odds": coef,
                "std_error": se,
                "p_value": pval,
                "95ci_log_odds": [ci_low, ci_high],
                "odds_ratio": or_point,
                "95ci_odds_ratio": [or_ci_low, or_ci_high],
                "interpretation": verdict
            }

    # Create a brief overall summary across genera
    summary_lines = []
    homo_higher_count = 0
    nonhuman_higher_count = 0
    nonsig_count = 0
    for g, info in comparisons.items():
        p = info["p_value"]
        coef = info["coef_log_odds"]
        if p < 0.05:
            if coef < 0:
                homo_higher_count += 1
            else:
                nonhuman_higher_count += 1
        else:
            nonsig_count += 1

    summary = {
        "n_comparisons": len(comparisons),
        "homo_sapiens_higher_significant": homo_higher_count,
        "nonhuman_higher_significant": nonhuman_higher_count,
        "no_significant_difference": nonsig_count
    }

    result_object = {
        "comparisons": comparisons,
        "summary_counts": summary
    }

    description = (
        "For each non-human genus (Pan, Pongo, Papio) this returns the GLM coefficient (log-odds difference: "
        "non-human minus Homo sapiens), its standard error, p-value, 95% confidence interval on the log-odds scale, "
        "the corresponding odds ratio and its 95% CI, and a plain-language interpretation indicating whether Homo "
        "sapiens has higher AMTL, the non-human genus has higher AMTL, or there is no significant difference."
    )

    return {"object": result_object, "description": description}