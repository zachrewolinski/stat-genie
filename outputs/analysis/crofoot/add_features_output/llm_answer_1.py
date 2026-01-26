def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels logistic regression (BinaryResultsWrapper).
    Expects the model to include the regressors:
      - 'rel_size_ratio_z'
      - 'focal_home'
      - 'rel_by_home'   (interaction = rel_size_ratio_z * focal_home)

    Returns a dictionary with:
      - "object": dict containing coefficients, SEs, z-stats, p-values, 95% CIs, odds-ratios and CIs
                  for the main terms and for the combined effect of relative size when focal_home=1.
      - "description": short interpretation guidance about what the numbers mean for the research question.
    """
    import numpy as np
    from math import exp, sqrt
    from scipy import stats

    # Helper to safe-get parameter or raise informative error
    def _get_param(name, params):
        if name not in params.index:
            raise KeyError(f"Model result does not contain parameter '{name}'. Available params: {list(params.index)}")
        return params[name]

    # Extract fundamental results objects
    try:
        params = model_output.params        # pandas Series
        cov = model_output.cov_params()     # DataFrame or ndarray-like
        # Many statsmodels results also expose bse, pvalues, conf_int
        bse = getattr(model_output, "bse", None)
        pvalues = getattr(model_output, "pvalues", None)
        conf_int = None
        try:
            conf_int = model_output.conf_int()
        except Exception:
            conf_int = None
    except Exception as e:
        raise RuntimeError(f"Failed to access parameters/covariance from model_output: {e}")

    terms = ['rel_size_ratio_z', 'focal_home', 'rel_by_home']

    results = {}
    for t in terms:
        try:
            coef = float(_get_param(t, params))
        except KeyError:
            # If a term is missing, skip it but note absence
            results[t] = {"present": False, "message": f"Term '{t}' not found in model."}
            continue

        # Standard error: prefer bse if available, otherwise use sqrt of diagonal of cov
        if bse is not None and t in bse.index:
            se = float(bse[t])
        else:
            try:
                se = float(np.sqrt(float(cov.loc[t, t])))
            except Exception:
                se = None

        # z-stat and p-value (use normal approximation)
        if se is not None and se > 0:
            z = coef / se
            p = float(2 * (1 - stats.norm.cdf(abs(z))))
        else:
            z = None
            p = None

        # 95% CI: prefer conf_int if available, otherwise coef +/- 1.96*se
        if conf_int is not None and t in conf_int.index:
            ci_lower, ci_upper = float(conf_int.loc[t, 0]), float(conf_int.loc[t, 1])
        elif se is not None:
            ci_lower, ci_upper = coef - 1.96 * se, coef + 1.96 * se
        else:
            ci_lower, ci_upper = None, None

        # Odds ratio and CI on OR scale
        try:
            or_val = exp(coef)
            or_ci = (exp(ci_lower) if ci_lower is not None else None,
                     exp(ci_upper) if ci_upper is not None else None)
        except Exception:
            or_val = None
            or_ci = (None, None)

        results[t] = {
            "present": True,
            "coef": coef,
            "se": se,
            "z": z,
            "p_value": p,
            "ci95": (ci_lower, ci_upper),
            "odds_ratio": or_val,
            "odds_ratio_ci95": or_ci
        }

    # Compute combined effect: effect of rel_size when focal_home == 1
    # coef_comb = coef_rel + coef_interaction
    if (results.get('rel_size_ratio_z', {}).get('present') and
        results.get('rel_by_home', {}).get('present')):
        a = float(params['rel_size_ratio_z'])
        b = float(params['rel_by_home'])
        coef_comb = a + b

        # variance = var(a) + var(b) + 2*cov(a,b)
        try:
            var_a = float(cov.loc['rel_size_ratio_z', 'rel_size_ratio_z'])
            var_b = float(cov.loc['rel_by_home', 'rel_by_home'])
            cov_ab = float(cov.loc['rel_size_ratio_z', 'rel_by_home'])
            var_comb = var_a + var_b + 2 * cov_ab
            se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 else None
        except Exception:
            se_comb = None

        if se_comb is not None and se_comb > 0:
            z_comb = coef_comb / se_comb
            p_comb = float(2 * (1 - stats.norm.cdf(abs(z_comb))))
            ci_lower_comb = coef_comb - 1.96 * se_comb
            ci_upper_comb = coef_comb + 1.96 * se_comb
        else:
            z_comb = None
            p_comb = None
            ci_lower_comb = None
            ci_upper_comb = None

        try:
            or_comb = exp(coef_comb)
            or_comb_ci = (exp(ci_lower_comb) if ci_lower_comb is not None else None,
                          exp(ci_upper_comb) if ci_upper_comb is not None else None)
        except Exception:
            or_comb = None
            or_comb_ci = (None, None)

        results['rel_size_when_focal_home_1'] = {
            "coef": coef_comb,
            "se": se_comb,
            "z": z_comb,
            "p_value": p_comb,
            "ci95": (ci_lower_comb, ci_upper_comb),
            "odds_ratio": or_comb,
            "odds_ratio_ci95": or_comb_ci,
            "interpretation_note": "This is the effect (log-odds) of a one-SD increase in relative group size when the contest is in the focal group's home area (focal_home=1)."
        }
    else:
        results['rel_size_when_focal_home_1'] = {
            "present": False,
            "message": "Could not compute combined effect because one or both terms are missing."
        }

    # Also compute effect of focal_home at rel_size = 0 (this is the focal_home main effect)
    # and at rel_size = +1 SD (i.e., focal_home effect when rel_size_z = 1)
    if results.get('focal_home', {}).get('present'):
        fh_coef = float(params['focal_home'])
        # var for focal_home
        try:
            var_fh = float(cov.loc['focal_home', 'focal_home'])
            se_fh = float(np.sqrt(var_fh))
            z_fh = fh_coef / se_fh
            p_fh = float(2 * (1 - stats.norm.cdf(abs(z_fh))))
            ci_lower_fh = fh_coef - 1.96 * se_fh
            ci_upper_fh = fh_coef + 1.96 * se_fh
            or_fh = exp(fh_coef)
            or_fh_ci = (exp(ci_lower_fh), exp(ci_upper_fh))
        except Exception:
            se_fh = z_fh = p_fh = ci_lower_fh = ci_upper_fh = or_fh = or_fh_ci = None

        results['focal_home_at_rel_size_0'] = {
            "coef": fh_coef,
            "se": se_fh,
            "z": z_fh,
            "p_value": p_fh,
            "ci95": (ci_lower_fh, ci_upper_fh),
            "odds_ratio": or_fh,
            "odds_ratio_ci95": or_fh_ci,
            "interpretation_note": "This is the effect (log-odds) of being in the focal group's home area when rel_size_z = 0 (i.e., at mean relative size)."
        }

        # focal_home effect at rel_size = 1 (coef = focal_home + rel_by_home*1)
        if results.get('rel_by_home', {}).get('present'):
            c = float(params['rel_by_home'])
            coef_fh_r1 = fh_coef + c
            # variance = var(fh) + var(c) + 2cov(fh,c)
            try:
                var_c = float(cov.loc['rel_by_home', 'rel_by_home'])
                cov_fh_c = float(cov.loc['focal_home', 'rel_by_home'])
                var_sum = var_fh + var_c + 2 * cov_fh_c
                se_sum = float(np.sqrt(var_sum))
                z_sum = coef_fh_r1 / se_sum
                p_sum = float(2 * (1 - stats.norm.cdf(abs(z_sum))))
                ci_low_sum = coef_fh_r1 - 1.96 * se_sum
                ci_up_sum = coef_fh_r1 + 1.96 * se_sum
                or_sum = exp(coef_fh_r1)
                or_sum_ci = (exp(ci_low_sum), exp(ci_up_sum))
            except Exception:
                se_sum = z_sum = p_sum = ci_low_sum = ci_up_sum = or_sum = or_sum_ci = None

            results['focal_home_at_rel_size_1'] = {
                "coef": coef_fh_r1,
                "se": se_sum,
                "z": z_sum,
                "p_value": p_sum,
                "ci95": (ci_low_sum, ci_up_sum),
                "odds_ratio": or_sum,
                "odds_ratio_ci95": or_sum_ci,
                "interpretation_note": "Effect of focal_home when relative size is +1 SD."
            }

    # Compose brief description for interpreting results in context
    description_lines = [
        "Extracted regression coefficients, standard errors, z-stats, p-values, 95% CIs, and odds-ratios for:",
        " - rel_size_ratio_z: effect of relative group size (per 1 SD) on log-odds of the focal group winning when focal_home=0 (i.e., baseline).",
        " - focal_home: effect of contest being in focal group's home when rel_size_z = 0 (mean relative size).",
        " - rel_by_home: interaction term; whether home-field moderates the effect of relative group size.",
        "Also computed:",
        " - rel_size_when_focal_home_1: the effect of relative group size when the contest is in the focal group's home (sum of rel_size_ratio_z and rel_by_home).",
        " - focal_home_at_rel_size_0 and focal_home_at_rel_size_1: focal_home effect at mean rel_size and at +1 SD rel_size, respectively.",
        "",
        "How to interpret:",
        " - For any coefficient: positive coef -> higher log-odds (and odds ratio >1) of focal group winning; negative -> lower odds.",
        " - Use p-values (two-sided) to assess statistical significance (conventional threshold p<0.05).",
        " - If rel_size_ratio_z is positive and significant, larger relative group size increases probability of winning.",
        " - If rel_by_home is significant, home-field modifies the size effect; check rel_size_when_focal_home_1 to see the size effect inside focal home.",
    ]
    description = "\n".join(description_lines)

    return {"object": results, "description": description}