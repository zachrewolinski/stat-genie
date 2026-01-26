def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of having children on extramarital affairs
    from the provided model_output dict with keys:
      - 'logit': fitted statsmodels Logit result (BinaryResultsWrapper)
      - 'count_nb': fitted statsmodels GLMResultsWrapper (NegativeBinomial) or None

    Returns:
      {
        "object": {
          "logistic": {
            "female": {coef, se, z, p, ci_lower, ci_upper, odds_ratio, or_ci_lower, or_ci_upper, significant},
            "male":   {same fields computed for (Children + Children_Gender)},
            "notes": textual note about interpretation
          },
          "count": None or {
            "female": {coef, se, z, p, ci_lower, ci_upper, irr, irr_ci_lower, irr_ci_upper, significant},
            "male":   {same for combined effect},
            "notes": textual note about interpretation (IRR = incidence rate ratio)
          }
        },
        "description": "Brief human-readable interpretation"
      }
    """
    import numpy as np
    from math import exp, sqrt
    from scipy.stats import norm

    out = {"logistic": None, "count": None}
    # helper to format results dict
    def make_result_dict(coef, se, pv, ci_low, ci_high, transform=None):
        z = coef / se if se > 0 else np.nan
        significant = (pv < 0.05)
        res = {
            "coef": float(coef),
            "se": float(se),
            "z": float(z),
            "p": float(pv),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
            "significant": bool(significant)
        }
        if transform == "exp":  # return exponentiated effect (OR or IRR) with CI
            res.update({
                "exp_coef": float(exp(coef)),
                "exp_ci_lower": float(exp(ci_low)),
                "exp_ci_upper": float(exp(ci_high))
            })
        return res

    # Validate presence
    if not isinstance(model_output, dict) or 'logit' not in model_output:
        raise ValueError("model_output must be a dict containing at least the 'logit' result object.")

    logit_res = model_output['logit']
    params = logit_res.params
    bse = logit_res.bse
    pvals = logit_res.pvalues
    conf = logit_res.conf_int()
    cov = logit_res.cov_params()

    req_names = ['Children', 'Children_Gender']
    for name in req_names:
        if name not in params.index:
            raise KeyError(f"Expected coefficient '{name}' not found in logit model parameters. Found: {list(params.index)}")

    # Female effect (GenderMale == 0): coefficient on Children
    coef_f = params['Children']
    se_f = bse['Children']
    p_f = pvals['Children']
    ci_f = conf.loc['Children'].tolist()
    logistic_female = make_result_dict(coef_f, se_f, p_f, ci_f[0], ci_f[1], transform="exp")

    # Male effect (GenderMale == 1): Children + Children_Gender
    coef_c = params['Children']
    coef_int = params['Children_Gender']
    coef_m = coef_c + coef_int
    # variance var(C) + var(Int) + 2*cov(C,Int)
    var_c = cov.loc['Children', 'Children']
    var_int = cov.loc['Children_Gender', 'Children_Gender']
    cov_c_int = cov.loc['Children', 'Children_Gender']
    var_m = var_c + var_int + 2 * cov_c_int
    se_m = sqrt(var_m) if var_m >= 0 else float('nan')
    # compute z and p
    z_m = coef_m / se_m if se_m > 0 else np.nan
    p_m = 2 * norm.sf(abs(z_m)) if se_m > 0 else np.nan
    # CI on log-odds scale
    ci_m_lower = coef_m - 1.96 * se_m
    ci_m_upper = coef_m + 1.96 * se_m
    logistic_male = make_result_dict(coef_m, se_m, p_m, ci_m_lower, ci_m_upper, transform="exp")

    out["logistic"] = {
        "female": logistic_female,
        "male": logistic_male,
        "notes": "Logistic model predicting probability of any affair. Coefficients are log-odds; exp(coef) is the odds ratio (OR)."
    }

    # Count model (if present)
    nb_res = model_output.get('count_nb')
    if nb_res is None:
        out["count"] = None
    else:
        params_c = nb_res.params
        bse_c = nb_res.bse
        try:
            pvals_c = nb_res.pvalues
        except Exception:
            # GLM results sometimes don't have pvalues? fallback to z from coef/bse
            pvals_c = (2 * norm.sf(np.abs(params_c / bse_c)))
            pvals_c = pvals_c.reindex(params_c.index)

        conf_c = nb_res.conf_int()
        cov_c = nb_res.cov_params()

        for name in req_names:
            if name not in params_c.index:
                raise KeyError(f"Expected coefficient '{name}' not found in count model parameters. Found: {list(params_c.index)}")

        # Female (GenderMale == 0)
        coef_f_c = params_c['Children']
        se_f_c = bse_c['Children']
        p_f_c = pvals_c['Children']
        ci_f_c = conf_c.loc['Children'].tolist()
        # For count model interpret exp(coef) as incidence rate ratio (IRR)
        count_female = make_result_dict(coef_f_c, se_f_c, p_f_c, ci_f_c[0], ci_f_c[1], transform="exp")

        # Male: sum coefficients
        coef_c_c = params_c['Children']
        coef_int_c = params_c['Children_Gender']
        coef_m_c = coef_c_c + coef_int_c
        var_c_c = cov_c.loc['Children', 'Children']
        var_int_c = cov_c.loc['Children_Gender', 'Children_Gender']
        cov_cint_c = cov_c.loc['Children', 'Children_Gender']
        var_m_c = var_c_c + var_int_c + 2 * cov_cint_c
        se_m_c = sqrt(var_m_c) if var_m_c >= 0 else float('nan')
        z_m_c = coef_m_c / se_m_c if se_m_c > 0 else np.nan
        p_m_c = 2 * norm.sf(abs(z_m_c)) if se_m_c > 0 else np.nan
        ci_m_c_lower = coef_m_c - 1.96 * se_m_c
        ci_m_c_upper = coef_m_c + 1.96 * se_m_c
        count_male = make_result_dict(coef_m_c, se_m_c, p_m_c, ci_m_c_lower, ci_m_c_upper, transform="exp")

        out["count"] = {
            "female": count_female,
            "male": count_male,
            "notes": "Count model predicting number of affairs among those with any affair. Coefficients are on log scale; exp(coef) is the incidence rate ratio (IRR)."
        }

    # Build short human-readable description
    desc_lines = []
    desc_lines.append("From the logistic model (probability of any affair):")
    desc_lines.append(f"  - Females (GenderMale=0): Children coef={logistic_female['coef']:.4f}, OR={logistic_female['exp_coef']:.3f}, p={logistic_female['p']:.3f}")
    desc_lines.append(f"  - Males   (GenderMale=1): Children effect (Children + interaction) coef={logistic_male['coef']:.4f}, OR={logistic_male['exp_coef']:.3f}, p={logistic_male['p']:.3f}")
    if out["count"] is not None:
        desc_lines.append("From the count model (frequency among those with any affair):")
        desc_lines.append(f"  - Females: Children coef={out['count']['female']['coef']:.4f}, IRR={out['count']['female']['exp_coef']:.3f}, p={out['count']['female']['p']:.3f}")
        desc_lines.append(f"  - Males:   Children effect coef={out['count']['male']['coef']:.4f}, IRR={out['count']['male']['exp_coef']:.3f}, p={out['count']['male']['p']:.3f}")
    else:
        desc_lines.append("No reliable count model was fitted (count_nb is None).")

    description = " ".join(desc_lines)

    return {"object": out, "description": description}