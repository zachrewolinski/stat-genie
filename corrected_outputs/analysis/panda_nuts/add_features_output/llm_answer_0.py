def extract_final_answer(model_output):
    """
    Extract relevant statistics from a fitted statsmodels MixedLMResults (or wrapper)
    to answer how age, sex, and receiving help influence log-transformed nut-cracking efficiency.

    Returns a dictionary with:
      - "object": a dict containing coefficients, standard errors, z-stats, p-values,
                  95% CIs, and interpretable percent-change effects for:
                    * age (when no help)
                    * age when help is received (age + age_help)
                    * sex_male (male vs female)
                    * help_yes (receiving help vs not)
      - "description": a brief explanation of what these numbers mean.

    The function handles the linear combination for the interaction (age * help_yes),
    computes standard errors and p-values for that combination using the covariance matrix,
    and converts log-scale coefficients to approximate percent changes in efficiency.
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output

    # Access params, bse, pvalues, conf_int, cov_params in a robust way
    params = getattr(res, "params")
    pvalues = getattr(res, "pvalues", None)
    conf_int_df = None
    try:
        conf_int_df = res.conf_int()
    except Exception:
        conf_int_df = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Names we care about
    names = list(params.index)

    def get_param(name):
        if name in params.index:
            coef = float(params.loc[name])
            se = float(np.sqrt(cov.loc[name, name])) if cov is not None else (float(getattr(res, "bse").loc[name]) if hasattr(res, "bse") else None)
            p = float(pvalues.loc[name]) if (pvalues is not None and name in pvalues.index) else None
            if conf_int_df is not None and name in conf_int_df.index:
                ci_low = float(conf_int_df.loc[name, 0])
                ci_high = float(conf_int_df.loc[name, 1])
            else:
                ci_low = ci_high = None
            return {"coef": coef, "se": se, "p": p, "ci_lower": ci_low, "ci_upper": ci_high}
        else:
            return None

    # Extract basic params
    age_res = get_param("age")
    sex_res = get_param("sex_male")
    help_res = get_param("help_yes")
    age_help_res = get_param("age_help")

    # Helper to convert log-coef to percent change
    def pct_change_from_log(coef):
        # multiplicative change = exp(coef); percent change = (exp(coef)-1)*100
        return (np.exp(coef) - 1) * 100

    # Compute combined effect: effect of age when help_yes = 1 -> age + age_help
    combined = {}
    if age_res is not None:
        # age when help = 0 (baseline)
        coef_age_nohelp = age_res["coef"]
        se_age_nohelp = age_res["se"]
        p_age_nohelp = age_res["p"]
        ci_low_age_nohelp = age_res["ci_lower"]
        ci_high_age_nohelp = age_res["ci_upper"]

        combined["age_no_help"] = {
            "coef_log": coef_age_nohelp,
            "se": se_age_nohelp,
            "p": p_age_nohelp,
            "ci_log": (ci_low_age_nohelp, ci_high_age_nohelp),
            "pct_change_per_year": pct_change_from_log(coef_age_nohelp),
        }

        # age when help = 1
        if age_help_res is not None and cov is not None:
            # Estimate and se for linear combination (age + age_help)
            coef_age_help = coef_age_nohelp + age_help_res["coef"]
            # Build weight vector aligned to params
            w = np.zeros(len(names))
            name_to_idx = {n: i for i, n in enumerate(names)}
            w[name_to_idx["age"]] = 1.0
            w[name_to_idx["age_help"]] = 1.0
            # compute variance and se
            cov_mat = np.asarray(cov.loc[names, names])
            var_lincomb = float(w @ cov_mat @ w)
            se_lincomb = float(np.sqrt(var_lincomb))
            z = coef_age_help / se_lincomb if se_lincomb > 0 else None
            p_lincomb = float(2 * (1 - norm.cdf(abs(z)))) if z is not None else None
            # CI on log scale
            ci_low = coef_age_help - norm.ppf(0.975) * se_lincomb
            ci_high = coef_age_help + norm.ppf(0.975) * se_lincomb

            combined["age_with_help"] = {
                "coef_log": coef_age_help,
                "se": se_lincomb,
                "z": z,
                "p": p_lincomb,
                "ci_log": (ci_low, ci_high),
                "pct_change_per_year": pct_change_from_log(coef_age_help),
            }
        else:
            combined["age_with_help"] = {
                "coef_log": None,
                "se": None,
                "p": None,
                "ci_log": (None, None),
                "pct_change_per_year": None,
            }
    else:
        combined["age_no_help"] = combined["age_with_help"] = None

    # Process sex and help main effects
    sex_summary = None
    if sex_res is not None:
        coef = sex_res["coef"]
        se = sex_res["se"]
        p = sex_res["p"]
        ci = (sex_res["ci_lower"], sex_res["ci_upper"])
        sex_summary = {
            "coef_log_male_vs_female": coef,
            "se": se,
            "p": p,
            "ci_log": ci,
            "pct_change_male_vs_female": pct_change_from_log(coef),
        }

    help_summary = None
    if help_res is not None:
        coef = help_res["coef"]
        se = help_res["se"]
        p = help_res["p"]
        ci = (help_res["ci_lower"], help_res["ci_upper"])
        help_summary = {
            "coef_log_help_vs_nohelp": coef,
            "se": se,
            "p": p,
            "ci_log": ci,
            "pct_change_help_vs_nohelp": pct_change_from_log(coef),
        }

    # Pack results
    out = {
        "coefficients": {
            "age": age_res,
            "age_help_interaction": age_help_res,
            "sex_male": sex_res,
            "help_yes": help_res,
        },
        "derived": combined,
        "sex_effect": sex_summary,
        "help_effect": help_summary,
        "notes": {
            "interpretation": "Coefficients are on the log(Efficiency) scale. "
                              "Percent-change values show (exp(coef)-1)*100: the approximate percent change in nuts/min associated with a one-unit change in the predictor.",
            "age_with_help_calculation": "Effect of age when help=1 is (coef_age + coef_age_help). Standard error and p-value computed via the covariance matrix (linear combination).",
        },
    }

    # Short human-readable description
    # State significance where available
    def sig_text(p):
        if p is None:
            return "p unknown"
        return ("significant (p < 0.05)" if p < 0.05 else f"not significant (p = {p:.3f})")

    desc_lines = []
    if combined.get("age_no_help") is not None:
        a0 = combined["age_no_help"]
        desc_lines.append(
            f"Age (no help): coef(log)={a0['coef_log']:.4f}, se={a0['se']:.4f}, p={a0['p']:.3f} -> "
            f"{a0['pct_change_per_year']:.2f}% change in efficiency per year (when not helped)."
            if a0['p'] is not None else
            f"Age (no help): coef(log)={a0['coef_log']:.4f} (p unknown)."
        )
    if combined.get("age_with_help") is not None and combined["age_with_help"]["coef_log"] is not None:
        a1 = combined["age_with_help"]
        desc_lines.append(
            f"Age (with help): coef(log)={a1['coef_log']:.4f}, se={a1['se']:.4f}, p={a1['p']:.3f} -> "
            f"{a1['pct_change_per_year']:.2f}% change in efficiency per year (when helped)."
            if a1['p'] is not None else
            f"Age (with help): coef(log)={a1['coef_log']:.4f} (p unknown)."
        )
    if sex_summary is not None:
        desc_lines.append(
            f"Sex (male vs female): coef(log)={sex_summary['coef_log_male_vs_female']:.4f}, "
            f"p={sex_summary['p']:.3f} -> {sex_summary['pct_change_male_vs_female']:.2f}% difference in efficiency for males vs females."
            if sex_summary['p'] is not None else
            f"Sex (male vs female): coef(log)={sex_summary['coef_log_male_vs_female']:.4f} (p unknown)."
        )
    if help_summary is not None:
        desc_lines.append(
            f"Receiving help (yes vs no): coef(log)={help_summary['coef_log_help_vs_nohelp']:.4f}, "
            f"p={help_summary['p']:.3f} -> {help_summary['pct_change_help_vs_nohelp']:.2f}% difference in efficiency when helped."
            if help_summary['p'] is not None else
            f"Receiving help (yes vs no): coef(log)={help_summary['coef_log_help_vs_nohelp']:.4f} (p unknown)."
        )

    description = " | ".join(desc_lines) if desc_lines else "No relevant parameters found in model output."

    return {"object": out, "description": description}