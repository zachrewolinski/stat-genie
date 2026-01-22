def extract_final_answer(model_output):
    """
    Extract statistics about the effect of having children on extramarital affairs
    from the provided model_output dict containing:
      - 'neg_binom': fitted (robust) GLM NegativeBinomial results wrapper
      - 'logit': fitted (robust) Logit results wrapper

    Returns:
      {
        "object": {
            "neg_binom": {
                "coef_children": float,
                "se_children": float,
                "p_children": float,
                "irr_children": float,
                "irr_children_ci95": (lower, upper),
                "coef_children_male": float,
                "se_children_male": float,
                "p_children_male": float,
                "irr_children_male": float,
                "irr_children_male_ci95": (lower, upper)
            },
            "logit": {
                "coef_children": float,
                "se_children": float,
                "p_children": float,
                "or_children": float,
                "or_children_ci95": (lower, upper),
                "coef_children_male": float,
                "se_children_male": float,
                "p_children_male": float,
                "or_children_male": float,
                "or_children_male_ci95": (lower, upper)
            }
        },
        "description": str
      }
    The description gives a concise interpretation of whether having children decreases
    engagement in extramarital affairs (overall / for females) and whether that effect
    differs for males (based on the children_x_male interaction).
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing 'neg_binom' and 'logit' results.")

    if 'neg_binom' not in model_output or 'logit' not in model_output:
        raise KeyError("model_output must contain keys 'neg_binom' and 'logit'")

    neg = model_output['neg_binom']
    logit = model_output['logit']

    # convenience to extract named objects
    def safe_get(series_like, name, label):
        # handle pandas Series-like with index, or dict-like
        try:
            idx = list(series_like.index)
        except Exception:
            # fallback for dict-like
            try:
                idx = list(series_like.keys())
            except Exception:
                idx = []
        if name not in idx and name not in getattr(series_like, '__dict__', {}):
            raise KeyError(f"Variable '{name}' not found in model {label}. Available: {idx}")
        # prefer Series/dict access
        try:
            return series_like[name]
        except Exception:
            # last resort: attribute access
            return getattr(series_like, name)

    results = {"neg_binom": {}, "logit": {}}

    # --- NEGATIVE BINOMIAL (count outcome) ---
    try:
        params_nb = neg.params
        bse_nb = neg.bse
        pvals_nb = neg.pvalues
        ci_nb = neg.conf_int()
        cov_nb = neg.cov_params()
    except Exception as e:
        raise RuntimeError("Failed to extract pieces from neg_binom results: " + str(e))

    # required term names
    term = 'children_binary'
    interaction = 'children_x_male'

    # extract main coef for children (female baseline when gender_male=0)
    coef_ch = safe_get(params_nb, term, 'neg_binom')
    se_ch = safe_get(bse_nb, term, 'neg_binom')
    p_ch = safe_get(pvals_nb, term, 'neg_binom')
    ci_ch = ci_nb.loc[term].tolist()  # [low, high]
    irr_ch = float(np.exp(coef_ch))
    irr_ch_ci = (float(np.exp(ci_ch[0])), float(np.exp(ci_ch[1])))

    # extract interaction coef
    coef_int = safe_get(params_nb, interaction, 'neg_binom')
    # compute combined effect for males: children effect + interaction
    coef_ch_male = coef_ch + coef_int

    # compute se for combined effect using covariance matrix
    var_ch = cov_nb.loc[term, term]
    var_int = cov_nb.loc[interaction, interaction]
    cov_ch_int = cov_nb.loc[term, interaction]
    se_ch_male = sqrt(var_ch + var_int + 2.0 * cov_ch_int)
    # p-value for combined
    z_ch_male = coef_ch_male / se_ch_male
    p_ch_male = 2.0 * (1.0 - stats.norm.cdf(abs(z_ch_male)))
    # IRR and CI for males
    irr_ch_male = float(np.exp(coef_ch_male))
    # CI for combined: use Wald CI from coef_ch_male +/- 1.96*se
    ci_ch_male = (float(np.exp(coef_ch_male - 1.96 * se_ch_male)), float(np.exp(coef_ch_male + 1.96 * se_ch_male)))

    results['neg_binom'] = {
        "coef_children": float(coef_ch),
        "se_children": float(se_ch),
        "p_children": float(p_ch),
        "irr_children": irr_ch,
        "irr_children_ci95": irr_ch_ci,
        "coef_children_male": float(coef_ch_male),
        "se_children_male": float(se_ch_male),
        "p_children_male": float(p_ch_male),
        "irr_children_male": irr_ch_male,
        "irr_children_male_ci95": ci_ch_male
    }

    # --- LOGIT (binary any_affair outcome) ---
    try:
        params_log = logit.params
        bse_log = logit.bse
        pvals_log = logit.pvalues
        ci_log = logit.conf_int()
        cov_log = logit.cov_params()
    except Exception as e:
        raise RuntimeError("Failed to extract pieces from logit results: " + str(e))

    # main effect
    coef_ch_l = safe_get(params_log, term, 'logit')
    se_ch_l = safe_get(bse_log, term, 'logit')
    p_ch_l = safe_get(pvals_log, term, 'logit')
    ci_ch_l = ci_log.loc[term].tolist()
    or_ch = float(np.exp(coef_ch_l))
    or_ch_ci = (float(np.exp(ci_ch_l[0])), float(np.exp(ci_ch_l[1])))

    # interaction
    coef_int_l = safe_get(params_log, interaction, 'logit')
    coef_ch_male_l = coef_ch_l + coef_int_l
    # se for combined (male)
    var_ch_l = cov_log.loc[term, term]
    var_int_l = cov_log.loc[interaction, interaction]
    cov_ch_int_l = cov_log.loc[term, interaction]
    se_ch_male_l = sqrt(var_ch_l + var_int_l + 2.0 * cov_ch_int_l)
    z_ch_male_l = coef_ch_male_l / se_ch_male_l
    p_ch_male_l = 2.0 * (1.0 - stats.norm.cdf(abs(z_ch_male_l)))
    or_ch_male = float(np.exp(coef_ch_male_l))
    or_ch_male_ci = (float(np.exp(coef_ch_male_l - 1.96 * se_ch_male_l)), float(np.exp(coef_ch_male_l + 1.96 * se_ch_male_l)))

    results['logit'] = {
        "coef_children": float(coef_ch_l),
        "se_children": float(se_ch_l),
        "p_children": float(p_ch_l),
        "or_children": or_ch,
        "or_children_ci95": or_ch_ci,
        "coef_children_male": float(coef_ch_male_l),
        "se_children_male": float(se_ch_male_l),
        "p_children_male": float(p_ch_male_l),
        "or_children_male": or_ch_male,
        "or_children_male_ci95": or_ch_male_ci
    }

    # Short interpretation text based on significance/direction
    def interpret_count(coef, pval, irr):
        if pval < 0.05:
            if irr < 1.0:
                return "Significant decrease in expected count (IRR<1)."
            else:
                return "Significant increase in expected count (IRR>1)."
        else:
            return "No statistically significant effect detected (p>=0.05)."

    def interpret_binary(coef, pval, orr):
        if pval < 0.05:
            if orr < 1.0:
                return "Significant decrease in odds of any affair (OR<1)."
            else:
                return "Significant increase in odds of any affair (OR>1)."
        else:
            return "No statistically significant effect detected (p>=0.05)."

    desc_lines = []
    # Females (baseline)
    desc_lines.append("Negative binomial (count): children effect (females): " +
                      interpret_count(results['neg_binom']['coef_children'],
                                      results['neg_binom']['p_children'],
                                      results['neg_binom']['irr_children']))
    # Males
    desc_lines.append("Negative binomial (count): children effect (males): " +
                      interpret_count(results['neg_binom']['coef_children_male'],
                                      results['neg_binom']['p_children_male'],
                                      results['neg_binom']['irr_children_male']))
    # Logit
    desc_lines.append("Logit (any_affair): children effect (females): " +
                      interpret_binary(results['logit']['coef_children'],
                                       results['logit']['p_children'],
                                       results['logit']['or_children']))
    desc_lines.append("Logit (any_affair): children effect (males): " +
                      interpret_binary(results['logit']['coef_children_male'],
                                       results['logit']['p_children_male'],
                                       results['logit']['or_children_male']))

    description = " | ".join(desc_lines)

    return {"object": results, "description": description}