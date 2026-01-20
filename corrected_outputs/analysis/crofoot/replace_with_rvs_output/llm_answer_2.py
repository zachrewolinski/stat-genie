import numpy as np


def extract_final_answer(model_output):
    """
    Extract coefficients, clustered SEs, p-values, 95% CIs, and odds ratios for the
    predictors of interest from the model output object returned by the provided
    modeling function.

    Returns a dictionary with keys:
      - "object": dict mapping predictor names to numeric summaries (coef, se, p,
                  95% CI on log-odds, odds ratio and its 95% CI)
      - "description": plain-language summary of what the results imply for the
                       effect of relative group size, location advantage, and
                       their interaction on probability of the focal group winning.

    Expects model_output to expose at least these attributes:
      - params (pd.Series indexed by parameter names, including 'const')
      - bse (pd.Series of clustered standard errors)
      - pvalues (pd.Series of p-values computed from clustered SEs)
    """
    # Predictors of interest
    predictors = ['RelSize_z', 'LocAdv_z', 'RelSize_x_LocAdv']

    # Try to access required attributes; raise informative error if missing
    missing_attrs = [a for a in ('params', 'bse', 'pvalues') if not hasattr(model_output, a)]
    if missing_attrs:
        raise AttributeError(f"model_output is missing required attributes: {missing_attrs}")

    params = model_output.params
    bse = model_output.bse
    pvalues = model_output.pvalues

    # Verify predictors exist in params
    missing_preds = [p for p in predictors if p not in params.index]
    if missing_preds:
        raise KeyError(f"The following predictors are not present in the model params: {missing_preds}")

    zcrit = 1.959963984540054  # ~1.96 for 95% CI

    summary = {}
    for p in predictors:
        coef = float(params[p])
        # bse and pvalues might be pd.Series; check membership safely
        try:
            se = float(bse[p]) if p in getattr(bse, "index", bse) else float(np.nan)
        except Exception:
            # Fallback: try indexing directly, else NaN
            try:
                se = float(bse[p])
            except Exception:
                se = float(np.nan)
        try:
            pval = float(pvalues[p]) if p in getattr(pvalues, "index", pvalues) else float(np.nan)
        except Exception:
            try:
                pval = float(pvalues[p])
            except Exception:
                pval = float(np.nan)

        ci_lower = coef - zcrit * se
        ci_upper = coef + zcrit * se
        # Odds ratio and its CI (exponentiate the log-odds)
        or_est = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower))
        or_ci_upper = float(np.exp(ci_upper))

        summary[p] = {
            'coef_log_odds': coef,
            'se': se,
            'p_value': pval,
            '95CI_log_odds': (ci_lower, ci_upper),
            'odds_ratio': or_est,
            '95CI_odds_ratio': (or_ci_lower, or_ci_upper),
        }

    # Interpret interaction significance
    interaction_p = summary['RelSize_x_LocAdv']['p_value']
    interaction_significant = (not np.isnan(interaction_p)) and (interaction_p < 0.05)

    # Build plain-language description
    lines = []
    # Relative size
    rs = summary['RelSize_z']
    lines.append(
        f"Relative group size (RelSize_z): coef={rs['coef_log_odds']:.3f}, se={rs['se']:.3f}, "
        f"p={rs['p_value']:.3f}. Odds ratio={rs['odds_ratio']:.3f} "
        f"(95% CI {rs['95CI_odds_ratio'][0]:.3f}–{rs['95CI_odds_ratio'][1]:.3f})."
    )
    lines.append(
        "Interpretation: A positive coef (OR>1) means that when the focal group is larger "
        "relative to the other group (1 SD increase in RelSize_z), the odds that the focal "
        "group wins increase; a negative coef (OR<1) means the opposite."
    )

    # Location advantage
    la = summary['LocAdv_z']
    lines.append(
        f"Location advantage (LocAdv_z): coef={la['coef_log_odds']:.3f}, se={la['se']:.3f}, "
        f"p={la['p_value']:.3f}. Odds ratio={la['odds_ratio']:.3f} "
        f"(95% CI {la['95CI_odds_ratio'][0]:.3f}–{la['95CI_odds_ratio'][1]:.3f})."
    )
    lines.append(
        "Interpretation: A positive coef (OR>1) means that when the focal group has a local "
        "location advantage (closer to its home-range center than the other group), its odds "
        "of winning increase."
    )

    # Interaction
    inter = summary['RelSize_x_LocAdv']
    lines.append(
        f"Interaction (RelSize_x_LocAdv): coef={inter['coef_log_odds']:.3f}, se={inter['se']:.3f}, "
        f"p={inter['p_value']:.3f}. Odds ratio={inter['odds_ratio']:.3f} "
        f"(95% CI {inter['95CI_odds_ratio'][0]:.3f}–{inter['95CI_odds_ratio'][1]:.3f})."
    )
    if interaction_significant:
        lines.append(
            "Interpretation: The interaction is statistically significant (p < 0.05), which means "
            "the effect of relative group size on win probability depends on contest location advantage. "
            "Specifically, the log-odds effect of RelSize_z changes by the interaction coefficient for each "
            "one-SD increase in LocAdv_z (and vice versa). Consider plotting or computing simple slopes "
            "to show the effect of RelSize_z at different levels of LocAdv_z."
        )
    else:
        lines.append(
            "Interpretation: The interaction is not statistically significant (p >= 0.05), so there is "
            "no strong evidence that the effect of relative group size on win probability depends on "
            "contest location advantage; main effects can be interpreted as the average effects."
        )

    description = " ".join(lines)

    return {
        "object": summary,
        "description": description
    }