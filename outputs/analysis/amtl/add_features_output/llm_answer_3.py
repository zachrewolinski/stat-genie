def extract_final_answer(model_output):
    """
    Extract contrasts comparing Homo sapiens to each non-human genus from a fitted
    statsmodels GLM (binomial logit) stored in model_output['glm_results'].

    Returns a dict with:
      - "object": {
            "baseline": <baseline genus or None if unknown>,
            "homo_level": <detected Homo sapiens level name>,
            "contrasts": [
                {
                  "other": <other genus name>,
                  "log_odds_diff": float (Homo - other),
                  "se": float,
                  "z": float,
                  "p": float,
                  "odds_ratio": float,
                  "ci_95_or": (lower, upper),
                  "interpretation": one of
                       "Homo > other (p<0.05)",
                       "Homo < other (p<0.05)",
                       "no significant difference (p>=0.05)"
                }, ...
            ],
            "overall_higher_than_all_nonhuman": bool
        }
      - "description": short textual explanation of what was computed and how to interpret.

    Notes:
      - Assumes a logit link (default for Binomial GLM), so coefficients are log-odds.
      - Contrasts are computed as log-odds(Homo) - log-odds(other). Positive => higher AMTL odds.
      - Uses the model covariance matrix to compute standard errors for contrasts.
    """
    import re
    import math
    import numpy as np
    from scipy import stats

    # Validate input
    if not isinstance(model_output, dict) or 'glm_results' not in model_output:
        raise ValueError("model_output must be a dict containing 'glm_results' (a fitted statsmodels GLM).")

    res = model_output['glm_results']

    # Try to use a robust covariance matrix if provided in the model_output
    cov = None
    if model_output.get('robust_results') is not None:
        try:
            cov = model_output['robust_results'].cov_params()
        except Exception:
            cov = None
    if cov is None:
        cov = res.cov_params()

    params = res.params  # Series
    param_names = list(params.index)

    # Find genotype-related parameter names (C(genus)[T.<level>])
    genus_param_pattern = re.compile(r"^C\(genus\)\[T\.(.+)\]$")
    genus_params = {}
    for pn in param_names:
        m = genus_param_pattern.match(pn)
        if m:
            level = m.group(1)
            genus_params[level] = pn

    # Try to recover the observed genus levels and baseline from the model's data frame
    baseline = None
    observed_levels = None
    try:
        df = res.model.data.frame  # may exist
        if 'genus' in df.columns:
            observed_levels = list(pd.unique(df['genus'])) if 'pd' in globals() else list(df['genus'].unique())
            # baseline is any observed level not appearing in genus_params
            missing = [lev for lev in observed_levels if lev not in genus_params]
            if len(missing) == 1:
                baseline = missing[0]
            else:
                # multiple or zero missing => baseline unknown or model encoding different
                baseline = None
    except Exception:
        # If the model's data frame is not available, leave baseline unknown
        baseline = None

    # Identify which level corresponds to Homo sapiens
    homo_level = None
    # Prefer exact match
    for level in genus_params.keys():
        if level == 'Homo sapiens':
            homo_level = level
            break
    # Fallback: case-insensitive contains 'homo'
    if homo_level is None:
        for level in genus_params.keys():
            if 'homo' in level.lower():
                homo_level = level
                break
    # If Homo is baseline (no param for Homo), and observed_levels known, detect that
    if homo_level is None and observed_levels is not None:
        for lev in observed_levels:
            if lev == 'Homo sapiens' or 'homo' in str(lev).lower():
                homo_level = lev
                break
        # If homo_level is in observed_levels but not in genus_params, that implies Homo is baseline.

    if homo_level is None:
        # Could not find Homo sapiens level among parameter names or observed levels
        return {
            "object": None,
            "description": "Could not identify a 'Homo sapiens' genus level in the fitted model's genus factor. "
                           "No pairwise contrasts computed."
        }

    # Build list of other non-human genera to compare against
    other_levels = []
    if observed_levels is not None:
        other_levels = [lev for lev in observed_levels if str(lev) != str(homo_level)]
    else:
        # If observed_levels not available, infer others from genus_params keys plus baseline if possible
        inferred = list(genus_params.keys())
        if baseline is not None and baseline not in inferred:
            inferred = inferred + [baseline]
        other_levels = [lev for lev in inferred if lev != homo_level]

    contrasts = []
    # For numeric computations, ensure we can reference parameter names in cov and params
    for other in other_levels:
        # Determine parameter names (if any) for homo and other
        pn_homo = genus_params.get(homo_level, None)
        pn_other = genus_params.get(other, None)

        # Compute log-odds difference D = logit(Homo) - logit(Other)
        # Cases:
        # 1) both have params (both non-baseline): D = coef_homo - coef_other
        # 2) homo has param, other does not (other is baseline): D = coef_homo - 0
        # 3) homo has no param (homo is baseline), other has param: D = 0 - coef_other
        # 4) neither has param: ambiguous -> skip
        if (pn_homo is None) and (pn_other is None):
            # ambiguous; cannot compute
            continue

        if pn_homo is not None and pn_other is not None:
            coef_diff = float(params[pn_homo] - params[pn_other])
            var = float(cov.loc[pn_homo, pn_homo] + cov.loc[pn_other, pn_other] - 2.0 * cov.loc[pn_homo, pn_other])
        elif pn_homo is not None and pn_other is None:
            coef_diff = float(params[pn_homo])
            var = float(cov.loc[pn_homo, pn_homo])
        elif pn_homo is None and pn_other is not None:
            coef_diff = float(-params[pn_other])
            var = float(cov.loc[pn_other, pn_other])
        else:
            continue  # defensive

        se = math.sqrt(var) if var >= 0 else float('nan')
        z = coef_diff / se if se and not math.isnan(se) else float('nan')
        p = 2.0 * float(stats.norm.sf(abs(z))) if not math.isnan(z) else float('nan')
        or_est = math.exp(coef_diff)
        # 95% CI on log-odds then exponentiate
        ci_log_lower = coef_diff - 1.96 * se
        ci_log_upper = coef_diff + 1.96 * se
        ci_or = (math.exp(ci_log_lower), math.exp(ci_log_upper))

        if (not math.isnan(p)) and (p < 0.05) and (coef_diff > 0):
            interp = f"Homo > {other} (p={p:.3g})"
        elif (not math.isnan(p)) and (p < 0.05) and (coef_diff < 0):
            interp = f"Homo < {other} (p={p:.3g})"
        else:
            interp = f"No significant difference vs {other} (p={p:.3g})"

        contrasts.append({
            "other": other,
            "log_odds_diff": coef_diff,
            "se": se,
            "z": z,
            "p": p,
            "odds_ratio": or_est,
            "ci_95_or": ci_or,
            "interpretation": interp
        })

    # Determine overall verdict: is Homo significantly higher than every non-human genus?
    overall_higher = True
    any_comparisons = False
    for c in contrasts:
        any_comparisons = True
        if not (c["p"] < 0.05 and c["log_odds_diff"] > 0):
            overall_higher = False
            break

    if not any_comparisons:
        return {
            "object": None,
            "description": "No valid pairwise genus contrasts could be computed (insufficient parameterization)."
        }

    conclusion_text = ("Homo sapiens shows significantly higher AMTL odds than all listed non-human genera "
                       "after adjustment." if overall_higher else
                       "Homo sapiens is not significantly higher than all non-human genera after adjustment "
                       "(see pairwise contrasts for details).")

    result_object = {
        "baseline": baseline,
        "homo_level": homo_level,
        "contrasts": contrasts,
        "overall_higher_than_all_nonhuman": bool(overall_higher)
    }

    description = ("Computed pairwise adjusted contrasts (log-odds differences and odds ratios) comparing "
                   "Homo sapiens to each non-human genus using the fitted binomial-logit GLM. "
                   "Positive log-odds_diff (and OR>1) indicate higher odds of AMTL in Homo; p-values test "
                   "whether the contrast differs from zero. " + conclusion_text)

    return {"object": result_object, "description": description}