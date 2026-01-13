def extract_final_answer(model_output):
    """
    Extract statistics needed to answer whether Homo sapiens have higher AMTL frequencies
    than non-human genera (Pan, Pongo, Papio), controlling for covariates in the fitted GLM.

    Returns a dict with keys:
      - "object": A dict containing pairwise comparisons (Homo sapiens vs each non-human genus)
                  with log-odds differences, SE, z, two-sided p-value, odds ratio, and 95% CI.
                  Also includes a simple summary conclusion for each comparison.
      - "description": A short explanation of what the object contains and how to interpret it.

    The function is robust to:
      - model_output is None (returns a clear message),
      - absence of some genus levels in the fitted model (reports missing levels),
      - models that use a reference (baseline) genus (handles implicit zero parameter).
    """
    import re
    import math
    from collections import OrderedDict

    try:
        from scipy import stats
        norm_cdf = stats.norm.cdf
    except Exception:
        # fallback to math.erf-based approximation if scipy is not available
        def norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Handle null model output
    if model_output is None:
        return {
            "object": None,
            "description": "Model output is None; the model did not fit or returned None. Cannot extract statistics or answer the question."
        }

    # Try to extract required pieces from the statsmodels results object
    try:
        params = model_output.params          # pandas Series of parameter estimates
        pvalues = model_output.pvalues        # pandas Series of p-values (not used for contrasts directly)
        cov = model_output.cov_params()       # DataFrame of covariance matrix
        conf_int_df = None
        try:
            conf_int_df = model_output.conf_int()  # two-column DataFrame
        except Exception:
            conf_int_df = None
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract params/pvalues/covariance from model_output: {e}"
        }

    # Identify genus levels present in the model/data
    # Preferred: try to read the original data frame used in the model to get actual levels
    genus_levels = None
    try:
        df_model = None
        if hasattr(model_output, 'model') and hasattr(model_output.model, 'data'):
            # statsmodels <-> different versions: try multiple ways to access the frame
            data_container = model_output.model.data
            if hasattr(data_container, 'frame') and data_container.frame is not None:
                df_model = data_container.frame
            elif hasattr(data_container, 'orig_endog') and hasattr(data_container, 'orig_exog'):
                # as fallback try to reconstruct, but this is less reliable
                # We'll not attempt heavy reconstruction here.
                df_model = None

        if df_model is not None and 'Genus' in df_model.columns:
            genus_levels = list(pd.Series(df_model['Genus']).dropna().unique())
    except Exception:
        genus_levels = None

    # If we couldn't get levels from data, parse parameter names for C(Genus)[T.<level>]
    import pandas as pd
    param_names = list(params.index.astype(str))
    genus_param_pattern = re.compile(r"C\(Genus\)\[T\.(.+)\]")
    parsed_levels = []
    param_name_for_level = {}  # map level -> param name
    for nm in param_names:
        m = genus_param_pattern.search(nm)
        if m:
            lvl = m.group(1)
            parsed_levels.append(lvl)
            param_name_for_level[lvl] = nm

    # If genus_levels still None, attempt to assemble from parsed_levels and known target genera
    target_genera = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']
    if genus_levels is None:
        # Use parsed_levels plus any of the target_genera that appear in parsed_levels
        genus_levels = list(dict.fromkeys(parsed_levels))  # preserve order, unique
        # If parsed levels are present but one of the target genera is missing, we may still infer
        # that missing one is the reference level (if target genera set - parsed_levels == 1).
        for tg in target_genera:
            if tg in genus_levels and tg not in param_name_for_level:
                # weird but continue
                pass

    # Determine reference (baseline) genus: one present in data but not among parameter T.<level> names
    # If we have genus_levels from data, find the level that does NOT appear in parsed_levels -> reference.
    reference = None
    try:
        if genus_levels is not None and isinstance(genus_levels, list) and len(genus_levels) > 0:
            # Clean potential whitespace and string forms
            gen_clean = [str(x).strip() for x in genus_levels]
            parsed_clean = [str(x).strip() for x in parsed_levels]
            # reference levels are those in gen_clean not in parsed_clean
            refs = [g for g in gen_clean if g not in parsed_clean]
            if len(refs) == 1:
                reference = refs[0]
            elif len(refs) > 1:
                # multiple potential references (unlikely) -> pick first but note ambiguity
                reference = refs[0]
            else:
                # No clear reference found (maybe parsed contains all levels if statsmodels used treatment coding
                # with k-1 parameters and we don't have the full list), leave reference=None
                reference = None
    except Exception:
        reference = None

    # Helper to get parameter estimate for a genus level (account for reference level having implicit 0)
    def get_param_for_level(level):
        """
        Returns (estimate, var, param_name) for the given genus level.
        If level is the model reference, returns (0.0, 0.0, None).
        If level has a parameter, returns its estimate and variance from cov matrix.
        If parameter not found, raises KeyError.
        """
        # clean input
        level = str(level).strip()
        # If this level is the reference
        if reference is not None and level == reference:
            return 0.0, 0.0, None
        # find parameter name
        # try exact match to parsed pattern
        pname = None
        # First try the param_name_for_level mapping (exact)
        if level in param_name_for_level:
            pname = param_name_for_level[level]
        else:
            # Try to find any param name that ends with ".<level>]" or contains the level string
            for nm in param_names:
                if f"C(Genus)[T.{level}]" == nm:
                    pname = nm
                    break
            if pname is None:
                # attempt more flexible matching (case-insensitive, spaces)
                for nm in param_names:
                    # remove prefix and bracket formatting to locate level text
                    mm = genus_param_pattern.search(nm)
                    if mm:
                        lvl = mm.group(1).strip()
                        if lvl.lower() == level.lower():
                            pname = nm
                            break
        if pname is None:
            # As a last resort, raise error to indicate missing parameter
            raise KeyError(f"No parameter found for genus level '{level}'. Parsed param names: {parsed_levels}")
        est = float(params[pname])
        var = float(cov.loc[pname, pname]) if (pname in cov.index and pname in cov.columns) else None
        return est, var, pname

    # Build comparisons for Homo sapiens vs each non-human genus
    comparisons = []
    missing_levels = []
    for other in ['Pan', 'Pongo', 'Papio']:
        comp = {}
        homo = 'Homo sapiens'
        comp['comparison'] = f"{homo} vs {other}"
        # Check presence in the inferred genus_levels
        present_homo = False
        present_other = False
        # determine presence: if genus_levels from data exists, check membership, otherwise check parsed_levels
        if genus_levels is not None and len(genus_levels) > 0:
            present_homo = any(str(x).strip().lower() == homo.lower() for x in genus_levels)
            present_other = any(str(x).strip().lower() == other.lower() for x in genus_levels)
        else:
            present_homo = any(str(x).strip().lower() == homo.lower() for x in parsed_levels)
            present_other = any(str(x).strip().lower() == other.lower() for x in parsed_levels)

        if not present_homo or not present_other:
            comp['error'] = ("One or both genera not present in the model data/parameters. "
                             f"Present in data? Homo: {present_homo}, {other}: {present_other}.")
            comparisons.append(comp)
            missing_levels.append((homo, other))
            continue

        # Get parameters (estimates and variances). For the reference level, get_param_for_level returns 0,0.
        try:
            est_h, var_h, pname_h = get_param_for_level(homo)
            est_o, var_o, pname_o = get_param_for_level(other)
        except KeyError as ke:
            comp['error'] = str(ke)
            comparisons.append(comp)
            missing_levels.append((homo, other))
            continue

        # Compute difference d = est_h - est_o (log-odds difference). Var(d) = var_h + var_o - 2*cov(h,o)
        # Handle cases where var is None (cov matrix missing entries) gracefully.
        if var_h is None or var_o is None:
            comp['error'] = "Could not obtain variance for one or both parameters from covariance matrix."
            comparisons.append(comp)
            continue

        # get covariance; if one param is None (reference) its cov is 0
        cov_ho = 0.0
        if pname_h is not None and pname_o is not None:
            # both have param names
            if pname_h in cov.index and pname_o in cov.columns:
                cov_ho = float(cov.loc[pname_h, pname_o])
            else:
                cov_ho = 0.0
        else:
            # if one is reference, cov with reference is zero
            cov_ho = 0.0

        diff = est_h - est_o
        var_diff = var_h + var_o - 2.0 * cov_ho
        if var_diff < 0 and abs(var_diff) < 1e-12:
            # numerical tiny negative -> set to zero
            var_diff = 0.0
        se_diff = math.sqrt(var_diff) if var_diff >= 0 else float('nan')

        # compute z and two-sided p-value using normal approximation
        if se_diff == 0 or math.isnan(se_diff):
            z = float('nan')
            p_two = float('nan')
        else:
            z = diff / se_diff
            p_two = 2.0 * (1.0 - norm_cdf(abs(z)))

        # odds ratio and 95% CI on odds ratio scale
        try:
            oratio = math.exp(diff)
            if not math.isnan(se_diff):
                ci_low = math.exp(diff - 1.96 * se_diff)
                ci_high = math.exp(diff + 1.96 * se_diff)
            else:
                ci_low = float('nan')
                ci_high = float('nan')
        except OverflowError:
            oratio = float('inf') if diff > 0 else 0.0
            ci_low = float('inf') if diff > 0 else 0.0
            ci_high = float('inf') if diff > 0 else 0.0

        comp.update({
            'log_odds_diff (Homo - Other)': diff,
            'se_diff': se_diff,
            'z': z,
            'p_value_two_sided': p_two,
            'odds_ratio (exp(diff))': oratio,
            'odds_ratio_95CI': (ci_low, ci_high),
            'interpretation': None
        })

        # Simple interpretation: if p < 0.05 then difference is statistically significant.
        if isinstance(p_two, float) and (not math.isnan(p_two)):
            if p_two < 0.05:
                if oratio > 1.0:
                    interp = f"Homo sapiens has significantly higher odds of AMTL than {other} (p={p_two:.3g}, OR={oratio:.3g})."
                else:
                    interp = f"Homo sapiens has significantly lower odds of AMTL than {other} (p={p_two:.3g}, OR={oratio:.3g})."
            else:
                interp = f"No statistically significant difference between Homo sapiens and {other} (p={p_two:.3g}, OR={oratio:.3g})."
        else:
            interp = "Could not compute statistical test (missing SE or numeric issue)."

        comp['interpretation'] = interp
        comparisons.append(comp)

    # Build a short summary conclusion across the three non-human genera:
    summary_statements = []
    for c in comparisons:
        if 'interpretation' in c and c['interpretation'] is not None:
            summary_statements.append((c['comparison'], c['interpretation']))
        else:
            summary_statements.append((c.get('comparison', '?'), c.get('error', 'No result')))

    object_out = {
        'pairwise_comparisons': comparisons,
        'summary': summary_statements,
        'missing_levels': missing_levels,
        'model_params_snapshot': params.to_dict() if hasattr(params, 'to_dict') else str(params)
    }

    description = (
        "The 'object' contains pairwise contrasts (log-odds differences) comparing Homo sapiens "
        "to each non-human genus (Pan, Pongo, Papio). For each comparison you get the log-odds "
        "difference (Homo - other), its standard error, z-statistic, two-sided p-value, odds ratio "
        "and a 95% CI for the odds ratio, plus a brief interpretation. If the model output is missing "
        "or a genus is not present in the fitted model, that is reported under 'missing_levels' or in "
        "the per-comparison 'error' field."
    )

    return {"object": object_out, "description": description}